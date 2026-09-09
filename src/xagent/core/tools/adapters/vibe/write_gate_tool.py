"""The host-side boundary where a connector write waits for approval.

**Why this is a wrapper and not a check inside the adapter.** A supported
npx/uvx connector does not execute in this process at all: the sandbox
serializes the adapter class and its constructor data, and
``sandboxed_tool/tool_runner.py`` rebuilds it with ``importlib`` and
``cloudpickle`` in a guest process, then calls ``run_json_async`` there.
Anything the host installs as module state -- a hook, a database session --
does not exist on that side, so a gate placed inside the adapter is simply
absent for the transport that most needs it. The first version of this
feature did exactly that and left every sandboxed write ungated.

Placed here, the gate wraps *whatever* the loader produced: a bare
``MCPToolAdapter`` for a direct connection, or a ``SandboxedToolWrapper``
around one for npx/uvx. Both look identical from outside, both are entered
in the host process, and neither can be reached without passing through
this object first.

The wrapper is also what makes resumption honest. It replays an approved
call by re-entering the target's own ``run_json_async`` with a flag that
suppresses re-gating, so the authorization check, ``UserContext``, the
delegated-credential refresh retry and the safe connector-error mapping all
still run -- rather than reaching past them into the low-level call, which
is what an earlier revision did.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from contextvars import ContextVar
from typing import Any, Optional, Type

from pydantic import BaseModel

from ...user_interaction import WAITING_FOR_USER_STATUS
from .base import AbstractBaseTool, ToolMetadata
from .write_gate import (
    GatedCall,
    consult_write_gate,
    get_write_gate_hook,
    get_write_gate_resume_hook,
)

logger = logging.getLogger(__name__)

# Set while a wrapper is replaying an approved call, so the nested
# ``run_json_async`` runs its normal authorization, retry and error handling
# without asking the gate a second question about a call the user already
# answered. A ContextVar rather than an attribute: the replay is awaited on
# the same task, and a flag on the instance would leak across concurrent
# calls to the same tool.
_REPLAYING: ContextVar[bool] = ContextVar("xagent_write_gate_replaying", default=False)


def is_replaying_approved_call() -> bool:
    """Whether the current task is replaying an already-approved call."""
    return _REPLAYING.get()


class WriteGateTool(AbstractBaseTool):
    """Holds a connector write until a human answers, then replays it verbatim.

    Delegates every descriptive surface to the wrapped tool, so the model and
    the tool-selection layers cannot tell the difference: what a gate changes
    is *when* a call runs, never what it looks like.
    """

    def __init__(self, target_tool: AbstractBaseTool) -> None:
        self._target = target_tool

    @property
    def target(self) -> AbstractBaseTool:
        """The wrapped tool, for callers that need the transport itself."""
        return self._target

    @property
    def name(self) -> str:
        return self._target.name

    @property
    def description(self) -> str:
        return self._target.description

    @property
    def tags(self) -> list[str]:
        return self._target.tags

    @property
    def metadata(self) -> ToolMetadata:
        """The target's metadata, except that a gated write is never batched.

        ReAct groups consecutive ``concurrency_safe`` calls into one segment
        and then delivers **one** answer string to every pending interaction
        in it. A single ``approve`` would therefore authorize every
        independently frozen write in the batch, and the approver never got
        to judge them separately.

        Reporting ``False`` is not a workaround for that scheduling detail;
        it is what the flag actually means here. The pattern's own contract
        (see its I5 note) makes ``concurrency_safe`` an idempotency promise
        as well as a concurrency one, and a call parked for human approval
        is not idempotent -- replaying it publishes twice. So a gated tool
        runs alone, one approval decides one write, and the answer cannot be
        spread across calls the user never saw.

        Scope: only a connection whose operator opted into
        ``concurrency_safe`` is affected, and only while a gate is actually
        installed. Reads on such a connection lose batching too --
        ``metadata`` is read before there is a call to ask the hook about, so
        it cannot know whether *this* invocation would pause. That is the
        conservative direction; the alternative is a batch whose single
        answer authorizes a write nobody was shown.
        """
        metadata = self._target.metadata
        if not metadata.concurrency_safe:
            return metadata
        if get_write_gate_hook() is None:
            # Nothing in this process can pause a call, so nothing can spread
            # one answer across a batch. Returning the target's own metadata
            # is what keeps the off switch total: an unregistered gate must
            # not silently cost an operator the batching they configured.
            return metadata
        return metadata.model_copy(update={"concurrency_safe": False})

    @property
    def is_sandboxed(self) -> bool:
        return bool(getattr(self._target, "is_sandboxed", False))

    def args_type(self) -> Type[BaseModel]:
        return self._target.args_type()

    def return_type(self) -> Type[BaseModel]:
        return self._target.return_type()

    def state_type(self) -> Optional[Type[BaseModel]]:
        return self._target.state_type()

    def return_value_as_string(self, value: Any) -> str:
        return self._target.return_value_as_string(value)

    def is_async(self) -> bool:
        return True

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """Refused: nothing on this path can pause for an answer.

        ``MCPToolAdapter.run_json_sync`` already raises -- MCP tools are
        async-only -- so the direct transport was never reachable this way.
        The sandbox wrapper's is different: it drives its own async path
        through ``asyncio.run`` and works. Delegating here would therefore
        run a sandboxed npx/uvx write with no gate consulted at all, through
        exactly the transport this wrapper exists to cover.

        Raising instead keeps "no execution without passing the gate" a
        property of the object rather than of one of its two entry points,
        and makes both transports refuse identically -- which is the point of
        wrapping them in the same place.
        """
        if get_write_gate_hook() is None:
            # Nothing in this process can pause a call, so there is no gate
            # to bypass -- and refusing here anyway would contradict the off
            # switch this module documents. The target still refuses on its
            # own if it is async-only, which every MCP adapter is.
            return self._target.run_json_sync(args)
        raise RuntimeError(
            f"MCP tool {self.name} is async only; please use run_json_async()"
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        """Ask the gate, then either run the call or park it for approval."""
        if is_replaying_approved_call():
            # Already answered. Fall through so the target performs its own
            # authorization, retry and error mapping on the replay.
            return await self._target.run_json_async(args)

        metadata = self.metadata
        decision = consult_write_gate(
            GatedCall(
                tool_name=self.name,
                # Both fields are read through ``metadata`` rather than off
                # the target. The adapter mirrors its normalized server
                # identity and its write declaration there, and
                # ``SandboxedToolWrapper`` delegates ``metadata`` while
                # forwarding no other attribute -- so reaching for
                # ``target.source_server`` reported no server at all for
                # exactly the npx/uvx transport this wrapper exists to cover.
                server_name=metadata.source_server or "",
                # A copy, not the caller's mapping. The host records this
                # as the payload it will replay, and a live reference would
                # let anything that mutates ``args`` afterwards rewrite what
                # was approved.
                arguments=dict(args),
                write_hint=self._write_hint_value(metadata),
            )
        )
        if decision is None or not decision.approval_required:
            return await self._target.run_json_async(args)

        return {
            "status": WAITING_FOR_USER_STATUS,
            "interaction_id": decision.interaction_id,
            "message": decision.message or f"Approve running {self.name}?",
            "interactions": [
                {
                    "type": "confirm",
                    "field": "approve",
                    "label": "Approve",
                    "options": [
                        {"label": "Approve", "value": "approve"},
                        {"label": "Reject", "value": "reject"},
                    ],
                }
            ],
        }

    async def resume_user_interaction(
        self,
        *,
        interaction_id: str,
        response: str,
    ) -> Any:
        """Replay the approved call, or void it.

        The model is not consulted: the whole point is that the arguments
        which execute are the ones that were shown. The replay re-enters the
        target's ``run_json_async`` under ``_REPLAYING``, so authorization,
        credential refresh and error mapping behave exactly as they do for
        an ungated call.
        """
        hook = get_write_gate_resume_hook()
        if hook is None:
            return {
                "success": False,
                "status": "error",
                "error": "This approval can no longer be completed.",
            }
        return await hook(
            interaction_id=interaction_id,
            response=response,
            tool_name=self.name,
            executor=self._replay,
        )

    async def _replay(self, arguments: Mapping[str, Any]) -> Any:
        """Run one frozen argument set through the target's normal path."""
        token = _REPLAYING.set(True)
        try:
            return await self._target.run_json_async(arguments)
        finally:
            _REPLAYING.reset(token)

    @property
    def category(self) -> Any:
        """Forwarded like the rest of the descriptive surface.

        Delegated explicitly rather than through ``__getattr__``: every name
        below is defined on ``AbstractBaseTool``, so normal lookup succeeds
        on this class and ``__getattr__`` is never consulted -- it would
        silently answer with the *wrapper's* base implementation instead of
        the target's. ``metadata`` is the case that makes this concrete: the
        base rebuilds it from attributes this wrapper does not have, which
        is how a wrapped tool once reported no source server at all.

        A blanket ``__getattr__`` would also forward ``__sandbox_config__``,
        which is *not* on the base -- and ``resolve_sandbox_config`` reading
        it through the wrapper would offer an already-sandboxed tool up for
        sandboxing a second time.
        """
        return getattr(self._target, "category", None)

    async def setup(self, task_id: Optional[str] = None) -> None:
        await self._target.setup(task_id)

    async def teardown(self, task_id: Optional[str] = None) -> None:
        await self._target.teardown(task_id)

    async def save_state_json(self) -> Mapping[str, Any]:
        return await self._target.save_state_json()

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        await self._target.load_state_json(state)

    def _write_hint_value(self, metadata: ToolMetadata) -> str:
        """The target's write declaration, as a plain string."""
        hint = metadata.mcp_write_hint
        return hint if isinstance(hint, str) else "undeclared"


def gate_mcp_tools(tools: "Iterable[AbstractBaseTool]") -> list[AbstractBaseTool]:
    """Wrap loaded MCP tools so a write cannot execute before approval.

    Applied once where the direct and sandboxed loaders converge, so neither
    transport can be gated by accident and neither can be missed. Wrapping is
    unconditional and cheap: whether a given call actually needs approval is
    the installed hook's decision, made per call, and with no hook installed
    every wrapper is a pass-through.
    """
    return [WriteGateTool(tool) for tool in tools]
