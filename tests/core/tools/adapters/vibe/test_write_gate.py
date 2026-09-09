"""A gated MCP write must not execute before it is approved -- on either transport.

The first version consulted the gate inside the adapter -- the one place a
supported npx/uvx connector never reaches (``write_gate_tool``'s module
docstring has the mechanism). Every sandboxed write went out ungated while
the host tests stayed green, because those tests called the adapter
directly.

So the tests here refuse a hand-rolled stand-in for that boundary. They drive
a real ``SandboxedToolWrapper`` and watch its guest dispatch (``sandbox.exec``
carrying ``--args-b64``), which means a gate that is absent for the sandbox
transport, or an approval whose arguments are re-derived on the way into the
guest, fails here rather than in production.
"""

import asyncio
import base64
import json
from typing import Any, Mapping, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.core.tools.adapters.sandboxed_tool.conftest import FakeBaseTool
from xagent.core.tools.adapters.vibe.mcp_adapter import MCPWriteHint
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandbox_config import sandbox_config
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    SandboxedToolWrapper,
)
from xagent.core.tools.adapters.vibe.write_gate import (
    GatedCall,
    GateDecision,
    set_write_gate_hook,
    set_write_gate_resume_hook,
)
from xagent.core.tools.adapters.vibe.write_gate_tool import (
    gate_mcp_tools,
    is_replaying_approved_call,
)
from xagent.core.tools.user_interaction import WAITING_FOR_USER_STATUS

# The arguments a human would have been shown. Deliberately not a flat
# {"text": "hi"}: nesting and a non-ASCII body are where a "re-serialize it
# again on the way out" bug shows up as a difference instead of a coincidence.
SHOWN_ARGUMENTS = {
    "channel": "C123",
    "blocks": [{"type": "section", "text": "内容 A"}],
    "unfurl": False,
}


@sandbox_config()
class _FakeMCPTool(FakeBaseTool):
    """One MCP tool adapter's surface, without a connection behind it.

    ``source_server``/``concurrency_safe``/``write_hint`` are the three
    attributes ``AbstractBaseTool.metadata`` reads off a concrete tool, so a
    wrapper that goes through ``metadata`` sees exactly what it would see
    from a real adapter -- and one that reaches for a raw attribute instead
    sees nothing once this is behind the sandbox wrapper.
    """

    source_server = "slack"
    concurrency_safe = True

    def __init__(self) -> None:
        self.direct_calls: list[dict[str, Any]] = []
        self.replay_flags: list[bool] = []

    @property
    def name(self) -> str:
        return "slack_post_message"

    @property
    def write_hint(self) -> MCPWriteHint:
        return MCPWriteHint.UNDECLARED

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        self.direct_calls.append(dict(args))
        self.replay_flags.append(is_replaying_approved_call())
        return {"success": True, "posted": dict(args)}


class _RecordingGate:
    """A host gate that always demands approval and remembers what it saw."""

    def __init__(self) -> None:
        self.calls: list[GatedCall] = []

    def __call__(self, call: GatedCall) -> Optional[GateDecision]:
        self.calls.append(call)
        return GateDecision(
            approval_required=True,
            interaction_id="interaction-1",
            message="Post this to Slack?",
        )


class _RecordingResume:
    """The host half: replays what the gate recorded, and only that.

    Takes the gate rather than a copy of the arguments. Reading them from a
    separate test constant would let a gate that recorded the *wrong*
    arguments still pass every fidelity assertion here -- the payload and
    the expectation would come from the same place, and the subject would
    never be consulted.
    """

    def __init__(self, gate: "_RecordingGate") -> None:
        self._gate = gate
        self.calls: list[tuple[str, str, str]] = []

    @property
    def frozen(self) -> dict[str, Any]:
        return dict(self._gate.calls[0].arguments)

    async def __call__(
        self,
        *,
        interaction_id: str,
        response: str,
        tool_name: str,
        executor: Any,
    ) -> Any:
        # The keyword set is the contract the saas side implements against.
        # Pinned by signature: a new kwarg on the caller breaks this here
        # rather than at runtime in the other repository.
        self.calls.append((interaction_id, response, tool_name))
        if response != "approve":
            return {"success": False, "status": "voided"}
        return await executor(self.frozen)


def _make_sandbox() -> MagicMock:
    """A sandbox that reports success without running anything."""
    payload = {"success": True, "output": "sent"}

    def _exec(*args: Any, **kwargs: Any) -> MagicMock:
        result = MagicMock()
        result.exit_code = 0
        result.stdout = json.dumps(payload) if args[0] == "cat" else ""
        result.stderr = ""
        return result

    sandbox = MagicMock()
    sandbox.name = "sandbox-test"
    sandbox.exec = AsyncMock(side_effect=_exec)
    sandbox.write_file = AsyncMock()
    return sandbox


def _guest_arguments(sandbox: MagicMock) -> list[dict[str, Any]]:
    """The argument payloads that actually crossed into the guest process.

    Read back out of the ``--args-b64`` the wrapper puts on the tool-runner
    command line, so this asserts on what the guest would decode -- not on
    what the host meant to send.
    """
    payloads = []
    for call in sandbox.exec.call_args_list:
        argv = list(call.args)
        if "--args-b64" not in argv:
            continue
        encoded = argv[argv.index("--args-b64") + 1]
        payloads.append(json.loads(base64.b64decode(encoded).decode("utf-8")))
    return payloads


@pytest.fixture(autouse=True)
def _clear_gate_hooks():
    """The hooks are process-global; a leak would silently gate other tests."""
    yield
    set_write_gate_hook(None)
    set_write_gate_resume_hook(None)


def _sandboxed_gated_tool() -> tuple[Any, _FakeMCPTool, MagicMock]:
    target = _FakeMCPTool()
    sandbox = _make_sandbox()
    wrapper = SandboxedToolWrapper(target, sandbox)
    (gated,) = gate_mcp_tools([wrapper])
    return gated, target, sandbox


async def test_sandboxed_write_does_not_reach_the_guest_before_approval():
    """The regression the whole redesign exists for.

    A gate inside the adapter cannot stop this call: by the time the adapter
    runs, it is running in the guest. Asserting on ``sandbox.exec`` is what
    makes that failure visible from the host side.
    """
    gate = _RecordingGate()
    set_write_gate_hook(gate)
    gated, target, sandbox = _sandboxed_gated_tool()

    result = await gated.run_json_async(SHOWN_ARGUMENTS)

    assert result["status"] == WAITING_FOR_USER_STATUS
    assert result["interaction_id"] == "interaction-1"
    assert _guest_arguments(sandbox) == []
    assert sandbox.exec.await_count == 0
    assert target.direct_calls == []
    assert len(gate.calls) == 1


async def test_approved_sandboxed_call_carries_the_shown_arguments_into_the_guest():
    """What executes is what was shown, across the wrapper/runner seam."""
    gate = _RecordingGate()
    set_write_gate_hook(gate)
    resume = _RecordingResume(gate)
    set_write_gate_resume_hook(resume)
    gated, _target, sandbox = _sandboxed_gated_tool()

    paused = await gated.run_json_async(SHOWN_ARGUMENTS)
    assert paused["status"] == WAITING_FOR_USER_STATUS

    await gated.resume_user_interaction(
        interaction_id=paused["interaction_id"], response="approve"
    )

    assert _guest_arguments(sandbox) == [SHOWN_ARGUMENTS]
    assert resume.calls == [("interaction-1", "approve", "slack_post_message")]


async def test_rejected_sandboxed_call_never_reaches_the_guest():
    gate = _RecordingGate()
    set_write_gate_hook(gate)
    set_write_gate_resume_hook(_RecordingResume(gate))
    gated, _target, sandbox = _sandboxed_gated_tool()

    paused = await gated.run_json_async(SHOWN_ARGUMENTS)
    settled = await gated.resume_user_interaction(
        interaction_id=paused["interaction_id"], response="reject"
    )

    assert settled["success"] is False
    assert _guest_arguments(sandbox) == []


async def test_the_gate_sees_the_server_name_through_the_sandbox_wrapper():
    """``SandboxedToolWrapper`` forwards ``metadata`` and nothing else.

    Reading ``target.source_server`` therefore reported no server at all for
    exactly the transport this wrapper exists to cover, leaving the host
    policy to decide about an anonymous call.
    """
    gate = _RecordingGate()
    set_write_gate_hook(gate)
    gated, _target, _sandbox = _sandboxed_gated_tool()

    await gated.run_json_async(SHOWN_ARGUMENTS)

    (call,) = gate.calls
    assert call.server_name == "slack"
    assert call.tool_name == "slack_post_message"
    assert call.write_hint == MCPWriteHint.UNDECLARED.value
    assert call.arguments == SHOWN_ARGUMENTS


async def test_approved_direct_call_replays_through_the_targets_own_entry_point():
    """Not past it.

    The replay re-enters ``run_json_async``, so the target's authorization
    check, delegated-credential refresh retry and connector-error mapping all
    still run for an approved call. A replay that reached for a lower-level
    call would skip every one of them, and the row is already spent.
    """
    gate = _RecordingGate()
    set_write_gate_hook(gate)
    resume = _RecordingResume(gate)
    set_write_gate_resume_hook(resume)
    target = _FakeMCPTool()
    (gated,) = gate_mcp_tools([target])

    paused = await gated.run_json_async(SHOWN_ARGUMENTS)
    assert target.direct_calls == []

    result = await gated.resume_user_interaction(
        interaction_id=paused["interaction_id"], response="approve"
    )

    assert target.direct_calls == [SHOWN_ARGUMENTS]
    assert result["posted"] == SHOWN_ARGUMENTS


async def test_replay_is_not_gated_a_second_time():
    """And the replay marker does not outlive the replay."""
    gate = _RecordingGate()
    set_write_gate_hook(gate)
    set_write_gate_resume_hook(_RecordingResume(gate))
    target = _FakeMCPTool()
    (gated,) = gate_mcp_tools([target])

    paused = await gated.run_json_async(SHOWN_ARGUMENTS)
    await gated.resume_user_interaction(
        interaction_id=paused["interaction_id"], response="approve"
    )

    assert len(gate.calls) == 1
    assert target.replay_flags == [True]
    assert is_replaying_approved_call() is False


async def test_a_gated_write_is_never_batched_with_another():
    """ReAct copies one answer to every interaction in a concurrent segment.

    One ``approve`` would then authorize every independently frozen write in
    the batch, including the ones the approver was never shown.
    """
    set_write_gate_hook(_RecordingGate())
    target = _FakeMCPTool()
    (gated,) = gate_mcp_tools([target])

    assert target.metadata.concurrency_safe is True
    assert gated.metadata.concurrency_safe is False
    # Everything else about the tool is unchanged: the model must not be able
    # to tell a gated tool from an ungated one.
    assert gated.metadata.model_dump(exclude={"concurrency_safe"}) == (
        target.metadata.model_dump(exclude={"concurrency_safe"})
    )


async def test_no_installed_gate_means_no_behavior_change_at_all():
    """The off switch is "install nothing", and it has to be total.

    Suppressing batching unconditionally would quietly cost every operator
    who opted into ``concurrency_safe`` on a connection their batching, in a
    process where nothing can pause a call in the first place.
    """
    target = _FakeMCPTool()
    (gated,) = gate_mcp_tools([target])

    assert gated.metadata == target.metadata

    result = await gated.run_json_async(SHOWN_ARGUMENTS)

    assert target.direct_calls == [SHOWN_ARGUMENTS]
    assert result["success"] is True


async def test_a_failing_gate_executes_ungated():
    """Deliberate, and the opposite of what a security boundary would do.

    This seam makes an approved call faithful; it is not what keeps a
    dangerous call from running. Failing closed here would strand every
    connector call in a workspace behind an approval nobody can grant.
    """

    def _broken_gate(call: GatedCall) -> Optional[GateDecision]:
        raise RuntimeError("policy lookup failed")

    set_write_gate_hook(_broken_gate)
    target = _FakeMCPTool()
    (gated,) = gate_mcp_tools([target])

    result = await gated.run_json_async(SHOWN_ARGUMENTS)

    assert result["success"] is True
    assert target.direct_calls == [SHOWN_ARGUMENTS]


async def test_an_approval_with_no_resume_hook_reports_an_error_instead_of_executing():
    set_write_gate_hook(_RecordingGate())
    target = _FakeMCPTool()
    (gated,) = gate_mcp_tools([target])

    paused = await gated.run_json_async(SHOWN_ARGUMENTS)
    settled = await gated.resume_user_interaction(
        interaction_id=paused["interaction_id"], response="approve"
    )

    assert settled["success"] is False
    assert settled["status"] == "error"
    assert target.direct_calls == []


def test_the_sync_entry_point_cannot_run_a_gated_write():
    """A gate is only a gate if there is no second door.

    The direct adapter refuses ``run_json_sync`` on its own (MCP tools are
    async-only), but ``SandboxedToolWrapper.run_json_sync`` works -- it just
    drives its async path through ``asyncio.run``. Forwarding here would run
    a sandboxed npx/uvx write with nothing consulted, through the one
    transport the wrapper exists to cover.
    """
    set_write_gate_hook(_RecordingGate())
    _gated, target, sandbox = _sandboxed_gated_tool()

    with pytest.raises(RuntimeError, match="async only"):
        _gated.run_json_sync(SHOWN_ARGUMENTS)

    assert _guest_arguments(sandbox) == []
    assert target.direct_calls == []


async def test_a_hook_that_returns_a_coroutine_executes_ungated():
    """The easy mistake, and it must not be a crash.

    Writing the hook ``async def`` returns a coroutine instead of raising,
    so a type error would surface as ``AttributeError`` on
    ``decision.approval_required`` -- and it would do that for every gated
    call afterwards, since nothing about the hook has changed. Treated as no
    decision, the same as a hook that raised.
    """

    async def _async_hook(call: GatedCall) -> Optional[GateDecision]:
        return GateDecision(approval_required=True, interaction_id="i-1")

    set_write_gate_hook(_async_hook)  # type: ignore[arg-type]
    target = _FakeMCPTool()
    (gated,) = gate_mcp_tools([target])

    result = await gated.run_json_async(SHOWN_ARGUMENTS)

    assert result["success"] is True
    assert target.direct_calls == [SHOWN_ARGUMENTS]


async def test_a_hook_that_returns_a_bare_string_executes_ungated():
    """Any non-decision, not just a coroutine."""
    set_write_gate_hook(lambda call: "approve")  # type: ignore[arg-type,return-value]
    target = _FakeMCPTool()
    (gated,) = gate_mcp_tools([target])

    assert (await gated.run_json_async(SHOWN_ARGUMENTS))["success"] is True
    assert target.direct_calls == [SHOWN_ARGUMENTS]


def test_a_pause_without_an_interaction_id_is_refused():
    """The identity is what makes the two halves meet.

    An empty one does not weaken the pause, it loses the write: ReAct falls
    back to the raw ``tool_call_id``, which the host never recorded, so the
    answer arrives against an interaction nobody holds. Refused where it is
    constructed rather than three layers later.
    """
    with pytest.raises(ValueError, match="interaction id"):
        GateDecision(approval_required=True, interaction_id="")

    # A decision that does not pause needs no identity.
    assert GateDecision(approval_required=False, interaction_id="").interaction_id == ""


def test_the_sync_entry_point_passes_through_with_no_hook_installed():
    """The off switch has to cover both entry points.

    With nothing installed there is no gate to bypass, so refusing here
    would contradict the module's own "no behavior change" claim. The target
    still refuses on its own if it is async-only, which every MCP adapter
    is.
    """
    target = _FakeMCPTool()
    (gated,) = gate_mcp_tools([target])

    assert gated.run_json_sync(SHOWN_ARGUMENTS) == {}


class _SelectiveGate:
    """Pauses one tool by name and lets everything else through."""

    def __init__(self, paused_tool: str) -> None:
        self._paused = paused_tool
        self.calls: list[GatedCall] = []

    def __call__(self, call: GatedCall) -> Optional[GateDecision]:
        self.calls.append(call)
        if call.tool_name != self._paused:
            return None
        return GateDecision(approval_required=True, interaction_id="interaction-1")


@sandbox_config()
class _SlowMCPTool(_FakeMCPTool):
    """Blocks inside ``run_json_async`` so two calls genuinely interleave."""

    def __init__(self, released: asyncio.Event, entered: asyncio.Event) -> None:
        super().__init__()
        self._released = released
        self._entered = entered

    @property
    def name(self) -> str:
        return "slack_post_message"

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        self.replay_flags.append(is_replaying_approved_call())
        self._entered.set()
        await self._released.wait()
        self.direct_calls.append(dict(args))
        return {"success": True}


@sandbox_config()
class _OtherMCPTool(_FakeMCPTool):
    @property
    def name(self) -> str:
        return "slack_list_channels"


async def test_the_replay_marker_does_not_leak_into_a_concurrent_call():
    """A ContextVar, not an attribute, and this is why.

    While one call is mid-replay, an ordinary call on another tool runs in a
    different task. If the marker were shared state, that second call would
    see itself as an approved replay and skip the gate entirely.
    """
    released = asyncio.Event()
    entered = asyncio.Event()
    gate = _SelectiveGate("slack_post_message")
    set_write_gate_hook(gate)
    slow = _SlowMCPTool(released, entered)
    other = _OtherMCPTool()
    gated_slow, gated_other = gate_mcp_tools([slow, other])
    set_write_gate_resume_hook(_RecordingResume(gate))  # type: ignore[arg-type]

    paused = await gated_slow.run_json_async(SHOWN_ARGUMENTS)
    assert paused["status"] == WAITING_FOR_USER_STATUS

    replay = asyncio.ensure_future(
        gated_slow.resume_user_interaction(
            interaction_id=paused["interaction_id"], response="approve"
        )
    )
    await entered.wait()

    # Mid-replay: an unrelated call must not inherit the marker.
    assert is_replaying_approved_call() is False
    await gated_other.run_json_async({"limit": 10})
    assert other.replay_flags == [False]

    released.set()
    await replay

    assert slow.replay_flags == [True]
    assert is_replaying_approved_call() is False
