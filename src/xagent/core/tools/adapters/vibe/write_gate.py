"""The seam a host uses to require approval before an MCP write executes.

A gated call is not executed. The hook is handed the call exactly as it was
about to run -- tool name and arguments -- and answers with either "run it"
or a pause carrying the frozen payload's identity. On approval the same
arguments are executed verbatim, because they were never handed back to the
model to be written a second time.

**Why a host-injected hook rather than a direct call into the web layer.**
This module is imported by the tool adapters, which
``sandboxed_tool/tool_runner.py`` reconstructs inside the sandbox for every
npx/uvx MCP tool -- a process where sqlalchemy is not installed. Anything
here that reached for a database would turn every sandboxed tool call into a
``ModuleNotFoundError``. So this file stays free of the web layer and the
host injects the policy instead.

Where the gate is *consulted* is a separate question, answered in
``write_gate_tool.py``: on the host, in ``WriteGateTool``, above both the
direct adapter and the sandbox wrapper. A sandboxed write is therefore gated
before anything crosses into the guest, which an earlier revision -- asking
from inside the adapter, where a supported npx/uvx connector only ever runs
in the guest -- got exactly backwards.

Not registering a hook *is* the off switch. No hook means no gate: nothing
to consult, nothing to record, and no behavior change at all -- including
the tool metadata the scheduler reads.

**Not a trust boundary.** The hook decides using, among other things, a
server's own ``readOnlyHint``/``destructiveHint`` annotations, which the MCP
spec says a client must not trust from an untrusted server. What this seam
guarantees is narrower and worth stating exactly: *if* a call is gated, the
arguments that eventually execute are the ones that were shown, byte for
byte. It does not guarantee that every dangerous call gets gated.

**Known limits of this version.** Both are deliberate, and neither is
described elsewhere in this module as if it were solved:

*An approval is not bound to a connector identity.* It is bound to the
interaction it was shown for, and nothing more. If a same-name MCP server is
repointed to another endpoint, reauthorized to a different account, or
deleted and recreated between the pause and the answer, the approved
arguments still execute -- against whatever that name resolves to when the
answer arrives. Carrying an immutable server/account identity would need the
MCP connection layer to expose one, which it does not today.

*Only a surface that delivers the chosen option value verbatim can grant an
approval.* The pause offers machine values (``approve``/``reject``), and the
host decides what counts as consent. xagent's own ``ClarificationForm``
cannot take part: it submits each answer's *display label* rather than the
option's value -- for every interaction type, not just ``confirm``, which it
renders as a localized yes/no switch that ignores ``options`` entirely. A
pause surfaced through that form therefore reaches the host as ``"Yes"`` and
is voided, not executed. The supported surface in this version is a host
that passes the value through untouched (Toby delivers the Slack button's
value as the resume message).

*Runtime-bound arguments are re-derived at execution, not frozen.* What is
frozen and replayed byte for byte is everything the model authored -- which
is also everything the approver was shown. On top of that the adapter
injects its runtime bindings' current values (``_runtime_tool_arguments``,
``_runtime_mcp_meta``), resolved from the connector runtime of whichever
adapter instance executes. A resume runs on a rebuilt instance, so if a
binding's source changed while the question waited, the value that goes out
is the new one. This is the connector-identity limit above, seen in the
argument dimension rather than the endpoint dimension.

Closing it needs one of two things that are deliberately not in this
change. Freezing the *prepared* payload means the guest can no longer
prepare it: ``sandboxed_tool/tool_runner.py`` re-enters ``run_json_async``
with the arguments it is handed, and preparation is not idempotent: the key
set ``_runtime_bound_tool_argument_names`` strips is exactly the key set
``_runtime_tool_arguments`` injects, so a prepared payload fed back through
that entry point has its frozen runtime values stripped as though the model
had set them and then replaced with current ones -- the round trip discards
precisely the half that was worth freezing. So it needs a second guest
entry point and a marker in the execution spec. Detecting the change instead
needs the *host* to store the binding values with the approval and compare
them at replay, which is a consumer this seam does not have; adding the
field without it would be another value nobody reads.
"""

from __future__ import annotations

from collections.abc import Mapping
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GatedCall:
    """One MCP call presented to the gate before it runs."""

    tool_name: str
    """The tool's runtime name, as the model called it."""

    server_name: str
    """Normalized identity of the MCP server the tool came from."""

    arguments: Mapping[str, Any]
    """The model-authored arguments, exactly as the model produced them.

    Not the payload that goes on the wire, and the difference is worth
    stating precisely because this seam's whole promise is about fidelity.
    Before a call executes, the adapter normalizes these against the tool's
    schema, applies the args model's defaults and coercion, strips any
    runtime-bound field the model tried to set, and injects the current
    values of its runtime bindings.

    The first three of those are pure functions of these arguments and the
    tool's schema, so they produce the same result at preview and at replay
    -- the replay re-enters the same public entry point with these exact
    arguments. The injected runtime values are not: see the third known
    limit in the module docstring.
    """

    write_hint: str
    """The server's own write declaration: an ``MCPWriteHint`` value.

    Carried as a plain string so this module stays independent of the
    adapter's enum. A hook must treat everything except ``"read_only"`` as a
    write: ``"undeclared"`` is the common case, not a promise of safety.
    """


@dataclass(frozen=True)
class GateDecision:
    """What the host decided about one gated call.

    ``interaction_id`` is the identity the frozen payload was stored under
    and the identity the resume callback will be handed back. It is the
    hook's to mint: the adapter neither generates nor interprets it, it only
    carries it into the pause so the two halves meet.
    """

    approval_required: bool
    interaction_id: str
    message: str = ""

    def __post_init__(self) -> None:
        # Required, not defaulted. An empty identity does not degrade the
        # pause, it loses the write: ReAct falls back to the raw
        # ``tool_call_id`` (see its pending-response construction), which the
        # host never recorded, so the answer arrives against an interaction
        # nobody is holding and the approved call never runs. That was a
        # silent default away, so the field has no default and an empty one
        # is refused here rather than three layers later.
        if self.approval_required and not self.interaction_id:
            raise ValueError("a pause must carry the interaction id it is stored under")


WriteGateHook = Callable[[GatedCall], Optional[GateDecision]]

# Given the interaction's identity, whether the user approved, and a callable
# that runs one frozen argument set, the host loads the payload it recorded,
# settles that approval exactly once, and returns the tool result. The
# executor is passed in rather than imported because only the tool knows how
# to place a call on its own connection -- and going through it is what keeps
# the replay on the tool's normal authorization and error-mapping path.
#
# **Precondition on the host.** This seam correlates a resume by the
# interaction id and the tool's runtime name, and neither is proof of
# ownership: the name is user-editable and an id is only as scoped as
# whoever minted it. So the host must verify that the interaction it loads
# belongs to the conversation, workspace and user the answer arrived from,
# before it hands the payload to the executor. Nothing here can do that
# check -- this module has no notion of a tenant.
WriteGateResumeHook = Callable[..., Any]

_HOOK: WriteGateHook | None = None
_RESUME_HOOK: WriteGateResumeHook | None = None


def set_write_gate_hook(hook: WriteGateHook | None) -> None:
    """Install (or clear) the process-wide approval hook.

    Idempotent and last-writer-wins, matching ``set_connector_runtime_resolver``.
    Passing ``None`` restores ungated execution.
    """
    global _HOOK
    _HOOK = hook


def get_write_gate_hook() -> WriteGateHook | None:
    """Return the installed hook, or ``None`` when nothing gates writes."""
    return _HOOK


def set_write_gate_resume_hook(hook: WriteGateResumeHook | None) -> None:
    """Install (or clear) the hook that settles an approved or rejected call."""
    global _RESUME_HOOK
    _RESUME_HOOK = hook


def get_write_gate_resume_hook() -> WriteGateResumeHook | None:
    """Return the installed resume hook, or ``None``."""
    return _RESUME_HOOK


def consult_write_gate(call: GatedCall) -> GateDecision | None:
    """Ask the installed hook about ``call``; ``None`` means "just run it".

    A hook that raises is treated as no decision and the call proceeds. That
    direction is deliberate and is the opposite of what a security boundary
    would do, for the reason in the module docstring: this seam makes an
    approved call faithful, it is not what keeps a dangerous call from
    running. Failing closed here would let a transient database error strand
    every connector call in a workspace behind an approval nobody can grant,
    which trades a bounded loss of gating for an unbounded loss of function.
    The host owns the decision to fail closed on its own side, where it can
    tell a policy miss from an outage.
    """
    hook = _HOOK
    if hook is None:
        return None
    try:
        decision = hook(call)
    except Exception:  # noqa: BLE001 - see the docstring
        logger.warning(
            "Write gate hook failed for %s; executing ungated",
            call.tool_name,
            exc_info=True,
        )
        return None
    if decision is None or isinstance(decision, GateDecision):
        return decision
    # Checked, not trusted. A hook written ``async def`` returns a coroutine
    # rather than raising, so the type error would surface as an
    # ``AttributeError`` on ``decision.approval_required`` at the call site
    # -- and it would do that for every gated call thereafter, since nothing
    # about the hook has changed. Treated as no decision, the same as a hook
    # that raised: this seam's documented direction on hook failure is to
    # execute rather than strand a workspace behind an unanswerable pause.
    logger.warning(
        "Write gate hook returned %s rather than a GateDecision for %s; "
        "executing ungated",
        type(decision).__name__,
        call.tool_name,
    )
    return None
