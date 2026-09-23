"""A delegated run must not dispatch connectors the parent had to get approved.

``AgentTool`` builds its child a fresh execution context that inherits no
``task_source``, so a gate registered for the parent's source sees an unbound
source on the child's calls and passes them straight through -- an approval
bypass the model itself can reach, just by calling the ``agent`` tool with a
published agent whose persisted selection contains ``mcp:<server>``.

Inheriting the source instead would not fix it: the gate would pause the
child, and a paused child is classified as an unsupported nested interaction
(``agent_tool._classify_delegated_child_failure``), so the approval could
never be resumed and the host would keep a dangling prompt. The connectors are
therefore refused for the child, and only while the parent's own source
actually has a registration.
"""

from __future__ import annotations

from typing import Any

import pytest

from xagent.core.tools.adapters.vibe.agent_tool import _nested_mcp_refusal_reason
from xagent.core.tools.adapters.vibe.config import (
    NESTED_DELEGATION_NOT_APPROVABLE_REASON,
    _PUBLIC_MCP_UNAVAILABLE_REASONS,
)
from xagent.core.tools.adapters.vibe.mcp_approval_gate import (
    GateDecision,
    ToolCallExecutionContext,
    bind_tool_call_execution_context,
    has_approval_gate_for_source,
    register_mcp_approval_gate,
    unregister_mcp_approval_gate,
)
from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec
from xagent.web.tools.config import WebToolConfig


def _parent_context(task_source: str | None) -> ToolCallExecutionContext:
    """The identity ReAct binds around the parent's ``agent`` tool call."""

    return ToolCallExecutionContext(
        task_source=task_source,
        task_id="42",
        run_id="run-1",
        turn_id="turn-1",
        tool_call_id="call-agent-tool",
        pattern="react",
        react_step_id="step-1",
    )


@pytest.fixture
def slack_gate() -> Any:
    async def gate(_call: Any) -> GateDecision:
        return GateDecision.require_approval(interaction_id="i-1")

    async def resume(**_: Any) -> None:
        raise AssertionError("resume must not run")

    handle = register_mcp_approval_gate(
        task_source="toby-slack", gate=gate, resume=resume
    )
    try:
        yield handle
    finally:
        unregister_mcp_approval_gate(handle)


class _MinimalRequest:
    """Mirrors the stand-in request ``AgentTool`` builds for a child."""

    def __init__(self, user_id: int) -> None:
        self.user = type("obj", (), {"id": user_id})()


def _child_config(categories: list[str] | None = None, **kwargs: Any) -> WebToolConfig:
    """A child config shaped like the one ``AgentTool`` builds."""

    return WebToolConfig(
        db=None,
        request=_MinimalRequest(7),
        user_id=7,
        tool_selection_spec=ToolSelectionSpec.from_raw(
            tool_categories=categories or ["mcp:linkedin", "mcp:github"],
            exclude_custom_api_when_unconfigured=True,
        ),
        include_mcp_tools=True,
        **kwargs,
    )


def test_the_refusal_reason_is_publicly_reportable() -> None:
    """Otherwise ``UnavailableMCPTool`` drops it and the model learns nothing."""

    assert NESTED_DELEGATION_NOT_APPROVABLE_REASON in _PUBLIC_MCP_UNAVAILABLE_REASONS


def test_gate_predicate_tracks_registration_not_mere_presence(slack_gate: Any) -> None:
    assert has_approval_gate_for_source("toby-slack") is True
    # Another source with a live registration elsewhere is NOT gated: the
    # refusal must not widen to "any registration exists".
    assert has_approval_gate_for_source("internal") is False
    assert has_approval_gate_for_source(None) is False


@pytest.mark.parametrize(
    ("parent_source", "expected"),
    [
        # The bypass case: the parent's own source is gated.
        ("toby-slack", NESTED_DELEGATION_NOT_APPROVABLE_REASON),
        # A different source, even while a gate is live elsewhere. The refusal
        # must not widen to "any registration exists" -- that would strip MCP
        # from every delegation in the process the moment one tenant enrolled.
        ("internal", None),
        # An explicit but unregistered source.
        ("sdk", None),
        # No parent identity bound at all (a non-ReAct caller).
        (None, None),
    ],
)
def test_agent_tool_refuses_connectors_only_for_a_gated_parent(
    slack_gate: Any, parent_source: str | None, expected: str | None
) -> None:
    """The decision ``AgentTool`` actually makes, at the seam it makes it."""

    with bind_tool_call_execution_context(_parent_context(parent_source)):
        assert _nested_mcp_refusal_reason() == expected


def test_agent_tool_refuses_nothing_while_no_gate_is_registered() -> None:
    """With an empty registry the delegation path is byte-identical."""

    with bind_tool_call_execution_context(_parent_context("toby-slack")):
        assert _nested_mcp_refusal_reason() is None


def _stub_loader(config: WebToolConfig, calls: list[str]) -> None:
    """Replace the real DB loader so the ordinary path is observable."""

    async def _load() -> list[dict[str, Any]]:
        calls.append("loaded")
        return [{"name": "linkedin", "config": {"transport": "sse", "url": "u"}}]

    config._load_mcp_server_configs = _load  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_registered_parent_source_refuses_the_child_s_connectors(
    slack_gate: Any,
) -> None:
    """The bypass case: parent gated, so the child gets no live MCP tool."""

    calls: list[str] = []
    with bind_tool_call_execution_context(_parent_context("toby-slack")):
        assert has_approval_gate_for_source("toby-slack")
        config = _child_config(
            mcp_unavailable_reason=NESTED_DELEGATION_NOT_APPROVABLE_REASON
        )
        _stub_loader(config, calls)
        configs = await config.get_mcp_server_configs()

    # Nothing was loaded: the refusal short-circuits before any connector
    # config exists, so there is nothing downstream could dispatch.
    assert calls == []
    # Every selected server is still reported, and every one is refused.
    assert sorted(entry["name"] for entry in configs) == ["github", "linkedin"]
    for entry in configs:
        assert entry["config"] == {
            "unavailable": True,
            "reason": NESTED_DELEGATION_NOT_APPROVABLE_REASON,
        }
        assert "transport" not in entry["config"]


@pytest.mark.asyncio
async def test_unregistered_parent_source_leaves_the_child_untouched(
    slack_gate: Any,
) -> None:
    """The blast-radius case: a parent whose source is not gated is unchanged.

    ``AgentTool`` passes no reason at all here, so the config takes exactly
    the loader path it took before the refusal existed.
    """

    calls: list[str] = []
    with bind_tool_call_execution_context(_parent_context("internal")):
        assert not has_approval_gate_for_source("internal")
        config = _child_config()
        assert config._mcp_unavailable_reason is None
        _stub_loader(config, calls)
        configs = await config.get_mcp_server_configs()

    assert calls == ["loaded"]
    assert configs == [{"name": "linkedin", "config": {"transport": "sse", "url": "u"}}]


@pytest.mark.asyncio
async def test_an_unbound_parent_is_not_treated_as_gated(slack_gate: Any) -> None:
    """No bound identity (a non-ReAct caller) must not fail closed."""

    calls: list[str] = []
    assert has_approval_gate_for_source(None) is False
    config = _child_config()
    _stub_loader(config, calls)

    assert await config.get_mcp_server_configs() == [
        {"name": "linkedin", "config": {"transport": "sse", "url": "u"}}
    ]
    assert calls == ["loaded"]


@pytest.mark.asyncio
async def test_an_unrestricted_selection_still_materializes_nothing(
    slack_gate: Any,
) -> None:
    """``scoped_mcp_servers() is None`` has no names to report.

    The refusal still holds -- the explanation degrades, not the security
    property.
    """

    calls: list[str] = []
    config = _child_config(
        categories=["mcp"],
        mcp_unavailable_reason=NESTED_DELEGATION_NOT_APPROVABLE_REASON,
    )
    _stub_loader(config, calls)
    spec = config.get_tool_selection_spec()
    assert spec.scoped_mcp_servers() is None

    assert await config.get_mcp_server_configs() == []
    assert calls == []


def test_the_delegated_config_is_actually_wired_to_the_refusal() -> None:
    """The decision must reach the config that builds the child's tools.

    Asserted structurally rather than by running a delegation: constructing a
    real one needs a database, an LLM and a published agent row, none of which
    this property depends on. What it does depend on is that the one
    ``WebToolConfig`` ``AgentTool`` builds passes
    ``mcp_unavailable_reason=_nested_mcp_refusal_reason()`` -- drop that
    keyword and every behavioural test above still passes while the bypass is
    wide open again.
    """

    import ast
    import inspect

    from xagent.core.tools.adapters.vibe import agent_tool

    tree = ast.parse(inspect.getsource(agent_tool))
    wired = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "WebToolConfig"
    ]
    assert wired, "AgentTool no longer builds a WebToolConfig; update this test"

    for call in wired:
        reason = [kw for kw in call.keywords if kw.arg == "mcp_unavailable_reason"]
        assert reason, (
            "a delegated WebToolConfig with no mcp_unavailable_reason lets a "
            "gated parent's child materialize live MCP connectors"
        )
        value = reason[0].value
        assert isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
        assert value.func.id == "_nested_mcp_refusal_reason"
