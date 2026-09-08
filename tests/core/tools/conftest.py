"""Shared helpers for the MCP tool-loading tests."""

from typing import Any

from xagent.core.tools.adapters.vibe.write_gate_tool import WriteGateTool


def gated_targets(tools: Any) -> tuple[Any, ...]:
    """The loader's tools, unwrapped from the approval gate it applies.

    Every loaded MCP tool leaves ``load_mcp_tools_as_agent_tools`` inside a
    ``WriteGateTool``: that function is the one place the direct and
    sandboxed transports converge, which is why the gate is applied there
    rather than inside the adapter -- a sandboxed npx/uvx connector rebuilds
    the adapter in a guest process where no host hook exists, so a gate
    placed there is absent for exactly the transport that most needs it.

    Asserting through this helper keeps each caller's test about the thing it
    was written for -- which transport ran, what a timeout did -- while still
    failing if a tool ever escapes the loader ungated.
    """
    assert all(isinstance(tool, WriteGateTool) for tool in tools), tools
    return tuple(tool.target for tool in tools)
