"""A healthy connector must not be reported as having returned no tools.

``_build_mcp_load_summary`` decided which servers loaded by reading
``source_server`` off each tool. Every wrapper in this pipeline delegates
``metadata`` and forwards no other attribute, so a wrapped tool answered
``None``: its tools were counted, its server was never marked loaded, and the
loop that follows then synthesized ``no_tools_returned`` for it -- failing
STRICT setup for a connector that had just answered.

That was already true on the sandbox transport before any gate existed, which
is why the fix is here and not on one wrapper: reading through the metadata
contract covers every wrapper that honours it, including the ones not written
yet.
"""

from __future__ import annotations

from typing import Any, Mapping, Type

import pytest
from pydantic import BaseModel

from xagent.core.tools.adapters.vibe.base import AbstractBaseTool
from xagent.core.tools.adapters.vibe.mcp_tools import _build_mcp_load_summary
from xagent.core.tools.adapters.vibe.write_gate_tool import gate_mcp_tools

CONFIGS = [{"name": "slack"}]


class _ArgsModel(BaseModel):
    pass


class _AdapterLike(AbstractBaseTool):
    """The shape ``MCPToolAdapter`` presents: the attribute *and* the metadata."""

    source_server = "slack"

    @property
    def name(self) -> str:
        return "slack_post_message"

    @property
    def description(self) -> str:
        return "post"

    @property
    def tags(self) -> list[str]:
        return []

    def args_type(self) -> Type[BaseModel]:
        return _ArgsModel

    def return_type(self) -> Type[BaseModel]:
        return _ArgsModel

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        return {}

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return {}


class _MetadataOnlyWrapper(AbstractBaseTool):
    """A wrapper that delegates ``metadata`` and forwards nothing else.

    Not a stand-in for one particular class: this is the contract every
    wrapper in the pipeline actually keeps. ``SandboxedToolWrapper`` has kept
    exactly this shape since before the write gate existed.
    """

    def __init__(self, target: AbstractBaseTool) -> None:
        self._target = target

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
    def metadata(self):  # type: ignore[override]
        return self._target.metadata

    def args_type(self) -> Type[BaseModel]:
        return self._target.args_type()

    def return_type(self) -> Type[BaseModel]:
        return self._target.return_type()

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        return self._target.run_json_sync(args)

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return await self._target.run_json_async(args)


def test_a_bare_adapter_marks_its_server_loaded() -> None:
    """The baseline: this never broke, and must keep working."""
    summary = _build_mcp_load_summary(CONFIGS, [_AdapterLike()])

    assert summary.loaded_servers == ("slack",)
    assert summary.failures == ()
    assert summary.successful_tool_count == 1


def test_a_metadata_only_wrapper_marks_its_server_loaded() -> None:
    """The pre-existing bug, independent of the write gate.

    Reading the raw attribute made this answer ``None``, so the server fell
    through to ``no_tools_returned`` while its tool was still counted -- an
    internally contradictory summary that failed STRICT setup for a connector
    that had answered.
    """
    summary = _build_mcp_load_summary(CONFIGS, [_MetadataOnlyWrapper(_AdapterLike())])

    assert summary.loaded_servers == ("slack",)
    assert summary.failures == ()
    assert summary.successful_tool_count == 1


def test_a_gated_tool_marks_its_server_loaded() -> None:
    """And the wrapper this PR adds, through the real ``gate_mcp_tools``."""
    summary = _build_mcp_load_summary(CONFIGS, list(gate_mcp_tools([_AdapterLike()])))

    assert summary.loaded_servers == ("slack",)
    assert summary.failures == ()
    assert summary.successful_tool_count == 1


def test_a_gated_sandbox_wrapper_marks_its_server_loaded() -> None:
    """Both wrappers stacked, which is the production npx/uvx shape."""
    stacked = gate_mcp_tools([_MetadataOnlyWrapper(_AdapterLike())])

    summary = _build_mcp_load_summary(CONFIGS, list(stacked))

    assert summary.loaded_servers == ("slack",)
    assert summary.failures == ()


@pytest.mark.parametrize("tools", [[], None])
def test_a_server_that_returned_nothing_is_still_reported(tools) -> None:
    """The guard on the other side: the failure path must stay reachable.

    A fix that marked every requested server loaded would silence the real
    ``no_tools_returned``, which is the condition STRICT setup exists to catch.
    """
    summary = _build_mcp_load_summary(CONFIGS, list(tools or []))

    assert summary.loaded_servers == ()
    assert [failure.reason for failure in summary.failures] == ["no_tools_returned"]
    assert summary.successful_tool_count == 0
