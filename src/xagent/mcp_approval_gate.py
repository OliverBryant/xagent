"""Stable host integration surface for task-scoped MCP approval gates."""

from .core.tools.adapters.vibe.mcp_approval_gate import (
    GatedCall,
    GateDecision,
    MCPApprovalGateRegistration,
    MCPApprovalGateScope,
    MCPApprovalReplayContext,
    ToolCallExecutionContext,
    current_mcp_approval_replay_context,
    mcp_approval_gate_rollback_fence,
    mcp_approval_gate_scope,
    register_mcp_approval_gate,
    unregister_mcp_approval_gate,
)

__all__ = [
    "GatedCall",
    "GateDecision",
    "MCPApprovalGateRegistration",
    "MCPApprovalGateScope",
    "MCPApprovalReplayContext",
    "ToolCallExecutionContext",
    "current_mcp_approval_replay_context",
    "mcp_approval_gate_rollback_fence",
    "mcp_approval_gate_scope",
    "register_mcp_approval_gate",
    "unregister_mcp_approval_gate",
]
