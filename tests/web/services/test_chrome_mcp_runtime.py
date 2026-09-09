import pytest

from xagent.web.services.chrome_mcp_runtime import require_chrome_builtin_stdio_policy
from xagent.web.services.mcp_runtime import (
    MCPActorAuthorizationPolicy,
    MCPBuiltinOAuthActorPolicyRequiredError,
)


@pytest.mark.parametrize(
    "policy",
    [None, MCPActorAuthorizationPolicy(resource_owner_key="toby:owner")],
)
def test_chrome_stdio_requires_explicit_actor_capability(policy):
    with pytest.raises(MCPBuiltinOAuthActorPolicyRequiredError):
        require_chrome_builtin_stdio_policy(policy)


def test_chrome_stdio_gate_returns_the_same_immutable_policy():
    policy = MCPActorAuthorizationPolicy(
        resource_owner_key="toby:owner", allow_builtin_stdio=True
    )
    assert require_chrome_builtin_stdio_policy(policy) is policy
