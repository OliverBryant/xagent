"""Unit tests for the single source of truth for catalog app auth classification.

`classify_app_auth` is the one place backend + both frontend dialogs read from,
so its edge cases matter: oauth wins over any launch_config, key-based needs
BOTH required_env and command, everything else is unconnectable.
"""

import pytest

from xagent.web.mcp_apps import classify_app_auth


def test_oauth_transport_is_builtin_oauth():
    assert classify_app_auth("oauth", None) == "builtin_oauth"
    assert classify_app_auth("OAuth", {}) == "builtin_oauth"
    # oauth transport wins even if a stray launch_config is present
    assert (
        classify_app_auth("oauth", {"command": "npx", "required_env": ["X"]})
        == "builtin_oauth"
    )


def test_key_based_needs_command_and_required_env():
    assert (
        classify_app_auth("stdio", {"command": "npx", "required_env": ["KEY"]})
        == "api_key"
    )


def test_inconsistent_entries_are_unconnectable_not_misrouted():
    # required_env but no command -> not launchable
    assert classify_app_auth("stdio", {"required_env": ["KEY"]}) == "unconnectable"
    # command but no required_env -> nothing to prompt for
    assert classify_app_auth("stdio", {"command": "npx"}) == "unconnectable"
    assert classify_app_auth("stdio", None) == "unconnectable"
    assert classify_app_auth(None, None) == "unconnectable"
    # malformed launch_config (non-dict) must not raise AttributeError
    assert classify_app_auth("stdio", ["not", "a", "dict"]) == "unconnectable"
    assert classify_app_auth("stdio", "garbage") == "unconnectable"


def test_oauth_landing_rejects_non_oauth_app():
    # Symmetric guard: a key-based app must not be connectable via the OAuth
    # flow (it would get a token, never its required_env API key). The guard
    # runs before any DB access, so db=None is fine. It raises the dedicated
    # AppNotOAuthError so the batch loop can skip only this case.
    from xagent.web.api.auth import AppNotOAuthError, _ensure_user_mcp_server

    with pytest.raises(AppNotOAuthError, match="not an OAuth app"):
        _ensure_user_mcp_server(
            None,
            "1",
            {"id": "google-maps", "name": "Google Maps", "auth_type": "api_key"},
        )
