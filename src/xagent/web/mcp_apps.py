"""Centralized registry for MCP Applications and OAuth Providers.

This module provides a scalable structure for defining supported MCP applications,
their OAuth configurations, and server launch configurations.
"""

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from .builtin_mcp_registry import get_builtin_execution_fields_and_optional_scopes
from .models.mcp import MCPServer, UserMCPServer
from .models.public_mcp import PublicMCPApp

# Reasons explain_mcp_oauth_reconnect_refusal can return. Stable identifiers:
# a consumer (e.g. a "may I tear this connection down?" gate) branches on them,
# so they are part of this function's contract and must not be reworded to
# track an upstream error message.
MCP_OAUTH_RECONNECT_REFUSAL_APP_MISSING = "app_missing"
MCP_OAUTH_RECONNECT_REFUSAL_APP_HIDDEN = "app_hidden"
MCP_OAUTH_RECONNECT_REFUSAL_NOT_MCP_OAUTH = "not_mcp_oauth"
MCP_OAUTH_RECONNECT_REFUSAL_SERVER_CONFIG_DRIFT = "server_config_drift"
MCP_OAUTH_RECONNECT_REFUSAL_USER_OWNED_SQUAT = "user_owned_squat"
MCP_OAUTH_RECONNECT_REFUSAL_INVALID_SERVER_CONFIG = "invalid_server_config"
MCP_OAUTH_RECONNECT_REFUSAL_INVALID_AUTH_CONFIG = "invalid_auth_config"

# Apps that must not be satisfied by a bare provider-level OAuth grant (one
# created via the app_id-less connect flow, e.g. UserOAuth.provider == "meta").
# That flow requests only the provider's default scopes (see
# generic_oauth_login's app_scopes=None branch when app_id is absent), never
# an app's own oauth_scopes, so it can't carry a permission such as
# pages_read_user_content that was added after the bare flow already existed.
# Only an app-scoped grant (UserOAuth.provider == the app_id) counts for these
# apps; Instagram is deliberately excluded so its existing bare "meta" grants
# keep working, since its required scopes haven't changed.
#
# github: the github oauth_providers row's own default_scopes is
# identity-only ("read:user") -- the functional "repo"/"user:email" scopes
# live solely on the app row and are merged in only when generic_oauth_login
# is called with app_id="github". A bare GET /api/auth/github/login (no
# app_id) would otherwise request just "read:user", and the callback's bare
# batch-connect branch would still activate the github app's UserMCPServer
# against that under-scoped grant -- reporting "connected" while every
# repo-scoped tool then fails. This is a no-op for the normal connect flow
# (the catalog UI always passes app_id="github", and app_id == provider_name
# for this connector, so an already-connected grant already satisfies the
# app-scoped match trivially).
APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT = frozenset({"facebook", "github"})


def _normalize_oauth_grant_key(value: object) -> str | None:
    """Case/whitespace-insensitive key, matching mcp.py's _normalize_app_key.

    Duplicated rather than imported: mcp.py imports this module, so importing
    back would cycle. An admin-created PublicMCPApp.app_id is free-form (see
    POST /admin/mcp/apps), so every APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT
    membership test must normalize the same way the connector-display layer
    does, or a differently-cased app_id (e.g. "Facebook") silently bypasses
    the policy at whichever call site compares raw strings instead.
    """
    if value is None:
        return None
    normalized = "-".join(str(value).strip().lower().split())
    return normalized or None


def requires_app_scoped_oauth_grant(app_id: object) -> bool:
    """Whether app_id must not be satisfied by a bare provider-level grant.

    Normalized the same way _app_lookup_keys resolves an app's own id, so a
    differently-cased or whitespace-padded admin-created app_id (e.g.
    "Facebook") is covered consistently everywhere this policy is checked.
    """
    return _normalize_oauth_grant_key(app_id) in APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT


def restrict_to_app_scoped_oauth_grant(
    app_id: object, candidates: Iterable[object]
) -> list[str]:
    """Narrow OAuth provider/grant candidates to app-scoped ones where required.

    ``candidates`` is typically ``(provider_name, app_id)`` or a list of
    ``UserOAuth.provider`` values to try. For an app in
    ``APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT``, only the candidate matching
    ``app_id`` (normalized) survives — a bare provider-level grant is dropped.
    For every other app, candidates pass through unchanged (deduped, order
    preserved). Candidates are returned in their original casing/whitespace:
    callers match them against exact-case stored values (e.g.
    ``UserOAuth.provider``), which is why normalization only decides
    membership and is never applied to the returned strings.
    """
    deduped = list(dict.fromkeys(c for c in candidates if isinstance(c, str) and c))

    if not requires_app_scoped_oauth_grant(app_id):
        return deduped
    normalized_app_id = _normalize_oauth_grant_key(app_id)
    return [
        candidate
        for candidate in deduped
        if _normalize_oauth_grant_key(candidate) == normalized_app_id
    ]


def classify_app_auth(transport: Any, launch_config: Any) -> str:
    """Single source of truth for how a catalog app is connected.

    Derived from the entry's own fields so the backend connect gate and both
    frontend dialogs can't drift apart. Values:
        - "builtin_oauth": provider redirect flow (transport == "oauth")
        - "api_key": static key, connected via /api/mcp/apps/{id}/connect
        - "keyless": local stdio module that needs no secrets (e.g. Chrome),
          connected via the same endpoint with no env
        - "mcp_oauth": remote MCP server, connected via per-user OAuth
          Authorization Code + PKCE (Dynamic Client Registration when no
          static client_id is configured) — /api/mcp/apps/{id}/oauth/connect
        - "unconnectable": none of the above
    """
    # Reuses the runtime's notion of which transports are remote ("oauth"
    # here instead means a static-provider redirect wrapping our own stdio
    # module). Lowercased like the builtin_oauth check above: an admin PATCH
    # can store a mixed-case transport, and the two halves of this feature
    # must not disagree about the same row.
    from .services.mcp_runtime import HTTP_MCP_TRANSPORTS

    if str(transport or "").lower() == "oauth":
        return "builtin_oauth"
    launch = launch_config if isinstance(launch_config, dict) else {}
    if launch.get("required_env") and launch.get("command"):
        return "api_key"
    # Keyless is deliberately stdio-only: a command on a remote transport is a
    # mis-authored entry, not a connectable app. Also excludes env_mapping —
    # that shape means the launcher expects an injected token (the builtin
    # OAuth apps' pattern, e.g. env_mapping={"SLACK_ACCESS_TOKEN":
    # "access_token"}), so a custom app authored with that shape is not
    # actually secret-free even though it has no required_env; classifying it
    # keyless would offer a no-secrets Connect button for a server that fails
    # at tool-call time for a missing token.
    if (
        str(transport or "").lower() == "stdio"
        and launch.get("command")
        and not launch.get("required_env")
        and not launch.get("env_mapping")
    ):
        return "keyless"
    auth = launch.get("auth")
    if (
        str(transport or "").lower() in HTTP_MCP_TRANSPORTS
        and launch.get("url")
        and isinstance(auth, dict)
        and auth.get("type") == "mcp_oauth"
    ):
        return "mcp_oauth"
    return "unconnectable"


def _app_to_dict(app: PublicMCPApp) -> Dict[str, Any]:
    # One registry scan (not two - see the helper's own docstring) since
    # this runs per app on the connector-listing path.
    execution_fields, optional_oauth_scopes = (
        get_builtin_execution_fields_and_optional_scopes(app.app_id)
    )
    if execution_fields is None:
        execution_fields = {
            "name": app.name,
            "transport": app.transport,
            "provider_name": app.provider_name,
            "oauth_scopes": deepcopy(app.oauth_scopes or []),
            "launch_config": deepcopy(app.launch_config or {}),
        }

    transport = execution_fields["transport"]
    launch_config = deepcopy(execution_fields["launch_config"])
    return {
        "id": app.app_id,
        "name": execution_fields["name"],
        "description": app.description,
        "icon": app.icon,
        "transport": transport,
        "provider": execution_fields["provider_name"],
        "category": app.category,
        "oauth_scopes": deepcopy(execution_fields["oauth_scopes"]),
        # Only builtin apps can declare these today (see
        # get_builtin_execution_fields_and_optional_scopes) - a custom
        # admin-created app has no column for it and always gets [].
        "optional_oauth_scopes": optional_oauth_scopes,
        "is_visible_in_connector": bool(app.is_visible_in_connector),
        "launch_config": launch_config,
        "auth_type": classify_app_auth(transport, launch_config),
    }


def get_all_mcp_apps(db: Session) -> List[Dict[str, Any]]:
    """Retrieve all MCP apps from the database dynamically."""
    apps = db.query(PublicMCPApp).all()
    return [_app_to_dict(app) for app in apps]


def get_app_by_id(db: Session, app_id: str) -> Dict[str, Any] | None:
    """Retrieve an MCP app configuration by its ID."""
    app = db.query(PublicMCPApp).filter(PublicMCPApp.app_id == app_id).first()
    return _app_to_dict(app) if app else None


def get_app_by_name(db: Session, name: str) -> Dict[str, Any] | None:
    """Retrieve an MCP app configuration by its exact name."""
    app = db.query(PublicMCPApp).filter(PublicMCPApp.name == name).first()
    return _app_to_dict(app) if app else None


def get_app_for_mcp_server(db: Session, server: Any) -> Dict[str, Any] | None:
    """Resolve a server's catalog app by stable identity when it is available.

    Older server rows predate ``auth.app_id`` and are still resolved by their
    exact catalog name. Once a row carries ``app_id``, an invalid value must not
    fall back to a same-named app because that could select another connector's
    credentials or launch configuration.
    """
    auth = getattr(server, "auth", None)
    if isinstance(auth, Mapping) and "app_id" in auth:
        app_id = auth.get("app_id")
        if not isinstance(app_id, str) or not app_id:
            return None
        return get_app_by_id(db, app_id)
    return get_app_by_name(db, str(getattr(server, "name", "")))


def _bounded_auth_value_refusal(auth: Mapping[str, Any]) -> str | None:
    """Reject auth config that connect_mcp_oauth would 400 on deterministically.

    Only the statically decidable half of connect_mcp_oauth's validation. The
    ``issuer`` and ``resource`` bounds it also enforces are deliberately out of
    scope: those values come from runtime discovery against the remote server,
    so no amount of reading the catalog row can predict them.

    ``token_endpoint_auth_method`` is checked only on the static-``client_id``
    branch, mirroring connect_mcp_oauth exactly. Without a client_id the flow
    takes the Dynamic Client Registration branch, where the method is whatever
    the authorization server hands back at registration time — not this row's.
    """
    from .services.mcp_oauth import (
        MCP_OAUTH_PERSISTED_VALUE_MAX_LENGTH,
        MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD_MAX_LENGTH,
        MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHODS,
    )

    def _configured(key: str) -> str | None:
        # Matches _configured_mcp_oauth_value: strip, and treat falsy as absent.
        value = auth.get(key)
        return str(value).strip() if value else None

    # redirect_uri is bounded whether or not it is configured -- an absent one
    # falls back to a default that is itself bounded, and a default we generate
    # is never over the limit, so only a configured value can refuse here.
    redirect_uri = _configured("redirect_uri")
    if redirect_uri and len(redirect_uri) > MCP_OAUTH_PERSISTED_VALUE_MAX_LENGTH:
        return MCP_OAUTH_RECONNECT_REFUSAL_INVALID_AUTH_CONFIG

    client_id = _configured("client_id")
    if not client_id:
        return None
    if len(client_id) > MCP_OAUTH_PERSISTED_VALUE_MAX_LENGTH:
        return MCP_OAUTH_RECONNECT_REFUSAL_INVALID_AUTH_CONFIG

    # Same default as connect_mcp_oauth: an unset method means "post" when a
    # secret is configured and "none" when it is not, and both defaults are
    # members of the allowlist -- so only an explicit value can refuse.
    method = str(
        auth.get("token_endpoint_auth_method")
        or ("client_secret_post" if _configured("client_secret") else "none")
    )
    if len(method) > MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD_MAX_LENGTH:
        return MCP_OAUTH_RECONNECT_REFUSAL_INVALID_AUTH_CONFIG
    if method not in MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHODS:
        return MCP_OAUTH_RECONNECT_REFUSAL_INVALID_AUTH_CONFIG
    return None


def explain_mcp_oauth_reconnect_refusal(
    db: Session, app_id: str, user_id: int
) -> str | None:
    """Why an mcp_oauth reconnect for ``app_id`` would fail now, or None.

    Answers "if this user were to POST /api/mcp/apps/{app_id}/oauth/connect
    right now, would it be refused before any authorization redirect?" -- None
    means the deterministic gates all pass. A non-None result is one of the
    ``MCP_OAUTH_RECONNECT_REFUSAL_*`` codes.

    Single source of truth in the same spirit as classify_app_auth:
    connect_mcp_oauth_app calls this itself before it touches the database, so
    a consumer asking the question ahead of time and the connect path answering
    it for real cannot drift apart. A caller that only predicted the outcome by
    re-deriving these rules would be a second copy, silently going stale on the
    next hardening fix here.

    Only *deterministic* refusals are covered -- those decidable from the
    catalog row, the shared server row and this user's associations. Anything
    that depends on reaching the remote server (OAuth metadata discovery, DCR,
    the authorization server's own errors) is out of scope and may still fail
    after this returns None.

    Ordering note: calling this before the connect path mutates anything also
    fixes a pre-existing one-way door. connect_mcp_oauth_app used to commit the
    shared server row and this user's association *first* and only then call
    connect_mcp_oauth, which is where the auth-config bounds are enforced -- so
    a mis-authored catalog entry produced a connection that disconnected fine
    but whose every reconnect attempt created an association and then 400'd.

    Not covered on purpose: whether a *shared* server row could be created at
    all when none exists yet. That path (_add_catalog_server_with_race_recovery)
    can fail on infrastructure grounds, which is not statically decidable; the
    schema-level half of it -- a catalog transport MCPServerConfig would reject
    -- is checked here.

    ``user_id`` is deliberately part of the signature even though no refusal
    reads it today: every gate the connect path applies before the redirect
    happens to be user-independent. The owned-row check is the near miss --
    it rejects a row *any* user owns, the caller included, so it is not a
    per-user rule either. Asking the question per user is still the honest
    contract (whether *this* user can reconnect), and it means a future
    user-scoped gate -- a per-user block, a quota, an association in a state
    that cannot be revived -- lands here rather than forcing every consumer to
    change signature.
    """
    app_info = get_app_by_id(db, app_id)
    if not app_info:
        return MCP_OAUTH_RECONNECT_REFUSAL_APP_MISSING
    # Mirrors _reject_hidden_catalog_app: hiding an app is used as a release
    # gate and blocks reconnect for already-connected users too, not just
    # fresh connects.
    if not app_info.get("is_visible_in_connector", True):
        return MCP_OAUTH_RECONNECT_REFUSAL_APP_HIDDEN
    if app_info.get("auth_type") != "mcp_oauth":
        return MCP_OAUTH_RECONNECT_REFUSAL_NOT_MCP_OAUTH

    launch = app_info.get("launch_config") or {}
    url = launch.get("url")
    auth = launch.get("auth")
    auth = auth if isinstance(auth, Mapping) else {}
    transport = str(app_info["transport"])
    server_name = str(app_info["id"])

    # The auth bounds are a property of the catalog entry alone, so they refuse
    # identically on the reuse and the create path. Checked before either so a
    # mis-authored entry reports the authoring fault rather than whichever
    # row-shaped symptom it happens to hit first.
    auth_refusal = _bounded_auth_value_refusal(auth)
    if auth_refusal is not None:
        return auth_refusal

    server = db.query(MCPServer).filter(MCPServer.name == server_name).first()
    if server is None:
        # Create path: the catalog's own transport must survive
        # MCPServerConfig's validator, which matches exact case -- while
        # classify_app_auth lowercases before testing HTTP_MCP_TRANSPORTS. An
        # entry authored as "Streamable_HTTP" therefore classifies as
        # mcp_oauth and then cannot be persisted at all.
        from ..core.tools.core.mcp.data_config import MCPServerConfig

        try:
            MCPServerConfig(
                name=server_name,
                transport=transport,
                url=url,
                managed="external",
            )
        except ValueError:
            return MCP_OAUTH_RECONNECT_REFUSAL_INVALID_SERVER_CONFIG
        return None

    # Reuse path: the existing shared row must still match the catalog, or
    # _ensure_catalog_mcp_oauth_server 409s rather than adopting a possible
    # hijack. Transport compared case-insensitively, exactly as there.
    if str(server.transport or "").lower() != transport.lower() or server.url != url:
        return MCP_OAUTH_RECONNECT_REFUSAL_SERVER_CONFIG_DRIFT
    owned = (
        db.query(UserMCPServer)
        .filter(
            UserMCPServer.mcpserver_id == server.id,
            UserMCPServer.is_owner.is_(True),
        )
        .first()
    )
    if owned is not None:
        return MCP_OAUTH_RECONNECT_REFUSAL_USER_OWNED_SQUAT
    return None
