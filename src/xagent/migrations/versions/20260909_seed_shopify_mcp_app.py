"""Seed the built-in Shopify key-based MCP connector.

Revision ID: 20260909_seed_shopify_mcp_app
Revises: 20260901_seed_zendesk_mcp_app
"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

revision: str = "20260909_seed_shopify_mcp_app"
down_revision: Union[str, None] = "20260901_seed_zendesk_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("name", sa.String),
    sa.column("description", sa.Text),
    sa.column("icon", sa.String),
    sa.column("transport", sa.String),
    sa.column("provider_name", sa.String),
    sa.column("category", sa.String),
    sa.column("oauth_scopes", sa.JSON),
    sa.column("is_visible_in_connector", sa.Boolean),
    sa.column("launch_config", sa.JSON),
)
MCP_SERVERS_TABLE = sa.table(
    "mcp_servers",
    sa.column("name", sa.String),
)

APP_ID = "shopify"
BUILTIN_PROVENANCE = {
    "registry": "xagent",
    "app_id": APP_ID,
    "version": 1,
}
ROW = {
    "app_id": APP_ID,
    "name": "Shopify",
    "description": "Connect a Shopify custom app with a store label (for example, acme for acme.myshopify.com) and Admin API access token. Grant write_products, write_orders, and read_customers; read_all_orders is optional for orders older than 60 days.",
    "icon": "https://www.google.com/s2/favicons?domain=shopify.com&sz=128",
    "transport": "stdio",
    "provider_name": None,
    "category": "Commerce",
    "oauth_scopes": None,
    "is_visible_in_connector": True,
    "launch_config": {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.shopify"],
        "required_env": ["SHOPIFY_STORE_DOMAIN", "SHOPIFY_ACCESS_TOKEN"],
        "required_admin_scopes": [
            "write_products",
            "write_orders",
            "read_customers",
        ],
        "optional_admin_scopes": ["read_all_orders"],
        "credential_scope": "personal",
        "builtin_provenance": BUILTIN_PROVENANCE,
    },
}


def _has_provenance(launch_config: object) -> bool:
    return (
        isinstance(launch_config, dict)
        and launch_config.get("builtin_provenance") == BUILTIN_PROVENANCE
    )


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "public_mcp_apps" not in tables:
        return

    columns = {column["name"] for column in inspector.get_columns("public_mcp_apps")}
    if "launch_config" not in columns:
        raise RuntimeError(
            "Cannot seed builtin Shopify identity: public_mcp_apps.launch_config "
            "is required for provenance"
        )

    existing = (
        bind.execute(
            sa.select(
                PUBLIC_MCP_APPS_TABLE.c.app_id,
                PUBLIC_MCP_APPS_TABLE.c.launch_config,
            ).where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
        )
        .mappings()
        .first()
    )
    if existing is not None:
        if _has_provenance(existing["launch_config"]):
            return
        raise RuntimeError(
            "Cannot seed builtin Shopify connector: custom public_mcp_apps "
            "row with app_id 'shopify' already exists"
        )

    if "mcp_servers" in tables:
        server_collision = bind.execute(
            sa.select(MCP_SERVERS_TABLE.c.name).where(
                MCP_SERVERS_TABLE.c.name == APP_ID
            )
        ).first()
        if server_collision is not None:
            raise RuntimeError(
                "Cannot seed builtin Shopify connector: custom mcp_servers "
                "row named 'shopify' already exists"
            )

    dropped_keys = sorted(set(ROW) - columns)
    if dropped_keys:
        logger.warning(
            "public_mcp_apps is missing columns %s; seeding %r without them",
            dropped_keys,
            APP_ID,
        )
    bind.execute(
        sa.insert(PUBLIC_MCP_APPS_TABLE),
        [{key: value for key, value in ROW.items() if key in columns}],
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns("public_mcp_apps")}
    if "launch_config" not in columns:
        return
    existing = bind.execute(
        sa.select(PUBLIC_MCP_APPS_TABLE.c.launch_config).where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID
        )
    ).scalar_one_or_none()
    if not _has_provenance(existing):
        return
    bind.execute(
        sa.delete(PUBLIC_MCP_APPS_TABLE).where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
    )
