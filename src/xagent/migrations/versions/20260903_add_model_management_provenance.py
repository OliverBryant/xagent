"""Add durable model management provenance.

Revision ID: 20260903_model_management
Revises: 20260901_seed_chartmogul_mcp_app
Create Date: 2026-09-03
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260903_model_management"
down_revision: Union[str, None] = "20260901_seed_chartmogul_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("models", sa.Column("managed_by", sa.String(50), nullable=True))


def downgrade() -> None:
    op.drop_column("models", "managed_by")
