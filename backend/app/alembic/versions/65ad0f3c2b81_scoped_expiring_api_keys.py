"""Require explicit retrieval scopes and finite expiry for new API keys."""

from alembic import op
import sqlalchemy as sa

revision = "65ad0f3c2b81"
down_revision = "4b8e2a70c1d9"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("api_keys", sa.Column("expires_at", sa.DateTime(), nullable=True))
    op.add_column("api_keys", sa.Column("scopes", sa.JSON(), nullable=True))
    op.add_column("api_keys", sa.Column("knowledge_base_ids", sa.JSON(), nullable=True))
    # Existing unrestricted keys cannot be safely assigned a scope automatically.
    op.execute("UPDATE api_keys SET is_active = false")


def downgrade():
    # Do not turn constrained credentials into unrestricted credentials on rollback.
    op.execute("UPDATE api_keys SET is_active = false")
    for column in ("knowledge_base_ids", "scopes", "expires_at"):
        op.drop_column("api_keys", column)
