"""Revoke sessions and API keys with atomic credential epochs."""

from alembic import op
import sqlalchemy as sa

revision = "4b8e2a70c1d9"
down_revision = "04947f9684ab"
branch_labels = None
depends_on = None


def upgrade():
    for table in ("users", "user_sessions", "api_keys"):
        op.add_column(
            table,
            sa.Column("auth_version", sa.Integer(), nullable=False, server_default="0"),
        )


def downgrade():
    # Do not allow a rollback to reactivate credentials invalidated by a reset.
    op.execute("DELETE FROM user_sessions")
    op.execute("UPDATE api_keys SET is_active = false")
    for table in ("api_keys", "user_sessions", "users"):
        op.drop_column(table, "auth_version")
