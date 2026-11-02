"""Bind chat-capable API keys to engines and their own conversations."""

from alembic import op
import sqlalchemy as sa

revision = "0d7c24a5b6f8"
down_revision = "8f4c62e10a93"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("api_keys", sa.Column("chat_engine_ids", sa.JSON(), nullable=True))
    op.add_column("chats", sa.Column("api_key_id", sa.Integer(), nullable=True))


def downgrade():
    op.execute("UPDATE api_keys SET is_active = false")
    op.drop_column("chats", "api_key_id")
    op.drop_column("api_keys", "chat_engine_ids")
