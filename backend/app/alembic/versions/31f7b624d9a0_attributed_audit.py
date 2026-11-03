"""Attribute committed administrative mutations without recording secret values."""

from alembic import op
import sqlalchemy as sa

revision = "31f7b624d9a0"
down_revision = "0d7c24a5b6f8"
branch_labels = None
depends_on = None


def upgrade():
    for name, kind in [
        ("actor_id", sa.String(36)),
        ("actor_type", sa.String(16)),
        ("api_key_id", sa.Integer()),
        ("request_id", sa.String(36)),
        ("knowledge_base_id", sa.Integer()),
        ("target_identifier", sa.String(128)),
        ("changed_fields", sa.JSON()),
    ]:
        op.add_column("staff_action_logs", sa.Column(name, kind, nullable=True))
    for name in ("actor_id", "request_id", "knowledge_base_id"):
        op.create_index("ix_staff_action_logs_" + name, "staff_action_logs", [name])


def downgrade():
    for name in ("actor_id", "request_id", "knowledge_base_id"):
        op.drop_index("ix_staff_action_logs_" + name, table_name="staff_action_logs")
    for name in (
        "changed_fields",
        "target_identifier",
        "knowledge_base_id",
        "request_id",
        "api_key_id",
        "actor_type",
        "actor_id",
    ):
        op.drop_column("staff_action_logs", name)
