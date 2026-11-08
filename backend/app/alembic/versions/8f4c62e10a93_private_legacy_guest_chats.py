"""Stop publishing historical guest conversations that defaulted to public."""

from alembic import op

revision = "8f4c62e10a93"
down_revision = "65ad0f3c2b81"
branch_labels = None
depends_on = None


def upgrade():
    # Legacy unsigned browser IDs cannot establish ownership. Preserve the data
    # for authenticated administrators, but revoke old anonymous public links.
    op.execute(
        "UPDATE chats SET visibility = 0 WHERE user_id IS NULL AND visibility = 1"
    )


def downgrade():
    # A software rollback must never republish private conversations.
    pass
