"""empty message

Revision ID: e32f1e546eec
Revises: bd17a4ebccc5
Create Date: 2024-08-08 03:55:14.042290

"""

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from app.models.base import AESEncryptedColumn


# revision identifiers, used by Alembic.
revision = "e32f1e546eec"
down_revision = "bd17a4ebccc5"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "reranker_models",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(length=64), nullable=False),
        sa.Column(
            "provider",
            sa.Enum("JINA", "COHERE", name="rerankerprovider"),
            nullable=False,
        ),
        sa.Column(
            "model", sqlmodel.sql.sqltypes.AutoString(length=256), nullable=False
        ),
        sa.Column("top_n", sa.Integer(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("credentials", AESEncryptedColumn(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.add_column("chat_engines", sa.Column("reranker_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        None, "chat_engines", "reranker_models", ["reranker_id"], ["id"]
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("chat_engines", "reranker_id")
    op.drop_table("reranker_models")
    # ### end Alembic commands ###
