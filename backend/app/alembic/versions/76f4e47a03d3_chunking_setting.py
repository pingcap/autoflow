"""chunking_setting

Revision ID: 76f4e47a03d3
Revises: 2adc0b597dcd
Create Date: 2025-02-13 18:21:17.830980

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "76f4e47a03d3"
down_revision = "2adc0b597dcd"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column(
        "documents",
        sa.Column(
            "content_format",
            sa.Enum("TEXT", "MARKDOWN", name="contentformat"),
            nullable=False,
        ),
    )
    op.add_column(
        "knowledge_bases", sa.Column("chunking_config", sa.JSON(), nullable=True)
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("knowledge_bases", "chunking_config")
    op.drop_column("documents", "content_format")
    # ### end Alembic commands ###
