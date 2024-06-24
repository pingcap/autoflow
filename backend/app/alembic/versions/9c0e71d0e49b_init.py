"""init

Revision ID: 9c0e71d0e49b
Revises: 
Create Date: 2024-06-25 16:57:25.422188

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from tidb_vector.sqlalchemy import VectorType


# revision identifiers, used by Alembic.
revision = '9c0e71d0e49b'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('chat_engine',
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(length=256), nullable=False),
    sa.Column('engine_options', sa.JSON(), nullable=True),
    sa.Column('is_default', sa.Boolean(), nullable=False),
    sa.Column('deleted_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('document',
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('hash', sqlmodel.sql.sqltypes.AutoString(length=32), nullable=False),
    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(length=256), nullable=False),
    sa.Column('content', sa.Text(), nullable=True),
    sa.Column('mime_type', sqlmodel.sql.sqltypes.AutoString(length=64), nullable=False),
    sa.Column('source_uri', sqlmodel.sql.sqltypes.AutoString(length=512), nullable=False),
    sa.Column('last_modified_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('entities',
    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(length=512), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('meta', sa.JSON(), nullable=True),
    sa.Column('entity_type', sa.Enum('original', 'synopsis', name='entitytype'), nullable=False),
    sa.Column('synopsis_info', sa.JSON(), nullable=True),
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('description_vec', VectorType(dim=1536), nullable=True),
    sa.Column('meta_vec', VectorType(dim=1536), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('feedback',
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('feedback_type', sa.Enum('like', 'dislike', name='feedbacktype'), nullable=False),
    sa.Column('query', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('langfuse_link', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('relationships', sa.JSON(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('option',
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(length=256), nullable=False),
    sa.Column('group_name', sqlmodel.sql.sqltypes.AutoString(length=256), nullable=False),
    sa.Column('value', sa.JSON(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('staff_action_logs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('action', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('action_time', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('target_type', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('target_id', sa.Integer(), nullable=False),
    sa.Column('before', sa.JSON(), nullable=True),
    sa.Column('after', sa.JSON(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('chat',
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.Column('title', sqlmodel.sql.sqltypes.AutoString(length=256), nullable=False),
    sa.Column('engine_id', sa.Integer(), nullable=False),
    sa.Column('engine_options', sa.JSON(), nullable=True),
    sa.Column('deleted_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['engine_id'], ['chat_engine.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_chat_id'), 'chat', ['id'], unique=False)
    op.create_table('llama_index_document',
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('hash', sqlmodel.sql.sqltypes.AutoString(length=32), nullable=False),
    sa.Column('text', sa.Text(), nullable=True),
    sa.Column('meta', sa.JSON(), nullable=True),
    sa.Column('embedding', VectorType(dim=1536), nullable=True),
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['document.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('relationships',
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('meta', sa.JSON(), nullable=True),
    sa.Column('weight', sa.Integer(), nullable=False),
    sa.Column('source_entity_id', sa.Integer(), nullable=False),
    sa.Column('target_entity_id', sa.Integer(), nullable=False),
    sa.Column('last_modified_at', sa.DateTime(), nullable=True),
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('description_vec', VectorType(dim=1536), nullable=True),
    sa.ForeignKeyConstraint(['source_entity_id'], ['entities.id'], ),
    sa.ForeignKeyConstraint(['target_entity_id'], ['entities.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('chat_message',
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('ordinal', sa.Integer(), nullable=False),
    sa.Column('role', sqlmodel.sql.sqltypes.AutoString(length=64), nullable=False),
    sa.Column('content', sa.Text(), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('trace_url', sqlmodel.sql.sqltypes.AutoString(length=512), nullable=True),
    sa.Column('finshed_at', sa.DateTime(), nullable=True),
    sa.Column('chat_id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.ForeignKeyConstraint(['chat_id'], ['chat.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('llama_index_chunk',
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('hash', sqlmodel.sql.sqltypes.AutoString(length=32), nullable=False),
    sa.Column('text', sa.Text(), nullable=True),
    sa.Column('meta', sa.JSON(), nullable=True),
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['llama_index_document.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('llama_index_chunk')
    op.drop_table('chat_message')
    op.drop_table('relationships')
    op.drop_table('llama_index_document')
    op.drop_index(op.f('ix_chat_id'), table_name='chat')
    op.drop_table('chat')
    op.drop_table('staff_action_logs')
    op.drop_table('option')
    op.drop_table('feedback')
    op.drop_table('entities')
    op.drop_table('document')
    op.drop_table('chat_engine')
    # ### end Alembic commands ###