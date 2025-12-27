"""add training tasks table

Revision ID: 1e6e6c7af2f3
Revises: 001_initial_schema
Create Date: 2025-11-04 23:35:08.798236

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1e6e6c7af2f3'
down_revision: Union[str, Sequence[str], None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create training_tasks table used to track background AI jobs.
    op.create_table(
        'training_tasks',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('symbols', sa.String(), nullable=False),
        sa.Column('limit', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(), nullable=False, server_default=sa.text("'pending'")),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('details', sa.Text(), nullable=True),
    )
    op.create_index(op.f('ix_training_tasks_id'), 'training_tasks', ['id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_training_tasks_id'), table_name='training_tasks')
    op.drop_table('training_tasks')
