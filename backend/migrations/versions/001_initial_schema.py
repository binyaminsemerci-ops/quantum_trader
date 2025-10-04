"""Initial schema - TradeLog and Settings tables

Revision ID: 001_initial_schema
Revises:
Create Date: 2025-10-04 13:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial schema with TradeLog and Settings tables."""
    # Create trade_logs table
    op.create_table(
        'trade_logs',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('symbol', sa.String(), index=True),
        sa.Column('side', sa.String()),
        sa.Column('qty', sa.Float()),
        sa.Column('price', sa.Float()),
        sa.Column('status', sa.String()),
        sa.Column('reason', sa.String(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), default=sa.func.now()),
    )

    # Create settings table
    op.create_table(
        'settings',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('api_key', sa.String()),
        sa.Column('api_secret', sa.String()),
    )


def downgrade() -> None:
    """Drop the initial schema."""
    op.drop_table('settings')
    op.drop_table('trade_logs')
