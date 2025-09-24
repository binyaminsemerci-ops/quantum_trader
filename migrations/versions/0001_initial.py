
"""initial revision - create tables used by the app

Revision ID: 0001_initial
Revises: 
Create Date: 2025-09-24
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create `trade_logs` table
    op.create_table(
        'trade_logs',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('symbol', sa.String(), nullable=False),
        sa.Column('side', sa.String(), nullable=False),
        sa.Column('qty', sa.Float(), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('reason', sa.String(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
    )
    op.create_index('ix_trade_logs_symbol', 'trade_logs', ['symbol'])

    # Create `settings` table
    op.create_table(
        'settings',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('api_key', sa.String(), nullable=True),
        sa.Column('api_secret', sa.String(), nullable=True),
    )


def downgrade():
    op.drop_index('ix_trade_logs_symbol', table_name='trade_logs')
    op.drop_table('trade_logs')
    op.drop_table('settings')
