"""add ai training tables

Revision ID: add_ai_training
Revises: 8c2c58a0d6d7
Create Date: 2025-11-12

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_ai_training'
down_revision = '8c2c58a0d6d7'
branch_labels = None
depends_on = None


def upgrade():
    # Create ai_training_samples table
    op.create_table(
        'ai_training_samples',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True),
        sa.Column('symbol', sa.String(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=True),
        sa.Column('predicted_action', sa.String(), nullable=False),
        sa.Column('prediction_score', sa.Float(), nullable=False),
        sa.Column('prediction_confidence', sa.Float(), nullable=False),
        sa.Column('model_version', sa.String(), nullable=True),
        sa.Column('features', sa.Text(), nullable=False),
        sa.Column('feature_names', sa.Text(), nullable=True),
        sa.Column('executed', sa.Boolean(), default=False),
        sa.Column('execution_side', sa.String(), nullable=True),
        sa.Column('entry_price', sa.Float(), nullable=True),
        sa.Column('entry_quantity', sa.Float(), nullable=True),
        sa.Column('entry_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('outcome_known', sa.Boolean(), default=False),
        sa.Column('exit_price', sa.Float(), nullable=True),
        sa.Column('exit_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('realized_pnl', sa.Float(), nullable=True),
        sa.Column('hold_duration_seconds', sa.Integer(), nullable=True),
        sa.Column('target_label', sa.Float(), nullable=True),
        sa.Column('target_class', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['run_id'], ['liquidity_runs.id'], ),
    )
    op.create_index('ix_ai_training_samples_symbol', 'ai_training_samples', ['symbol'])
    op.create_index('ix_ai_training_samples_timestamp', 'ai_training_samples', ['timestamp'])
    
    # Create ai_model_versions table
    op.create_table(
        'ai_model_versions',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True),
        sa.Column('version_id', sa.String(), nullable=False),
        sa.Column('model_type', sa.String(), nullable=False),
        sa.Column('file_path', sa.String(), nullable=False),
        sa.Column('trained_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('training_samples', sa.Integer(), nullable=False),
        sa.Column('training_duration_seconds', sa.Float(), nullable=True),
        sa.Column('train_accuracy', sa.Float(), nullable=True),
        sa.Column('validation_accuracy', sa.Float(), nullable=True),
        sa.Column('train_mae', sa.Float(), nullable=True),
        sa.Column('validation_mae', sa.Float(), nullable=True),
        sa.Column('total_predictions', sa.Integer(), default=0),
        sa.Column('correct_predictions', sa.Integer(), default=0),
        sa.Column('live_accuracy', sa.Float(), nullable=True),
        sa.Column('total_pnl', sa.Float(), default=0.0),
        sa.Column('is_active', sa.Boolean(), default=False),
        sa.Column('replaced_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
    )
    op.create_index('ix_ai_model_versions_version_id', 'ai_model_versions', ['version_id'], unique=True)


def downgrade():
    op.drop_index('ix_ai_model_versions_version_id', table_name='ai_model_versions')
    op.drop_table('ai_model_versions')
    op.drop_index('ix_ai_training_samples_timestamp', table_name='ai_training_samples')
    op.drop_index('ix_ai_training_samples_symbol', table_name='ai_training_samples')
    op.drop_table('ai_training_samples')
