"""Initial database schema"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20240929_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "trades",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(length=50), nullable=False, index=True),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("qty", sa.Float(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
    )

    op.create_table(
        "trade_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
        sa.Column("symbol", sa.String(length=50), nullable=False, index=True),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("qty", sa.Float(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
    )

    op.create_table(
        "candles",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(length=50), nullable=False, index=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), nullable=False),
    )

    op.create_table(
        "equity_curve",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("date", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("equity", sa.Float(), nullable=False),
    )

    op.create_table(
        "training_tasks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbols", sa.Text(), nullable=False),
        sa.Column("limit", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("details", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "settings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("api_key", sa.String(length=255), nullable=True),
        sa.Column("api_secret", sa.String(length=255), nullable=True),
    )


def downgrade():
    op.drop_table("settings")
    op.drop_table("training_tasks")
    op.drop_table("equity_curve")
    op.drop_table("candles")
    op.drop_table("trade_logs")
    op.drop_table("trades")
