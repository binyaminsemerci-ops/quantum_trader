"""Baseline schema revision.

This captures the current tables so future diffs are incremental.
"""
from __future__ import annotations

from alembic import op  # type: ignore
import sqlalchemy as sa  # type: ignore

# Revision identifiers, used by Alembic.
revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:  # pragma: no cover
    # Recreate minimal schema definition for baseline.
    op.create_table(
        "trades",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("symbol", sa.String(50), nullable=False, index=True),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("qty", sa.Float, nullable=False),
        sa.Column("price", sa.Float, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), index=True),
    )
    op.create_table(
        "trade_logs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), index=True),
        sa.Column("symbol", sa.String(50), nullable=False, index=True),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("qty", sa.Float, nullable=False),
        sa.Column("price", sa.Float, nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("reason", sa.Text),
    )
    op.create_table(
        "candles",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("symbol", sa.String(50), nullable=False, index=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), index=True),
        sa.Column("open", sa.Float, nullable=False),
        sa.Column("high", sa.Float, nullable=False),
        sa.Column("low", sa.Float, nullable=False),
        sa.Column("close", sa.Float, nullable=False),
        sa.Column("volume", sa.Float, nullable=False),
    )
    op.create_table(
        "model_registry",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("tag", sa.String(64), index=True),
        sa.Column("metrics_json", sa.Text),
        sa.Column("is_active", sa.Integer, index=True, server_default="0"),
        sa.Column("created_at", sa.DateTime),
        sa.Column("accuracy", sa.Float),
        sa.Column("path", sa.String(500)),
    )
    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("condition", sa.String(500), nullable=False),
        sa.Column("threshold", sa.Float, nullable=False),
        sa.Column("is_active", sa.Integer, server_default="1"),
        sa.Column("created_at", sa.DateTime),
    )
    op.create_table(
        "watchlist",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=False, unique=True),
        sa.Column("name", sa.String(100)),
        sa.Column("category", sa.String(50)),
        sa.Column("added_at", sa.DateTime),
    )
    op.create_table(
        "equity_curve",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("date", sa.DateTime(timezone=True), index=True),
        sa.Column("equity", sa.Float, nullable=False),
    )
    op.create_table(
        "training_tasks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("symbols", sa.Text, nullable=False),
        sa.Column("limit", sa.Integer, nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("details", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "settings",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("api_key", sa.String(255)),
        sa.Column("api_secret", sa.String(255)),
    )
    op.create_table(
        "alert_events",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("alert_id", sa.Integer, index=True, nullable=False),
        sa.Column("event", sa.String(64), nullable=False, server_default="trigger"),
        sa.Column("created_at", sa.DateTime(timezone=True), index=True),
    )


def downgrade() -> None:  # pragma: no cover
    for table in [
        "alert_events",
        "settings",
        "training_tasks",
        "equity_curve",
        "watchlist",
        "alerts",
        "model_registry",
        "candles",
        "trade_logs",
        "trades",
    ]:
        op.drop_table(table)
