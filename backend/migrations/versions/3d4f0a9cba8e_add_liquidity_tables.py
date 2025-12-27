"""add liquidity tables

Revision ID: 3d4f0a9cba8e
Revises: 1e6e6c7af2f3
Create Date: 2025-11-05 01:12:04.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "3d4f0a9cba8e"
down_revision: Union[str, Sequence[str], None] = "1e6e6c7af2f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
	"""Create liquidity tracking tables."""

	op.create_table(
		"liquidity_runs",
		sa.Column("id", sa.Integer(), primary_key=True),
		sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
		sa.Column("universe_size", sa.Integer(), nullable=False),
		sa.Column("selection_size", sa.Integer(), nullable=False),
		sa.Column("provider_primary", sa.String(length=32), nullable=True),
		sa.Column("status", sa.String(length=32), nullable=False, server_default=sa.text("'completed'")),
		sa.Column("message", sa.Text(), nullable=True),
	)
	op.create_index(op.f("ix_liquidity_runs_fetched_at"), "liquidity_runs", ["fetched_at"], unique=False)

	op.create_table(
		"liquidity_snapshots",
		sa.Column("id", sa.Integer(), primary_key=True),
		sa.Column("run_id", sa.Integer(), sa.ForeignKey("liquidity_runs.id", ondelete="CASCADE"), nullable=False),
		sa.Column("rank", sa.Integer(), nullable=False),
		sa.Column("symbol", sa.String(length=24), nullable=False),
		sa.Column("price", sa.Float(), nullable=True),
		sa.Column("change_percent", sa.Float(), nullable=True),
		sa.Column("base_volume", sa.Float(), nullable=True),
		sa.Column("quote_volume", sa.Float(), nullable=True),
		sa.Column("market_cap", sa.Float(), nullable=True),
		sa.Column("liquidity_score", sa.Float(), nullable=False),
		sa.Column("momentum_score", sa.Float(), nullable=True),
		sa.Column("aggregate_score", sa.Float(), nullable=False),
		sa.Column("payload", sa.Text(), nullable=True),
		sa.UniqueConstraint("run_id", "symbol", name="uq_liquidity_snapshot_symbol"),
	)
	op.create_index(op.f("ix_liquidity_snapshots_run_id"), "liquidity_snapshots", ["run_id"], unique=False)
	op.create_index(op.f("ix_liquidity_snapshots_symbol"), "liquidity_snapshots", ["symbol"], unique=False)

	op.create_table(
		"portfolio_allocations",
		sa.Column("id", sa.Integer(), primary_key=True),
		sa.Column("run_id", sa.Integer(), sa.ForeignKey("liquidity_runs.id", ondelete="CASCADE"), nullable=False),
		sa.Column("symbol", sa.String(length=24), nullable=False),
		sa.Column("weight", sa.Float(), nullable=False),
		sa.Column("score", sa.Float(), nullable=False),
		sa.Column("reason", sa.Text(), nullable=True),
		sa.UniqueConstraint("run_id", "symbol", name="uq_portfolio_allocation_symbol"),
	)
	op.create_index(op.f("ix_portfolio_allocations_run_id"), "portfolio_allocations", ["run_id"], unique=False)
	op.create_index(op.f("ix_portfolio_allocations_symbol"), "portfolio_allocations", ["symbol"], unique=False)


def downgrade() -> None:
	"""Drop liquidity tracking tables."""

	op.drop_index(op.f("ix_portfolio_allocations_symbol"), table_name="portfolio_allocations")
	op.drop_index(op.f("ix_portfolio_allocations_run_id"), table_name="portfolio_allocations")
	op.drop_table("portfolio_allocations")

	op.drop_index(op.f("ix_liquidity_snapshots_symbol"), table_name="liquidity_snapshots")
	op.drop_index(op.f("ix_liquidity_snapshots_run_id"), table_name="liquidity_snapshots")
	op.drop_table("liquidity_snapshots")

	op.drop_index(op.f("ix_liquidity_runs_fetched_at"), table_name="liquidity_runs")
	op.drop_table("liquidity_runs")
