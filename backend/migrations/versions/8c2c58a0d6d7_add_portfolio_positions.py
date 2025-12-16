"""Add portfolio positions table for execution exposure tracking."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "8c2c58a0d6d7"
down_revision = "6275e0f5b98c"
branch_labels = None
depends_on = None


def upgrade() -> None:
	op.create_table(
		"portfolio_positions",
		sa.Column("id", sa.Integer(), primary_key=True),
		sa.Column("symbol", sa.String(length=24), nullable=False),
		sa.Column("quantity", sa.Float(), nullable=False, server_default=sa.text("0")),
		sa.Column("notional", sa.Float(), nullable=False, server_default=sa.text("0")),
		sa.Column(
			"updated_at",
			sa.DateTime(),
			nullable=False,
			server_default=sa.text("CURRENT_TIMESTAMP"),
		),
		sa.UniqueConstraint("symbol", name="uq_portfolio_position_symbol"),
	)


def downgrade() -> None:
	op.drop_table("portfolio_positions")
