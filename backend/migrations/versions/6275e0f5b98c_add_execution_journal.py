"""add execution journal

Revision ID: 6275e0f5b98c
Revises: 3d4f0a9cba8e
Create Date: 2025-11-05 04:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "6275e0f5b98c"
down_revision: Union[str, Sequence[str], None] = "3d4f0a9cba8e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
	op.create_table(
		"execution_journal",
		sa.Column("id", sa.Integer(), primary_key=True),
		sa.Column("run_id", sa.Integer(), sa.ForeignKey("liquidity_runs.id", ondelete="SET NULL"), nullable=True),
		sa.Column("symbol", sa.String(length=24), nullable=False),
		sa.Column("side", sa.String(length=8), nullable=False),
		sa.Column("target_weight", sa.Float(), nullable=False),
		sa.Column("quantity", sa.Float(), nullable=False),
		sa.Column("status", sa.String(length=16), nullable=False, server_default=sa.text("'pending'")),
		sa.Column("reason", sa.Text(), nullable=True),
		sa.Column("error", sa.Text(), nullable=True),
		sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
		sa.Column("executed_at", sa.DateTime(timezone=True), nullable=True),
	)
	op.create_index(op.f("ix_execution_journal_run_id"), "execution_journal", ["run_id"], unique=False)
	op.create_index(op.f("ix_execution_journal_status"), "execution_journal", ["status"], unique=False)
	op.create_index(op.f("ix_execution_journal_symbol"), "execution_journal", ["symbol"], unique=False)


def downgrade() -> None:
	op.drop_index(op.f("ix_execution_journal_status"), table_name="execution_journal")
	op.drop_index(op.f("ix_execution_journal_run_id"), table_name="execution_journal")
	op.drop_table("execution_journal")
