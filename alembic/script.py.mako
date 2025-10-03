"""Alembic script template."""
from __future__ import annotations
from alembic import op  # type: ignore
import sqlalchemy as sa  # type: ignore

revision = ${up_revision!r}
down_revision = ${down_revision!r}
branch_labels = ${branch_labels!r}
depends_on = ${depends_on!r}

def upgrade() -> None:  # noqa: D401
    pass

def downgrade() -> None:  # noqa: D401
    pass
