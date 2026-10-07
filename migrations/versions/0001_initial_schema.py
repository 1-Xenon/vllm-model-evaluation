"""Create the initial Phase 1 evaluation evidence schema."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy import MetaData

from vllm_evaluation.models import Base

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    phase_one_metadata = MetaData()
    for table in Base.metadata.sorted_tables:
        if table.name != "task_inputs":
            table.to_metadata(phase_one_metadata)
    phase_one_metadata.create_all(bind=bind)

    op.create_table(
        "task_inputs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("input_type", sa.String(length=20), nullable=False),
        sa.Column("text_content", sa.Text(), nullable=True),
        sa.Column("media_asset_id", sa.String(length=36), nullable=True),
        sa.ForeignKeyConstraint(["media_asset_id"], ["media_assets.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("task_id", "ordinal", name="uq_task_inputs_task_ordinal"),
        sa.CheckConstraint(
            "(input_type = 'text' AND text_content IS NOT NULL AND media_asset_id IS NULL) OR "
            "(input_type = 'image' AND text_content IS NULL AND media_asset_id IS NOT NULL)",
            name="ck_task_inputs_payload_matches_type",
        ),
    )


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
