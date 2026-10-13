"""Persist automated evaluation configuration and progress."""

import sqlalchemy as sa
from alembic import op

revision = "0003_automation_jobs"
down_revision = "0002_task_input_transform_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "automation_jobs" in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        "automation_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("config_json", sa.JSON(), nullable=False),
        sa.Column("progress_json", sa.JSON(), nullable=False),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("error_message", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )


def downgrade() -> None:
    op.drop_table("automation_jobs")
