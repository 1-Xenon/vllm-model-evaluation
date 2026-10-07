"""Record application-controlled transformations per task input."""

import sqlalchemy as sa
from alembic import op

revision = "0002_task_input_transform_metadata"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("task_inputs")}
    if "transform_metadata_json" not in columns:
        op.add_column("task_inputs", sa.Column("transform_metadata_json", sa.JSON(), nullable=True))


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("task_inputs")}
    if "transform_metadata_json" in columns:
        op.drop_column("task_inputs", "transform_metadata_json")
