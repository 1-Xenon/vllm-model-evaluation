"""SQLAlchemy persistence models for the evaluation evidence chain."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id() -> str:
    """Return a portable string identifier for a persisted record."""

    return str(uuid4())


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Base class for all persisted models."""


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AttemptStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    TRUNCATED = "truncated"
    CANCELLED = "cancelled"


class ComparisonStatus(StrEnum):
    DRAFT = "draft"
    FROZEN = "frozen"
    OPEN = "open"
    CLOSED = "closed"


class AssignmentStatus(StrEnum):
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    SKIPPED = "skipped"


class BaseModel:
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class TaskSnapshot(Base, BaseModel):
    __tablename__ = "task_snapshots"
    __table_args__ = (UniqueConstraint("snapshot_hash", name="uq_task_snapshots_hash"),)

    source_name: Mapped[str | None] = mapped_column(String(255))
    manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    task_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    notes_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    tasks: Mapped[list[Task]] = relationship(back_populates="snapshot")
    model_runs: Mapped[list[ModelRun]] = relationship(back_populates="snapshot")
    comparison_sessions: Mapped[list[ComparisonSession]] = relationship(back_populates="snapshot")


class Task(Base, BaseModel):
    __tablename__ = "tasks"
    __table_args__ = (
        UniqueConstraint("snapshot_id", "stable_task_id", name="uq_tasks_snapshot_stable_id"),
        Index("ix_tasks_snapshot_order", "snapshot_id", "task_order"),
    )

    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("task_snapshots.id", ondelete="CASCADE"), nullable=False
    )
    stable_task_id: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    system_instruction: Mapped[str | None] = mapped_column(Text)
    operator_notes: Mapped[str | None] = mapped_column(Text)
    source_metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    task_order: Mapped[int] = mapped_column(Integer, nullable=False)

    snapshot: Mapped[TaskSnapshot] = relationship(back_populates="tasks")
    inputs: Mapped[list[TaskInput]] = relationship(
        back_populates="task", cascade="all, delete-orphan", order_by="TaskInput.ordinal"
    )
    attempts: Mapped[list[GenerationAttempt]] = relationship(back_populates="task")
    comparison_items: Mapped[list[ComparisonItem]] = relationship(back_populates="task")


class MediaAsset(Base, BaseModel):
    __tablename__ = "media_assets"
    __table_args__ = (
        UniqueConstraint("snapshot_id", "relative_path", name="uq_media_assets_snapshot_path"),
        Index("ix_media_assets_content_hash", "content_hash"),
    )

    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("task_snapshots.id", ondelete="CASCADE"), nullable=False
    )
    relative_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(255))
    media_type: Mapped[str] = mapped_column(String(100), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    transform_metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    inputs: Mapped[list[TaskInput]] = relationship(back_populates="media_asset")


class TaskInput(Base, BaseModel):
    __tablename__ = "task_inputs"
    __table_args__ = (
        UniqueConstraint("task_id", "ordinal", name="uq_task_inputs_task_ordinal"),
        CheckConstraint(
            "(input_type = 'text' AND text_content IS NOT NULL AND media_asset_id IS NULL) OR "
            "(input_type = 'image' AND text_content IS NULL AND media_asset_id IS NOT NULL)",
            name="ck_task_inputs_payload_matches_type",
        ),
    )

    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    input_type: Mapped[str] = mapped_column(String(20), nullable=False)
    text_content: Mapped[str | None] = mapped_column(Text)
    transform_metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    media_asset_id: Mapped[str | None] = mapped_column(
        ForeignKey("media_assets.id", ondelete="RESTRICT")
    )

    task: Mapped[Task] = relationship(back_populates="inputs")
    media_asset: Mapped[MediaAsset | None] = relationship(back_populates="inputs")


class ModelConfig(Base, BaseModel):
    __tablename__ = "model_configs"

    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    endpoint_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    served_model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    checkpoint_identity: Mapped[str | None] = mapped_column(String(1024))
    endpoint_reported_identity: Mapped[str | None] = mapped_column(String(1024))
    auth_reference: Mapped[str | None] = mapped_column(String(255))
    generation_settings_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    serving_environment_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    runs: Mapped[list[ModelRun]] = relationship(back_populates="model_config")


class ModelRun(Base, BaseModel):
    __tablename__ = "model_runs"
    __table_args__ = (Index("ix_model_runs_snapshot_status", "snapshot_id", "status"),)

    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("task_snapshots.id", ondelete="RESTRICT"), nullable=False
    )
    model_config_id: Mapped[str] = mapped_column(
        ForeignKey("model_configs.id", ondelete="RESTRICT"), nullable=False
    )
    run_label: Mapped[str | None] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=RunStatus.CREATED)
    run_settings_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    warmup_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    snapshot: Mapped[TaskSnapshot] = relationship(back_populates="model_runs")
    model_config: Mapped[ModelConfig] = relationship(back_populates="runs")
    attempts: Mapped[list[GenerationAttempt]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class GenerationAttempt(Base, BaseModel):
    __tablename__ = "generation_attempts"
    __table_args__ = (
        UniqueConstraint("run_id", "task_id", "attempt_number", name="uq_attempts_run_task_number"),
        Index("ix_attempts_run_status", "run_id", "status"),
    )

    run_id: Mapped[str] = mapped_column(
        ForeignKey("model_runs.id", ondelete="CASCADE"), nullable=False
    )
    task_id: Mapped[str] = mapped_column(
        ForeignKey("tasks.id", ondelete="RESTRICT"), nullable=False
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=AttemptStatus.CREATED)
    final_answer: Mapped[str | None] = mapped_column(Text)
    reasoning: Mapped[str | None] = mapped_column(Text)
    reasoning_status: Mapped[str | None] = mapped_column(String(32))
    finish_reason: Mapped[str | None] = mapped_column(String(100))
    error_code: Mapped[str | None] = mapped_column(String(100))
    error_message: Mapped[str | None] = mapped_column(Text)
    raw_response_json: Mapped[dict[str, Any] | list[Any] | None] = mapped_column(JSON)
    token_usage_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    ttft_seconds: Mapped[float | None]
    first_answer_seconds: Mapped[float | None]
    total_latency_seconds: Mapped[float | None]
    output_token_count: Mapped[int | None]
    tps: Mapped[float | None]
    token_count_includes_reasoning: Mapped[bool | None] = mapped_column(Boolean)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    run: Mapped[ModelRun] = relationship(back_populates="attempts")
    task: Mapped[Task] = relationship(back_populates="attempts")


class ComparisonSession(Base, BaseModel):
    __tablename__ = "comparison_sessions"
    __table_args__ = (Index("ix_comparison_sessions_snapshot_status", "snapshot_id", "status"),)

    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("task_snapshots.id", ondelete="RESTRICT"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=ComparisonStatus.DRAFT)
    created_by: Mapped[str | None] = mapped_column(String(255))
    assignment_seed: Mapped[str] = mapped_column(String(255), nullable=False)
    frozen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    snapshot: Mapped[TaskSnapshot] = relationship(back_populates="comparison_sessions")
    items: Mapped[list[ComparisonItem]] = relationship(
        back_populates="comparison_session", cascade="all, delete-orphan"
    )


class ComparisonItem(Base, BaseModel):
    __tablename__ = "comparison_items"
    __table_args__ = (
        UniqueConstraint(
            "comparison_session_id", "task_id", name="uq_comparison_items_session_task"
        ),
    )

    comparison_session_id: Mapped[str] = mapped_column(
        ForeignKey("comparison_sessions.id", ondelete="CASCADE"), nullable=False
    )
    task_id: Mapped[str] = mapped_column(
        ForeignKey("tasks.id", ondelete="RESTRICT"), nullable=False
    )
    left_attempt_id: Mapped[str] = mapped_column(
        ForeignKey("generation_attempts.id", ondelete="RESTRICT"), nullable=False
    )
    right_attempt_id: Mapped[str] = mapped_column(
        ForeignKey("generation_attempts.id", ondelete="RESTRICT"), nullable=False
    )

    comparison_session: Mapped[ComparisonSession] = relationship(back_populates="items")
    task: Mapped[Task] = relationship(back_populates="comparison_items")
    left_attempt: Mapped[GenerationAttempt] = relationship(foreign_keys=[left_attempt_id])
    right_attempt: Mapped[GenerationAttempt] = relationship(foreign_keys=[right_attempt_id])
    assignments: Mapped[list[ReviewerAssignment]] = relationship(
        back_populates="comparison_item", cascade="all, delete-orphan"
    )


class ReviewerAssignment(Base, BaseModel):
    __tablename__ = "reviewer_assignments"
    __table_args__ = (
        UniqueConstraint(
            "comparison_item_id", "reviewer_id", name="uq_reviewer_assignments_item_reviewer"
        ),
    )

    comparison_item_id: Mapped[str] = mapped_column(
        ForeignKey("comparison_items.id", ondelete="CASCADE"), nullable=False
    )
    reviewer_id: Mapped[str] = mapped_column(String(255), nullable=False)
    left_attempt_id: Mapped[str] = mapped_column(
        ForeignKey("generation_attempts.id", ondelete="RESTRICT"), nullable=False
    )
    right_attempt_id: Mapped[str] = mapped_column(
        ForeignKey("generation_attempts.id", ondelete="RESTRICT"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=AssignmentStatus.ASSIGNED
    )
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    comparison_item: Mapped[ComparisonItem] = relationship(back_populates="assignments")
    left_attempt: Mapped[GenerationAttempt] = relationship(foreign_keys=[left_attempt_id])
    right_attempt: Mapped[GenerationAttempt] = relationship(foreign_keys=[right_attempt_id])
    judgments: Mapped[list[ReviewerJudgment]] = relationship(
        back_populates="assignment",
        cascade="all, delete-orphan",
        order_by="ReviewerJudgment.version",
    )


class ReviewerJudgment(Base, BaseModel):
    __tablename__ = "reviewer_judgments"
    __table_args__ = (
        UniqueConstraint(
            "assignment_id", "version", name="uq_reviewer_judgments_assignment_version"
        ),
    )

    assignment_id: Mapped[str] = mapped_column(
        ForeignKey("reviewer_assignments.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    overall_preference: Mapped[str | None] = mapped_column(String(32))
    left_acceptability: Mapped[str | None] = mapped_column(String(32))
    right_acceptability: Mapped[str | None] = mapped_column(String(32))
    disposition: Mapped[str | None] = mapped_column(String(32))
    error_tags_json: Mapped[list[str] | None] = mapped_column(JSON)
    comments: Mapped[str | None] = mapped_column(Text)
    reasoning_assessment_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    assignment: Mapped[ReviewerAssignment] = relationship(back_populates="judgments")
