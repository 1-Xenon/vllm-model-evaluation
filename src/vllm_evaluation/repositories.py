"""Small persistence services for Phase 1 records."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from .errors import ValidationError
from .models import (
    ComparisonSession,
    GenerationAttempt,
    ModelConfig,
    ModelRun,
    Task,
    TaskSnapshot,
)


class TaskSnapshotRepository:
    """Persistence operations for immutable task snapshots."""

    def create(
        self,
        session: Session,
        *,
        source_name: str | None,
        manifest_hash: str,
        snapshot_hash: str,
        task_count: int = 0,
        notes: dict | None = None,
    ) -> TaskSnapshot:
        snapshot = TaskSnapshot(
            source_name=source_name,
            manifest_hash=manifest_hash,
            snapshot_hash=snapshot_hash,
            task_count=task_count,
            notes_json=notes,
        )
        session.add(snapshot)
        session.flush()
        return snapshot

    def get(self, session: Session, snapshot_id: str) -> TaskSnapshot | None:
        return session.get(TaskSnapshot, snapshot_id)

    def list(self, session: Session) -> list[TaskSnapshot]:
        return list(session.scalars(select(TaskSnapshot).order_by(TaskSnapshot.created_at)))


class RunRepository:
    """Persistence operations for model configurations, runs, and attempts."""

    def create_model_config(
        self,
        session: Session,
        *,
        display_name: str,
        endpoint_url: str,
        served_model_name: str,
        checkpoint_identity: str | None = None,
        endpoint_reported_identity: str | None = None,
        auth_reference: str | None = None,
        generation_settings: dict | None = None,
        serving_environment: dict | None = None,
    ) -> ModelConfig:
        model_config = ModelConfig(
            display_name=display_name,
            endpoint_url=endpoint_url,
            served_model_name=served_model_name,
            checkpoint_identity=checkpoint_identity,
            endpoint_reported_identity=endpoint_reported_identity,
            auth_reference=auth_reference,
            generation_settings_json=generation_settings,
            serving_environment_json=serving_environment,
        )
        session.add(model_config)
        session.flush()
        return model_config

    def create_run(
        self,
        session: Session,
        *,
        snapshot: TaskSnapshot,
        model_config: ModelConfig,
        run_label: str | None = None,
        run_settings: dict | None = None,
    ) -> ModelRun:
        run = ModelRun(
            snapshot=snapshot,
            model_config=model_config,
            run_label=run_label,
            run_settings_json=run_settings,
        )
        session.add(run)
        session.flush()
        return run

    def create_attempt(
        self,
        session: Session,
        *,
        run: ModelRun,
        task: Task,
        attempt_number: int,
    ) -> GenerationAttempt:
        if run.snapshot_id != task.snapshot_id:
            raise ValidationError("run and task must reference the same task snapshot")
        if attempt_number < 1:
            raise ValidationError("attempt number must be at least one")
        attempt = GenerationAttempt(
            run=run,
            task=task,
            attempt_number=attempt_number,
        )
        session.add(attempt)
        session.flush()
        return attempt


class ComparisonRepository:
    """Persistence operations for comparison sessions."""

    def create_session(
        self,
        session: Session,
        *,
        snapshot: TaskSnapshot,
        name: str,
        assignment_seed: str,
        created_by: str | None = None,
    ) -> ComparisonSession:
        comparison = ComparisonSession(
            snapshot=snapshot,
            name=name,
            assignment_seed=assignment_seed,
            created_by=created_by,
        )
        session.add(comparison)
        session.flush()
        return comparison


def ensure_unique_task_ids(tasks: Sequence[Task]) -> None:
    """Validate that a snapshot's stable task IDs are unique."""

    task_ids = [task.stable_task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValidationError("stable task IDs must be unique within a task snapshot")
