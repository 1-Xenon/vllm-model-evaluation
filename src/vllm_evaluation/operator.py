"""Operator-facing workflow services for importing, running, and inspecting evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from .execution import ExecutionConfig, RunProgress, execute_run, progress_for_run
from .importer import import_task_bundle
from .models import AttemptStatus, GenerationAttempt, ModelConfig, ModelRun, Task, TaskSnapshot
from .repositories import RunRepository
from .runner import ModelRunner


@dataclass(frozen=True, slots=True)
class FailureRecord:
    task_id: str
    attempt_number: int
    status: str
    error_code: str | None
    error_message: str | None


def import_bundle(
    session: Session, bundle_root: str | Path, manifest_path: str = "manifest.jsonl"
) -> TaskSnapshot:
    """Import one validated immutable bundle snapshot."""

    return import_task_bundle(session, bundle_root=bundle_root, manifest_path=manifest_path)


def list_snapshots(session: Session) -> list[TaskSnapshot]:
    return list(session.scalars(select(TaskSnapshot).order_by(TaskSnapshot.created_at)))


def get_snapshot_detail(session: Session, snapshot_id: str) -> TaskSnapshot | None:
    return session.scalar(
        select(TaskSnapshot)
        .options(joinedload(TaskSnapshot.tasks).joinedload(Task.inputs))
        .where(TaskSnapshot.id == snapshot_id)
    )


def create_model_config(session: Session, payload: dict[str, Any]) -> ModelConfig:
    required = ("display_name", "endpoint_url", "served_model_name")
    missing = [key for key in required if not isinstance(payload.get(key), str) or not payload[key]]
    if missing:
        raise ValueError(f"missing model configuration fields: {', '.join(missing)}")
    return RunRepository().create_model_config(
        session,
        display_name=payload["display_name"],
        endpoint_url=payload["endpoint_url"],
        served_model_name=payload["served_model_name"],
        checkpoint_identity=payload.get("checkpoint_identity"),
        endpoint_reported_identity=payload.get("endpoint_reported_identity"),
        auth_reference=payload.get("auth_reference"),
        generation_settings=payload.get("generation_settings"),
        serving_environment=payload.get("serving_environment"),
    )


def create_run(session: Session, payload: dict[str, Any]) -> ModelRun:
    snapshot = session.get(TaskSnapshot, payload.get("snapshot_id"))
    model_config = session.get(ModelConfig, payload.get("model_config_id"))
    if snapshot is None or model_config is None:
        raise ValueError("snapshot_id and model_config_id must identify existing records")
    return RunRepository().create_run(
        session,
        snapshot=snapshot,
        model_config=model_config,
        run_label=payload.get("run_label"),
        run_settings=payload.get("run_settings"),
    )


def execute_config_from_settings(
    settings: Any, *, asset_root: Path | None = None, retry_failed: bool = False
) -> ExecutionConfig:
    return ExecutionConfig(
        asset_root=asset_root,
        concurrency=settings.runner_concurrency,
        warmup_count=settings.runner_warmup_count,
        max_retries=settings.runner_max_retries,
        retry_backoff_seconds=settings.runner_retry_backoff_seconds,
        retry_failed=retry_failed,
    )


def execute_existing_run(
    session: Session,
    run_id: str,
    runner: ModelRunner,
    *,
    config: ExecutionConfig,
) -> RunProgress:
    run = session.get(ModelRun, run_id)
    if run is None:
        raise ValueError(f"model run not found: {run_id}")
    return execute_run(session, run, runner, config=config)


def inspect_failures(session: Session, run_id: str) -> list[FailureRecord]:
    attempts = session.scalars(
        select(GenerationAttempt)
        .join(Task, GenerationAttempt.task_id == Task.id)
        .where(GenerationAttempt.run_id == run_id)
        .where(
            GenerationAttempt.status.in_(
                [
                    status.value
                    for status in (
                        AttemptStatus.FAILED,
                        AttemptStatus.TIMED_OUT,
                        AttemptStatus.TRUNCATED,
                    )
                ]
            )
        )
        .order_by(Task.task_order, GenerationAttempt.attempt_number)
    )
    return [
        FailureRecord(
            task_id=attempt.task.stable_task_id,
            attempt_number=attempt.attempt_number,
            status=attempt.status,
            error_code=attempt.error_code,
            error_message=attempt.error_message,
        )
        for attempt in attempts
    ]


def run_progress(session: Session, run_id: str) -> RunProgress:
    run = session.get(ModelRun, run_id)
    if run is None:
        raise ValueError(f"model run not found: {run_id}")
    return progress_for_run(session, run)
