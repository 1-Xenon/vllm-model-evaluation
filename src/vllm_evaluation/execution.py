"""Run frozen task snapshots and persist measured model attempts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .errors import ValidationError
from .models import (
    AttemptStatus,
    GenerationAttempt,
    ModelConfig,
    ModelRun,
    RunStatus,
    Task,
    TaskSnapshot,
    utc_now,
)
from .repositories import RunRepository
from .runner import (
    ModelRunner,
    RunnerAttempt,
    RunnerInput,
    RunnerRequest,
    RunnerResult,
    execute_with_retries,
)


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Execution controls that are recorded or supplied for one run."""

    asset_root: Path | None = None
    concurrency: int = 1
    warmup_count: int = 0
    max_retries: int = 0
    retry_backoff_seconds: float = 0.0
    retry_failed: bool = False


@dataclass(frozen=True, slots=True)
class RunProgress:
    run_id: str
    total_tasks: int
    completed_tasks: int
    succeeded_tasks: int
    failed_tasks: int
    truncated_tasks: int
    pending_tasks: int


@dataclass(frozen=True, slots=True)
class _TimedRunnerAttempts:
    attempts: list[RunnerAttempt]
    started_at: datetime
    finished_at: datetime


def validate_execution_config(config: ExecutionConfig) -> None:
    if config.concurrency < 1:
        raise ValidationError("execution concurrency must be at least one")
    if config.warmup_count < 0:
        raise ValidationError("warmup count cannot be negative")
    if config.max_retries < 0:
        raise ValidationError("max retries cannot be negative")
    if config.retry_backoff_seconds < 0:
        raise ValidationError("retry backoff cannot be negative")


def create_model_run(
    session: Session,
    *,
    snapshot: TaskSnapshot,
    model_config: ModelConfig,
    run_label: str | None = None,
    run_settings: dict[str, Any] | None = None,
) -> ModelRun:
    """Create a run only against an existing frozen task snapshot."""

    if not snapshot.id:
        raise ValidationError("task snapshot must have an identity before creating a run")
    return RunRepository().create_run(
        session,
        snapshot=snapshot,
        model_config=model_config,
        run_label=run_label,
        run_settings=run_settings,
    )


def _asset_path(asset_root: Path, relative_path: str) -> Path:
    root = asset_root.resolve()
    candidate = (root / Path(*PurePosixPath(relative_path).parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValidationError(f"asset path escapes configured asset root: {relative_path}") from exc
    if not candidate.is_file():
        raise ValidationError(f"asset does not exist at configured asset root: {relative_path}")
    return candidate


def build_runner_request(
    run: ModelRun,
    task: Task,
    *,
    attempt_number: int,
    asset_root: Path | None,
) -> RunnerRequest:
    """Convert a persisted task into a model-facing stateless request."""

    model_name = run.model_config.served_model_name
    inputs: list[RunnerInput] = []
    for task_input in sorted(task.inputs, key=lambda item: item.ordinal):
        if task_input.input_type == "text":
            inputs.append(RunnerInput(input_type="text", text=task_input.text_content))
            continue
        if task_input.input_type != "image" or task_input.media_asset is None:
            raise ValidationError(f"task input {task_input.id} has an invalid persisted payload")
        if asset_root is None:
            raise ValidationError("asset_root is required to execute image tasks")
        image_path = _asset_path(asset_root, task_input.media_asset.relative_path)
        inputs.append(
            RunnerInput(
                input_type="image",
                image_bytes=image_path.read_bytes(),
                image_media_type=task_input.media_asset.media_type,
            )
        )

    run_settings = run.run_settings_json or {}
    generation_settings = run_settings.get("generation_settings")
    if not isinstance(generation_settings, dict):
        generation_settings = run.model_config.generation_settings_json or {}
    stream = run_settings.get("stream", True)
    if not isinstance(stream, bool):
        raise ValidationError("run setting 'stream' must be a boolean")
    return RunnerRequest(
        request_id=f"{run.id}:{task.id}:{attempt_number}",
        model_name=model_name,
        prompt=task.prompt,
        inputs=tuple(inputs),
        system_instruction=task.system_instruction,
        generation_settings=generation_settings,
        stream=stream,
    )


def _execute_request(
    runner: ModelRunner,
    request: RunnerRequest,
    *,
    max_retries: int,
    retry_backoff_seconds: float,
) -> _TimedRunnerAttempts:
    started_at = utc_now()
    try:
        attempts = execute_with_retries(
            runner,
            request,
            max_retries=max_retries,
            backoff_seconds=retry_backoff_seconds,
        )
    except Exception as exc:  # Persist unexpected runner failures as explicit attempts.
        attempts = [
            RunnerAttempt(
                attempt_number=1,
                result=RunnerResult(
                    status=AttemptStatus.FAILED,
                    error_code="runner_exception",
                    error_message=str(exc),
                    retryable=False,
                ),
            )
        ]
    return _TimedRunnerAttempts(attempts=attempts, started_at=started_at, finished_at=utc_now())


def _token_count(result: RunnerResult) -> int | None:
    if not result.token_usage:
        return None
    for key in ("completion_tokens", "output_tokens", "generated_tokens"):
        value = result.token_usage.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def calculate_tps(result: RunnerResult) -> float | None:
    """Calculate TPS only when matching timing and token measurements exist."""

    token_count = _token_count(result)
    if token_count is None or token_count <= 1:
        return None
    if result.ttft_seconds is None or result.total_latency_seconds is None:
        return None
    denominator = result.total_latency_seconds - result.ttft_seconds
    if denominator <= 0:
        return None
    return (token_count - 1) / denominator


def _persist_runner_attempt(
    session: Session,
    run: ModelRun,
    task: Task,
    runner_attempt: RunnerAttempt,
    *,
    started_at: datetime,
    finished_at: datetime,
) -> GenerationAttempt:
    result = runner_attempt.result
    attempt = GenerationAttempt(
        run=run,
        task=task,
        attempt_number=runner_attempt.attempt_number,
        status=result.status.value,
        final_answer=result.final_answer,
        reasoning=result.reasoning,
        reasoning_status=result.reasoning_status.value,
        finish_reason=result.finish_reason,
        error_code=result.error_code,
        error_message=result.error_message,
        raw_response_json=result.raw_response,
        token_usage_json=result.token_usage,
        ttft_seconds=result.ttft_seconds,
        first_answer_seconds=result.first_answer_seconds,
        total_latency_seconds=result.total_latency_seconds,
        output_token_count=_token_count(result),
        tps=calculate_tps(result),
        started_at=started_at,
        finished_at=finished_at,
    )
    session.add(attempt)
    session.flush()
    return attempt


def _existing_attempts(session: Session, run: ModelRun, task: Task) -> list[GenerationAttempt]:
    return list(
        session.scalars(
            select(GenerationAttempt)
            .where(
                GenerationAttempt.run_id == run.id,
                GenerationAttempt.task_id == task.id,
            )
            .order_by(GenerationAttempt.attempt_number)
        )
    )


def _should_execute(attempts: list[GenerationAttempt], retry_failed: bool) -> bool:
    if not attempts:
        return True
    latest = attempts[-1]
    if latest.status in {AttemptStatus.SUCCEEDED.value, AttemptStatus.TRUNCATED.value}:
        return False
    if latest.status in {
        AttemptStatus.FAILED.value,
        AttemptStatus.TIMED_OUT.value,
        AttemptStatus.CANCELLED.value,
    }:
        return retry_failed
    return True


def progress_for_run(session: Session, run: ModelRun) -> RunProgress:
    """Return task-level progress without treating retries as extra tasks."""

    tasks = list(
        session.scalars(
            select(Task).where(Task.snapshot_id == run.snapshot_id).order_by(Task.task_order)
        )
    )
    succeeded = failed = truncated = completed = 0
    for task in tasks:
        attempts = _existing_attempts(session, run, task)
        if not attempts:
            continue
        latest = attempts[-1]
        if latest.status == AttemptStatus.SUCCEEDED.value:
            succeeded += 1
            completed += 1
        elif latest.status == AttemptStatus.TRUNCATED.value:
            truncated += 1
            completed += 1
        elif latest.status in {
            AttemptStatus.FAILED.value,
            AttemptStatus.TIMED_OUT.value,
            AttemptStatus.CANCELLED.value,
        }:
            failed += 1
            completed += 1
    return RunProgress(
        run_id=run.id,
        total_tasks=len(tasks),
        completed_tasks=completed,
        succeeded_tasks=succeeded,
        failed_tasks=failed,
        truncated_tasks=truncated,
        pending_tasks=len(tasks) - completed,
    )


def execute_run(
    session: Session,
    run: ModelRun,
    runner: ModelRunner,
    *,
    config: ExecutionConfig,
) -> RunProgress:
    """Execute a run, committing every completed task's attempts as progress."""

    validate_execution_config(config)
    if run.status == RunStatus.COMPLETED.value and not config.retry_failed:
        return progress_for_run(session, run)

    tasks = list(
        session.scalars(
            select(Task).where(Task.snapshot_id == run.snapshot_id).order_by(Task.task_order)
        )
    )
    if not tasks:
        raise ValidationError("cannot execute a run with no tasks")

    run.status = RunStatus.RUNNING.value
    run.started_at = run.started_at or utc_now()
    session.commit()

    if config.warmup_count:
        warmup_task = tasks[0]
        warmup_request = build_runner_request(
            run,
            warmup_task,
            attempt_number=0,
            asset_root=config.asset_root,
        )
        for _ in range(config.warmup_count):
            warmup_result = runner.generate(warmup_request)
            if warmup_result.status not in {
                AttemptStatus.SUCCEEDED,
                AttemptStatus.TRUNCATED,
            }:
                raise ValidationError(
                    "warm-up request failed: "
                    f"{warmup_result.error_code or warmup_result.status.value}"
                )
        run.warmup_count = config.warmup_count
        session.commit()

    work: list[tuple[Task, RunnerRequest]] = []
    for task in tasks:
        attempts = _existing_attempts(session, run, task)
        if not _should_execute(attempts, config.retry_failed):
            continue
        next_attempt_number = (attempts[-1].attempt_number + 1) if attempts else 1
        work.append(
            (
                task,
                build_runner_request(
                    run,
                    task,
                    attempt_number=next_attempt_number,
                    asset_root=config.asset_root,
                ),
            )
        )

    def persist(task: Task, timed_attempts: _TimedRunnerAttempts) -> None:
        for runner_attempt in timed_attempts.attempts:
            _persist_runner_attempt(
                session,
                run,
                task,
                runner_attempt,
                started_at=timed_attempts.started_at,
                finished_at=timed_attempts.finished_at,
            )
            session.commit()

    if config.concurrency == 1:
        for task, request in work:
            persist(
                task,
                _execute_request(
                    runner,
                    request,
                    max_retries=config.max_retries,
                    retry_backoff_seconds=config.retry_backoff_seconds,
                ),
            )
    else:
        with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
            futures = {
                executor.submit(
                    _execute_request,
                    runner,
                    request,
                    max_retries=config.max_retries,
                    retry_backoff_seconds=config.retry_backoff_seconds,
                ): task
                for task, request in work
            }
            for future in as_completed(futures):
                persist(futures[future], future.result())

    progress = progress_for_run(session, run)
    if progress.pending_tasks == 0:
        run.status = RunStatus.COMPLETED.value
        run.finished_at = utc_now()
    session.commit()
    return progress
