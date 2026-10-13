"""Persist and execute sequential evaluations with one managed GPU model at a time."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from .config import Settings
from .db import session_scope
from .execution import ExecutionConfig, execute_run, progress_for_run
from .importer import import_task_bundle
from .models import AutomationJob, GenerationAttempt, ModelConfig, ModelRun, TaskSnapshot, utc_now
from .repositories import RunRepository
from .review import create_comparison, freeze_comparison
from .runner_factory import create_runner

ACTIVE_STATUSES = ("queued", "running", "cancelling")


class AutomationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    bundles: list[str] = Field(min_length=1, max_length=1)
    model_config_ids: list[str] = Field(min_length=2, max_length=20)
    swap_models: bool = True
    ready_timeout_seconds: int = Field(default=900, ge=10, le=7200)
    assignment_seed: str = Field(default="automation", min_length=1, max_length=200)


class AutomationBusy(ValueError):
    pass


class AutomationCancelled(Exception):
    pass


class ExecutionLease:
    """Share a host-level exclusive lock with manual executions and other app workers."""

    def __init__(self, path: Path):
        self.file = path.open("a")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.file.close()
            raise AutomationBusy("an evaluation is already running; wait for it to finish") from exc

    def close(self) -> None:
        if not self.file.closed:
            fcntl.flock(self.file, fcntl.LOCK_UN)
            self.file.close()


def models_url(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("model endpoints must be HTTP or HTTPS URLs")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("model endpoints must not contain credentials, queries, or fragments")
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path.removesuffix("/chat/completions")
    return urlunsplit((parsed.scheme, parsed.netloc, path + "/models", "", ""))


class LocalModelServer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def validate(self, models: list[ModelConfig], swap_models: bool) -> None:
        endpoints = [models_url(model.endpoint_url) for model in models]
        if not swap_models:
            return
        for endpoint in endpoints:
            parsed = urlsplit(endpoint)
            if parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.scheme != "http":
                raise ValueError("automatic swapping requires a local HTTP model endpoint")
            if parsed.path != "/v1/models" or parsed.port == self.settings.port:
                raise ValueError("automatic swapping requires the local vLLM /v1 endpoint")
        if len({urlsplit(endpoint).port or 80 for endpoint in endpoints}) != 1:
            raise ValueError("models must use the same local port for automatic swapping")
        if not self.settings.automation_serve_script.resolve().is_file():
            raise ValueError("model-serving script not found; configure automation.serve_script")
        if not shutil.which("docker"):
            raise ValueError("automatic swapping requires Docker on the evaluation app host")
        try:
            result = subprocess.run(
                ["docker", "info"], capture_output=True, text=True, timeout=15, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(f"cannot connect to Docker: {exc}") from exc
        if result.returncode:
            raise ValueError("Docker is unavailable to the app: " + result.stderr[-2000:])

    def _ready(self, model: ModelConfig) -> bool:
        headers = {}
        token = os.environ.get(self.settings.runner_auth_token_env, "")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        with urlopen(Request(models_url(model.endpoint_url), headers=headers), timeout=5) as reply:
            payload = json.load(reply)
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            return False
        return any(
            isinstance(item, dict) and item.get("id") == model.served_model_name
            for item in payload.get("data", [])
        )

    def prepare(self, model: ModelConfig, *, swap_models: bool, timeout: int, cancelled) -> None:
        if cancelled():
            raise AutomationCancelled()
        try:
            if self._ready(model):
                return
        except (OSError, URLError, ValueError):
            pass
        if swap_models:
            script = self.settings.automation_serve_script.resolve()
            port = urlsplit(model.endpoint_url).port or 80
            result = subprocess.run(
                [
                    "bash",
                    str(script),
                    "--model",
                    model.served_model_name,
                    "--port",
                    str(port),
                    "--detach",
                ],
                cwd=script.parent.parent,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if result.returncode:
                raise RuntimeError(
                    "model startup failed: " + (result.stderr or result.stdout)[-4000:]
                )
        deadline = time.monotonic() + timeout
        last_error = "endpoint has not listed the requested model"
        while time.monotonic() < deadline:
            if cancelled():
                raise AutomationCancelled()
            try:
                if self._ready(model):
                    return
            except (OSError, URLError, ValueError) as exc:
                last_error = str(exc)
            time.sleep(1)
        raise TimeoutError(f"model {model.served_model_name} was not ready: {last_error}")


class AutomationManager:
    def __init__(self, settings: Settings, factory, *, server=None, runner_factory=create_runner):
        self.settings = settings
        self.factory = factory
        self.server = server or LocalModelServer(settings)
        self.runner_factory = runner_factory
        self.lock_path = settings.data_root.resolve() / "evaluation-execution.lock"
        self._recover_interrupted()

    @contextmanager
    def execution_slot(self):
        lease = ExecutionLease(self.lock_path)
        try:
            yield
        finally:
            lease.close()

    def _recover_interrupted(self) -> None:
        try:
            with self.execution_slot(), session_scope(self.factory) as session:
                for job in session.scalars(
                    select(AutomationJob).where(AutomationJob.status.in_(ACTIVE_STATUSES))
                ):
                    job.status = "interrupted"
                    job.error_message = (
                        "App stopped during automation. Completed results are saved; "
                        "start a new job to rerun."
                    )
                    details = {**deepcopy(job.progress_json), "stage": "Interrupted"}
                    for item in details.get("runs", []):
                        run = session.get(ModelRun, item["run_id"])
                        if run and run.status == "running":
                            run.status = "failed"
                            run.finished_at = utc_now()
                            item["status"] = "interrupted"
                    job.progress_json = details
                    job.finished_at = utc_now()
        except AutomationBusy:
            pass

    def _serialize(self, job: AutomationJob) -> dict[str, Any]:
        return {
            "id": job.id,
            "name": job.name,
            "status": job.status,
            "config": job.config_json,
            "progress": job.progress_json,
            "error_message": job.error_message,
            "cancel_requested": job.cancel_requested,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        }

    def list_jobs(self) -> list[dict[str, Any]]:
        self._recover_interrupted()
        with self.factory() as session:
            return [
                self._serialize(job)
                for job in session.scalars(
                    select(AutomationJob).order_by(AutomationJob.created_at.desc()).limit(50)
                )
            ]

    def get(self, job_id: str) -> dict[str, Any]:
        self._recover_interrupted()
        with self.factory() as session:
            job = session.get(AutomationJob, job_id)
            if job is None:
                raise ValueError("automation job not found")
            result = self._serialize(job)
            current = job.progress_json.get("current_run_id")
            if current:
                run = session.get(ModelRun, current)
                result["current_run_progress"] = asdict(progress_for_run(session, run))
            return result

    def start(self, request: AutomationRequest) -> dict[str, Any]:
        lease = ExecutionLease(self.lock_path)
        try:
            if self.settings.runner_type != "openai_compatible":
                raise ValueError("automation requires the live runner; disable fake mode first")
            if not request.name.strip():
                raise ValueError("automation name must not be blank")
            if len(set(request.model_config_ids)) != len(request.model_config_ids):
                raise ValueError("select at least two distinct model configurations")
            with session_scope(self.factory) as session:
                models = [
                    session.get(ModelConfig, model_id) for model_id in request.model_config_ids
                ]
                if any(model is None for model in models):
                    raise ValueError("selected model configuration no longer exists")
                self.server.validate(models, request.swap_models)
                bundles = []
                snapshot_ids = set()
                parent = self.settings.task_bundle_root.resolve()
                for selected in request.bundles:
                    root = Path(selected).resolve()
                    if root == parent or not root.is_relative_to(parent):
                        raise ValueError("bundles must be inside the configured task-bundle parent")
                    manifest = next(
                        (
                            name
                            for name in ("manifest.jsonl", "tasks.jsonl")
                            if (root / name).is_file()
                        ),
                        None,
                    )
                    if manifest is None:
                        raise ValueError(f"no task manifest in bundle {root.name}")
                    snapshot = import_task_bundle(session, bundle_root=root, manifest_path=manifest)
                    if not snapshot.task_count:
                        raise ValueError(f"bundle {root.name} has no tasks")
                    if snapshot.id in snapshot_ids:
                        raise ValueError("selected bundles contain duplicate snapshots")
                    snapshot_ids.add(snapshot.id)
                    bundles.append(
                        {
                            "name": root.name,
                            "root": str(root),
                            "snapshot_id": snapshot.id,
                            "task_count": snapshot.task_count,
                        }
                    )
                job = AutomationJob(
                    name=request.name.strip(),
                    status="queued",
                    config_json=request.model_dump(),
                    progress_json={
                        "stage": "Queued",
                        "bundles": bundles,
                        "runs": [],
                        "comparisons": [],
                        "events": [],
                    },
                )
                session.add(job)
                session.flush()
                runs = []
                repo = RunRepository()
                for model in models:
                    for bundle in bundles:
                        snapshot = session.get(TaskSnapshot, bundle["snapshot_id"])
                        run = repo.create_run(
                            session,
                            snapshot=snapshot,
                            model_config=model,
                            run_label=f"{job.name}: {bundle['name']} / {model.display_name}",
                            run_settings={
                                "stream": self.settings.runner_streaming,
                                "automation_id": job.id,
                            },
                        )
                        runs.append(
                            {
                                "run_id": run.id,
                                "model_config_id": model.id,
                                "model_name": model.display_name,
                                "bundle_name": bundle["name"],
                                "snapshot_id": snapshot.id,
                                "asset_root": bundle["root"],
                                "status": "created",
                            }
                        )
                job.progress_json = {**job.progress_json, "runs": runs}
                result = self._serialize(job)
            worker = threading.Thread(target=self._work, args=(job.id, lease), daemon=True)
            try:
                worker.start()
            except Exception as exc:
                self._update(job.id, status="failed", error_message=str(exc), finished_at=utc_now())
                raise
            return result
        except Exception:
            lease.close()
            raise

    def cancel(self, job_id: str) -> dict[str, Any]:
        with session_scope(self.factory) as session:
            job = session.get(AutomationJob, job_id)
            if job is None:
                raise ValueError("automation job not found")
            if job.status in ACTIVE_STATUSES:
                job.cancel_requested = True
                job.status = "cancelling"
            return self._serialize(job)

    def _cancelled(self, job_id: str) -> bool:
        with self.factory() as session:
            return session.get(AutomationJob, job_id).cancel_requested

    def _update(self, job_id: str, *, stage: str | None = None, progress=None, **fields) -> None:
        with session_scope(self.factory) as session:
            job = session.get(AutomationJob, job_id)
            for key, value in fields.items():
                setattr(job, key, value)
            details = {**job.progress_json, **(progress or {})}
            if stage:
                details["stage"] = stage
                details["events"] = [
                    *details.get("events", []),
                    {"at": utc_now().isoformat(), "message": stage},
                ][-200:]
            job.progress_json = details

    def _work(self, job_id: str, lease: ExecutionLease) -> None:
        current_run_id = None
        try:
            with self.factory() as session:
                job = session.get(AutomationJob, job_id)
                config = AutomationRequest(**job.config_json)
                details = job.progress_json
            runs = details["runs"]
            self._update(job_id, status="running", started_at=utc_now(), stage="Starting")
            for model_id in config.model_config_ids:
                if self._cancelled(job_id):
                    raise AutomationCancelled()
                with self.factory() as session:
                    model = session.get(ModelConfig, model_id)
                self._update(
                    job_id,
                    stage=f"Preparing {model.display_name}",
                    progress={"current_run_id": None},
                )
                self.server.prepare(
                    model,
                    swap_models=config.swap_models,
                    timeout=config.ready_timeout_seconds,
                    cancelled=lambda: self._cancelled(job_id),
                )
                runner = self.runner_factory(
                    replace(
                        self.settings,
                        runner_endpoint_url=model.endpoint_url,
                        runner_model_name=model.served_model_name,
                    )
                )
                for item in runs:
                    if item["model_config_id"] != model_id:
                        continue
                    if self._cancelled(job_id):
                        raise AutomationCancelled()
                    current_run_id = item["run_id"]
                    item["status"] = "running"
                    self._update(
                        job_id,
                        stage=f"Running {item['bundle_name']} on {model.display_name}",
                        progress={"current_run_id": current_run_id, "runs": runs},
                    )
                    with session_scope(self.factory) as session:
                        run = session.get(ModelRun, current_run_id)
                        progress = execute_run(
                            session,
                            run,
                            runner,
                            config=ExecutionConfig(
                                asset_root=Path(item["asset_root"]),
                                concurrency=1,
                                warmup_count=self.settings.runner_warmup_count,
                                max_retries=self.settings.runner_max_retries,
                                retry_backoff_seconds=self.settings.runner_retry_backoff_seconds,
                            ),
                            should_cancel=lambda: self._cancelled(job_id),
                        )
                    if self._cancelled(job_id):
                        raise AutomationCancelled()
                    item["progress"] = asdict(progress)
                    if progress.succeeded_tasks != progress.total_tasks:
                        item["status"] = "failed"
                        self._update(job_id, progress={"runs": runs})
                        raise RuntimeError(
                            f"Run {current_run_id}: "
                            f"{progress.succeeded_tasks}/{progress.total_tasks} "
                            "tasks succeeded. Inspect its attempts before starting another job."
                        )
                    item["status"] = "completed"
                    self._update(job_id, progress={"runs": runs, "current_run_id": None})
                    current_run_id = None
            comparisons = []
            self._update(job_id, stage="Creating and freezing comparisons")
            # Commit all comparisons together, so failures cannot leave partially frozen jobs.
            with session_scope(self.factory) as session:
                for bundle in details["bundles"]:
                    for model_a, model_b in combinations(config.model_config_ids, 2):
                        if self._cancelled(job_id):
                            raise AutomationCancelled()
                        pair = [
                            next(
                                item
                                for item in runs
                                if item["snapshot_id"] == bundle["snapshot_id"]
                                and item["model_config_id"] == model_id
                            )
                            for model_id in (model_a, model_b)
                        ]
                        attempt_ids = []
                        for item in pair:
                            latest = {}
                            for attempt in session.scalars(
                                select(GenerationAttempt)
                                .where(
                                    GenerationAttempt.run_id == item["run_id"],
                                    GenerationAttempt.status == "succeeded",
                                )
                                .order_by(GenerationAttempt.attempt_number)
                            ):
                                latest[attempt.task_id] = attempt.id
                            attempt_ids.append(list(latest.values()))
                        result = create_comparison(
                            session,
                            snapshot_id=bundle["snapshot_id"],
                            name=(
                                f"{config.name}: {bundle['name']} — "
                                f"{pair[0]['model_name']} vs {pair[1]['model_name']}"
                            ),
                            assignment_seed=config.assignment_seed,
                            model_a_attempt_ids=attempt_ids[0],
                            model_b_attempt_ids=attempt_ids[1],
                        )
                        if result.issues:
                            raise RuntimeError("comparison has incomplete task pairs; not freezing")
                        comparison = freeze_comparison(session, result.session.id)
                        comparisons.append(
                            {
                                "id": comparison.id,
                                "name": comparison.name,
                                "status": comparison.status,
                                "item_count": len(comparison.items),
                            }
                        )
                job = session.get(AutomationJob, job_id)
                job.progress_json = {
                    **job.progress_json,
                    "comparisons": comparisons,
                    "current_run_id": None,
                    "stage": "Completed",
                }
                job.status = "completed"
                job.finished_at = utc_now()
        except AutomationCancelled:
            self._update(job_id, status="cancelled", stage="Stopped", finished_at=utc_now())
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                stage="Failed",
                error_message=str(exc),
                finished_at=utc_now(),
            )
        finally:
            try:
                with session_scope(self.factory) as session:
                    job = session.get(AutomationJob, job_id)
                    details = deepcopy(job.progress_json)
                    for item in details.get("runs", []):
                        run = session.get(ModelRun, item["run_id"])
                        if run.status == "running" or (
                            run.status == "created" and job.status == "cancelled"
                        ):
                            run.status = "cancelled" if job.status == "cancelled" else "failed"
                            run.finished_at = utc_now()
                        if item["status"] != "failed":
                            item["status"] = run.status
                    job.progress_json = details
            finally:
                lease.close()
