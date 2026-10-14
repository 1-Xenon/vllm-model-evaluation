"""FastAPI application for the local operator and reviewer workflows."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from .automation import AutomationBusy, AutomationManager, AutomationRequest
from .config import load_settings, prepare_storage
from .db import create_engine_from_settings, create_schema, create_session_factory, session_scope
from .errors import ValidationError
from .execution import ExecutionConfig
from .export import comparison_summary, export_csv, export_jsonl
from .logging_config import configure_logging
from .models import (
    AssignmentStatus,
    ComparisonItem,
    ComparisonSession,
    GenerationAttempt,
    MediaAsset,
    ModelConfig,
    ModelRun,
    ReviewerAssignment,
    ReviewerJudgment,
    Task,
    TaskInput,
    TaskSnapshot,
)
from .operator import (
    create_model_config,
    create_run,
    execute_existing_run,
    import_bundle,
    inspect_failures,
    list_snapshots,
    run_progress,
)
from .review import create_comparison, ensure_reviewer_assignments, reviewer_view, submit_judgment
from .runner_factory import create_runner


def _snapshot_bundle_name(snapshot: TaskSnapshot | None) -> str | None:
    bundle_root = (snapshot.notes_json or {}).get("bundle_root") if snapshot else None
    return Path(bundle_root).name if bundle_root else None


def _serialize_snapshot(snapshot: TaskSnapshot) -> dict[str, Any]:
    return {
        "id": snapshot.id,
        "source_name": snapshot.source_name,
        "bundle_name": _snapshot_bundle_name(snapshot),
        "snapshot_hash": snapshot.snapshot_hash,
        "manifest_hash": snapshot.manifest_hash,
        "task_count": snapshot.task_count,
        "created_at": snapshot.created_at.isoformat(),
    }


def _serialize_model_config(config: ModelConfig) -> dict[str, Any]:
    return {
        "id": config.id,
        "display_name": config.display_name,
        "endpoint_url": config.endpoint_url,
        "served_model_name": config.served_model_name,
        "checkpoint_identity": config.checkpoint_identity,
        "endpoint_reported_identity": config.endpoint_reported_identity,
    }


def _serialize_run(run: ModelRun) -> dict[str, Any]:
    bundle_name = _snapshot_bundle_name(run.snapshot) or "bundle-not-recorded"
    model_name = run.model_config.display_name
    return {
        "id": run.id,
        "snapshot_id": run.snapshot_id,
        "model_config_id": run.model_config_id,
        "run_label": run.run_label,
        "run_name": f"{bundle_name}_{model_name}",
        "status": run.status,
        "created_at": run.created_at.isoformat(),
    }


def _serialize_attempt(attempt: GenerationAttempt) -> dict[str, Any]:
    return {
        "id": attempt.id,
        "run_id": attempt.run_id,
        "task_id": attempt.task.stable_task_id,
        "attempt_number": attempt.attempt_number,
        "status": attempt.status,
        "final_answer": attempt.final_answer,
        "error_code": attempt.error_code,
        "error_message": attempt.error_message,
        "total_latency_seconds": attempt.total_latency_seconds,
        "output_token_count": attempt.output_token_count,
        "created_at": attempt.created_at.isoformat(),
    }


def _serialize_comparison(comparison: ComparisonSession) -> dict[str, Any]:
    return {
        "id": comparison.id,
        "snapshot_id": comparison.snapshot_id,
        "name": comparison.name,
        "status": comparison.status,
        "item_count": len(comparison.items),
        "created_at": comparison.created_at.isoformat(),
    }


def _bundle_manifest_path(bundle: Path) -> str | None:
    for filename in ("manifest.jsonl", "tasks.jsonl"):
        if (bundle / filename).is_file():
            return filename
    return None


def _bundle_root_for_snapshot(
    snapshot: TaskSnapshot | None,
    configured_root: Path,
    relative_path: str | None = None,
) -> Path:
    stored_root = (snapshot.notes_json or {}).get("bundle_root") if snapshot else None
    if stored_root:
        return Path(stored_root).resolve()

    root = configured_root.resolve()
    if _bundle_manifest_path(root):
        return root
    if not root.is_dir():
        return root

    children = [child for child in root.iterdir() if child.is_dir()]
    if relative_path:
        relative = Path(*PurePosixPath(relative_path).parts)
        matches = [child for child in children if (child / relative).is_file()]
    else:
        matches = [child for child in children if _bundle_manifest_path(child)]
    return matches[0] if len(matches) == 1 else root


def _serialize_assignment(assignment: ReviewerAssignment) -> dict[str, Any]:
    return {
        "assignment_id": assignment.id,
        "comparison_id": assignment.comparison_item.comparison_session_id,
        "reviewer_id": assignment.reviewer_id,
        "task_id": assignment.comparison_item.task.stable_task_id,
        "status": assignment.status,
    }


def _serialize_judgment(judgment: ReviewerJudgment) -> dict[str, Any]:
    return {
        "judgment_id": judgment.id,
        "assignment_id": judgment.assignment_id,
        "reviewer_id": judgment.assignment.reviewer_id,
        "task_id": judgment.assignment.comparison_item.task.stable_task_id,
        "version": judgment.version,
        "overall_preference": judgment.overall_preference,
        "left_acceptability": judgment.left_acceptability,
        "right_acceptability": judgment.right_acceptability,
        "disposition": judgment.disposition,
        "error_tags": judgment.error_tags_json,
        "comments": judgment.comments,
        "submitted_at": judgment.submitted_at.isoformat(),
    }


def create_app(config_path: str | Path | None = None) -> FastAPI:
    settings = load_settings(config_path)
    prepare_storage(settings)
    configure_logging(settings.log_level)
    engine = create_engine_from_settings(settings)
    create_schema(engine)
    factory = create_session_factory(engine)
    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.state.settings = settings
    app.state.session_factory = factory
    automation = AutomationManager(settings, factory)
    app.state.automation = automation

    @app.get("/api/automations")
    def automation_jobs() -> list[dict[str, Any]]:
        return automation.list_jobs()

    @app.post("/api/automations", status_code=202)
    def start_automation(payload: AutomationRequest) -> dict[str, Any]:
        try:
            return automation.start(payload)
        except AutomationBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, ValidationError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/automations/{job_id}")
    def automation_status(job_id: str) -> dict[str, Any]:
        try:
            return automation.get(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/automations/{job_id}/cancel")
    def cancel_automation(job_id: str) -> dict[str, Any]:
        try:
            return automation.cancel(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "runner_type": settings.runner_type}

    @app.get("/", response_class=HTMLResponse)
    def operator_home() -> str:
        return _OPERATOR_HTML

    @app.get("/review", response_class=HTMLResponse)
    def reviewer_home() -> str:
        return _REVIEW_HTML

    @app.get("/api/snapshots")
    def snapshots() -> list[dict[str, Any]]:
        with session_scope(factory) as session:
            return [_serialize_snapshot(snapshot) for snapshot in list_snapshots(session)]

    @app.get("/api/task-bundles")
    def task_bundles() -> list[dict[str, str]]:
        root = settings.task_bundle_root.resolve()
        if not root.is_dir():
            return []
        return [
            {"id": str(bundle), "name": bundle.name, "manifest_path": manifest_path}
            for bundle in sorted(root.iterdir(), key=lambda path: path.name.lower())
            if bundle.is_dir() and (manifest_path := _bundle_manifest_path(bundle))
        ]

    @app.get("/api/model-configs")
    def model_configs_list() -> list[dict[str, Any]]:
        with session_scope(factory) as session:
            configs = session.scalars(select(ModelConfig).order_by(ModelConfig.created_at)).all()
            return [_serialize_model_config(config) for config in configs]

    @app.post("/api/snapshots/import")
    def import_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            bundle_root = Path(payload["bundle_root"]).resolve()
            bundle_parent = settings.task_bundle_root.resolve()
            if bundle_root == bundle_parent:
                raise ValueError("bundle_root must select a folder under the task-bundle parent")
            try:
                bundle_root.relative_to(bundle_parent)
            except ValueError as exc:
                raise ValueError(
                    "bundle_root must be under the configured task-bundle parent"
                ) from exc
            with session_scope(factory) as session:
                snapshot = import_bundle(
                    session,
                    bundle_root,
                    payload.get("manifest_path")
                    or _bundle_manifest_path(bundle_root)
                    or "manifest.jsonl",
                )
                return _serialize_snapshot(snapshot)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/model-configs")
    def model_configs(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with session_scope(factory) as session:
                return _serialize_model_config(create_model_config(session, payload))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/runs")
    def runs(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with session_scope(factory) as session:
                return _serialize_run(create_run(session, payload))
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs")
    def runs_list() -> list[dict[str, Any]]:
        with session_scope(factory) as session:
            runs = session.scalars(select(ModelRun).order_by(ModelRun.created_at.desc())).all()
            return [_serialize_run(run) for run in runs]

    @app.post("/api/runs/{run_id}/execute")
    def execute(run_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        try:
            with automation.execution_slot(), session_scope(factory) as session:
                run = session.get(ModelRun, run_id)
                if run is None:
                    raise ValueError(f"model run not found: {run_id}")
                runner_settings = replace(
                    settings,
                    runner_endpoint_url=run.model_config.endpoint_url,
                    runner_model_name=run.model_config.served_model_name,
                )
                asset_root = payload.get("asset_root")
                if not asset_root:
                    snapshot = session.get(TaskSnapshot, run.snapshot_id) if run else None
                    image_path = None
                    if run:
                        image_path = session.scalar(
                            select(MediaAsset.relative_path)
                            .join(TaskInput, TaskInput.media_asset_id == MediaAsset.id)
                            .join(Task, TaskInput.task_id == Task.id)
                            .where(Task.snapshot_id == run.snapshot_id)
                            .where(TaskInput.input_type == "image")
                            .order_by(Task.task_order, TaskInput.ordinal)
                        )
                    asset_root = _bundle_root_for_snapshot(
                        snapshot, settings.task_bundle_root, image_path
                    )
                progress = execute_existing_run(
                    session,
                    run_id,
                    create_runner(runner_settings),
                    config=ExecutionConfig(
                        asset_root=Path(asset_root) if asset_root else settings.task_bundle_root,
                        concurrency=int(payload.get("concurrency", settings.runner_concurrency)),
                        warmup_count=int(payload.get("warmup_count", settings.runner_warmup_count)),
                        max_retries=int(payload.get("max_retries", settings.runner_max_retries)),
                        retry_backoff_seconds=float(
                            payload.get(
                                "retry_backoff_seconds", settings.runner_retry_backoff_seconds
                            )
                        ),
                        retry_failed=bool(payload.get("retry_failed", False)),
                    ),
                )
                return asdict(progress)
        except AutomationBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/progress")
    def progress(run_id: str) -> dict[str, Any]:
        try:
            with session_scope(factory) as session:
                return asdict(run_progress(session, run_id))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/failures")
    def failures(run_id: str) -> list[dict[str, Any]]:
        with session_scope(factory) as session:
            return [asdict(failure) for failure in inspect_failures(session, run_id)]

    @app.get("/api/runs/{run_id}/attempts")
    def attempts(run_id: str) -> list[dict[str, Any]]:
        with session_scope(factory) as session:
            rows = session.scalars(
                select(GenerationAttempt)
                .options(joinedload(GenerationAttempt.task))
                .join(Task, GenerationAttempt.task_id == Task.id)
                .where(GenerationAttempt.run_id == run_id)
                .order_by(Task.task_order, GenerationAttempt.attempt_number)
            ).all()
            return [_serialize_attempt(attempt) for attempt in rows]

    @app.post("/api/comparisons")
    def comparisons(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with session_scope(factory) as session:
                result = create_comparison(
                    session,
                    snapshot_id=payload["snapshot_id"],
                    name=payload["name"],
                    assignment_seed=payload["assignment_seed"],
                    model_a_attempt_ids=payload["model_a_attempt_ids"],
                    model_b_attempt_ids=payload["model_b_attempt_ids"],
                    created_by=payload.get("created_by"),
                )
                return {
                    "comparison_id": result.session.id,
                    "issues": [asdict(issue) for issue in result.issues],
                }
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/comparisons")
    def comparisons_list() -> list[dict[str, Any]]:
        with session_scope(factory) as session:
            comparisons = session.scalars(
                select(ComparisonSession).order_by(ComparisonSession.created_at.desc())
            ).all()
            return [_serialize_comparison(comparison) for comparison in comparisons]

    @app.get("/api/assignments")
    def assignments_list() -> list[dict[str, Any]]:
        with session_scope(factory) as session:
            assignments = session.scalars(
                select(ReviewerAssignment)
                .options(
                    joinedload(ReviewerAssignment.comparison_item).joinedload(ComparisonItem.task)
                )
                .order_by(ReviewerAssignment.assigned_at, ReviewerAssignment.id)
            ).all()
            return [_serialize_assignment(assignment) for assignment in assignments]

    @app.post("/api/comparisons/{comparison_id}/freeze")
    def freeze(comparison_id: str) -> dict[str, str]:
        from .review import freeze_comparison

        try:
            with session_scope(factory) as session:
                comparison = freeze_comparison(session, comparison_id)
                return {"comparison_id": comparison.id, "status": comparison.status}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/comparisons/{comparison_id}/assignments/{reviewer_id}")
    def assignments(comparison_id: str, reviewer_id: str) -> list[dict[str, Any]]:
        try:
            with session_scope(factory) as session:
                return [
                    {"assignment_id": assignment.id, "status": assignment.status}
                    for assignment in ensure_reviewer_assignments(
                        session, comparison_id, reviewer_id
                    )
                ]
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/comparisons/{comparison_id}/judgments")
    def judgments(comparison_id: str) -> list[dict[str, Any]]:
        with session_scope(factory) as session:
            rows = session.scalars(
                select(ReviewerJudgment)
                .join(ReviewerAssignment)
                .join(ComparisonItem)
                .options(
                    joinedload(ReviewerJudgment.assignment)
                    .joinedload(ReviewerAssignment.comparison_item)
                    .joinedload(ComparisonItem.task)
                )
                .where(ComparisonItem.comparison_session_id == comparison_id)
                .order_by(ReviewerJudgment.submitted_at, ReviewerJudgment.version)
            ).all()
            return [_serialize_judgment(judgment) for judgment in rows]

    @app.get("/api/review/assignments")
    def review_assignments_list() -> list[dict[str, Any]]:
        with session_scope(factory) as session:
            assignments = session.scalars(
                select(ReviewerAssignment)
                .options(
                    joinedload(ReviewerAssignment.comparison_item).joinedload(ComparisonItem.task)
                )
                .where(
                    ReviewerAssignment.status.in_(
                        [AssignmentStatus.ASSIGNED.value, AssignmentStatus.IN_PROGRESS.value]
                    )
                )
                .order_by(ReviewerAssignment.assigned_at, ReviewerAssignment.id)
            ).all()
            return [
                {
                    "assignment_id": assignment.id,
                    "status": assignment.status,
                    "task_id": assignment.comparison_item.task.stable_task_id,
                }
                for assignment in assignments
            ]

    @app.get("/api/review/assignments/{assignment_id}/judgments")
    def assignment_judgments(assignment_id: str) -> list[dict[str, Any]]:
        with session_scope(factory) as session:
            rows = session.scalars(
                select(ReviewerJudgment)
                .where(ReviewerJudgment.assignment_id == assignment_id)
                .options(
                    joinedload(ReviewerJudgment.assignment)
                    .joinedload(ReviewerAssignment.comparison_item)
                    .joinedload(ComparisonItem.task)
                )
                .order_by(ReviewerJudgment.submitted_at, ReviewerJudgment.version)
            ).all()
            return [_serialize_judgment(judgment) for judgment in rows]

    @app.get("/api/review/assignments/{assignment_id}")
    def review_assignment(assignment_id: str) -> dict[str, Any]:
        with session_scope(factory) as session:
            view = reviewer_view(session, assignment_id)
            for task_input in view["task"]["inputs"]:
                if task_input["type"] == "image":
                    task_input["url"] = (
                        f"/api/snapshots/{view['snapshot_id']}/media/{task_input['relative_path']}"
                    )
            return view

    @app.get("/api/snapshots/{snapshot_id}/media/{asset_path:path}")
    def snapshot_media(snapshot_id: str, asset_path: str) -> FileResponse:
        with session_scope(factory) as session:
            asset = session.scalar(
                select(MediaAsset).where(
                    MediaAsset.snapshot_id == snapshot_id,
                    MediaAsset.relative_path == asset_path,
                )
            )
            snapshot = session.get(TaskSnapshot, asset.snapshot_id) if asset else None
        if asset is None:
            raise HTTPException(status_code=404, detail="media asset not found")
        root = _bundle_root_for_snapshot(snapshot, settings.task_bundle_root, asset.relative_path)
        candidate = (root / Path(*PurePosixPath(asset.relative_path).parts)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="media asset not found") from exc
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="media asset file is unavailable")
        return FileResponse(candidate, media_type=asset.media_type)

    @app.post("/api/review/assignments/{assignment_id}/judgment")
    def judgment(assignment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with session_scope(factory) as session:
            saved = submit_judgment(session, assignment_id, payload)
            return {"judgment_id": saved.id, "version": saved.version}

    @app.get("/api/exports/{comparison_id}/summary")
    def summary(comparison_id: str) -> dict[str, Any]:
        with session_scope(factory) as session:
            return comparison_summary(session, comparison_id)

    @app.post("/api/exports/{comparison_id}")
    def export(comparison_id: str, payload: dict[str, Any]) -> dict[str, str]:
        path = Path(
            payload.get("path", settings.exports_root / f"comparison-{comparison_id}.jsonl")
        )
        kind = payload.get("format", "jsonl")
        with session_scope(factory) as session:
            output = (
                export_csv(session, path, comparison_id=comparison_id)
                if kind == "csv"
                else export_jsonl(
                    session,
                    path,
                    comparison_id=comparison_id,
                    include_model_mapping=bool(payload.get("include_model_mapping", False)),
                )
            )
        return {"path": str(output), "format": kind}

    return app


_OPERATOR_HTML = """<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>vLLM Evaluation Operator</title>
<style>
body { color: #202124; font-family: system-ui, sans-serif; line-height: 1.4;
  margin: 2rem auto; max-width: 90rem; padding: 0 1rem; }
section { border: 1px solid #bbb; margin: 1rem 0; padding: 1rem; }
fieldset { border: 1px solid #ddd; margin: 1rem 0; padding: 1rem; }
legend { font-weight: 600; padding: 0 .35rem; }
label { display: block; margin: .45rem 0; }
input, select, textarea { box-sizing: border-box; font: inherit; max-width: 100%; padding: .4rem; }
label > input:not([type='checkbox']), label > select, label > textarea {
  display: block; margin-top: .25rem; min-width: 20rem; width: 100%; }
select[multiple] { min-height: 9rem; }
textarea { min-height: 5rem; width: 100%; }
button { cursor: pointer; font: inherit; margin: .25rem .25rem .25rem 0; padding: .45rem .7rem; }
.grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(24rem, 1fr)); }
.button-row { display: flex; flex-wrap: wrap; gap: .5rem; }
.hint { color: #555; font-size: .95rem; }
.output { background: #f6f6f6; display: none; min-height: 4rem; overflow: auto;
  padding: 1rem; white-space: pre-wrap; }
.output.has-content { display: block; }
table { border-collapse: collapse; display: block; overflow-x: auto; width: 100%; }
th, td { border: 1px solid #bbb; padding: .45rem; text-align: left; vertical-align: top; }
th { background: #eee; }
.status { min-height: 1.5rem; }
</style>
</head>
<body>
<h1>vLLM Evaluation Operator</h1>
<p class='hint'>Use this page to prepare runs, comparisons, assignments, judgment review,
  and exports.
  Reviewer judgment submission remains available at <a href='/review'>/review</a>.</p>
<p><a href='/docs'>OpenAPI documentation</a></p>
<p id='status' class='status' role='status' aria-live='polite'></p>
<p id='runner-mode' class='hint'></p>

<section>
<h2>Automated evaluation</h2>
<p class='hint'>Choose one bundle and at least two models. Each model runs the selected bundle
  before the next model loads. Every model pair gets a frozen comparison.</p>
<form id='automation-form'>
  <div class='grid'>
    <label>Name<input id='automation-name' required value='Evaluation batch'></label>
    <label>Assignment seed<input id='automation-seed' required value='automation'></label>
    <label>Bundle<select id='automation-bundles' required></select></label>
    <label>Models<select id='automation-models' multiple required size='5'></select></label>
    <label>Model startup timeout (seconds)
      <input id='automation-timeout' type='number' min='10' max='7200' value='900' required>
    </label>
  </div>
  <label><input id='automation-swap' type='checkbox' checked>
    Automatically swap the local GPU model</label>
  <p class='hint'>Automatic swapping replaces the evaluation model container. Uncheck it to use
    models already running at their configured endpoints. Select multiple models with Ctrl/Cmd
    or Shift. Models run in the order shown. The job stops if any task fails; incomplete
    comparisons are not frozen.</p>
  <p id='automation-plan' class='hint'></p>
  <button id='automation-start' type='submit'>Start automation</button>
</form>
<label>Automation history<select id='automation-job'></select></label>
<button id='automation-refresh' type='button'>Refresh automation</button>
<button id='automation-stop' type='button' disabled>Stop automation</button>
<p class='hint'>Progress updates automatically. Stop takes effect after the current request
  or startup command finishes. Closing this page does not stop the job; keep the evaluation
  app running.</p>
<p id='automation-status' role='status' aria-live='polite'></p>
<div id='automation-results'></div>
<details><summary>Automation details</summary>
  <pre id='automation-output' class='output'></pre></details>
</section>

<section>
<h2>Task snapshots</h2>
<form id='snapshot-form'>
  <label>Task bundle<select id='bundle-root' required></select></label>
  <label>Manifest path
    <input id='manifest-path' value='manifest.jsonl'>
  </label>
  <div class='button-row'>
    <button type='submit'>Import snapshot</button>
    <button id='refresh-button' type='button'>Refresh records</button>
  </div>
</form>
<pre id='snapshots-output' class='output'></pre>
</section>

<section>
<h2>Model configurations</h2>
<p class='hint'>Add one reusable configuration for each model you want to evaluate. The endpoint
  is where vLLM receives requests; the served model name identifies the model at that endpoint.</p>
<form id='config-form'>
  <div class='grid'>
    <label>Model Name
      <input id='config-display' required placeholder='Qwen 2.5 VL 7B'>
    </label>
    <label>vLLM endpoint URL
      <input id='config-endpoint' required
        value='http://127.0.0.1:8001/v1/chat/completions'>
      <span class='hint'>Usually the same URL for models switched on one vLLM server.</span>
    </label>
    <label>Served Model Name
      <input id='config-model' required placeholder='Qwen/Qwen2.5-VL-7B-Instruct-AWQ'>
      <span class='hint'>Copy this from the vLLM <code>/v1/models</code> response.</span>
    </label>
  </div>
  <button type='submit'>Add model configuration</button>
</form>
<pre id='configs-output' class='output'></pre>
</section>

<section>
<h2>Runs</h2>
<form id='run-form'>
  <div class='grid'>
    <label>Snapshot<select id='run-snapshot' required></select></label>
    <label>Model configuration<select id='run-config' required></select></label>
</div>
  <button type='submit'>Create run</button>
</form>
<label>Run ID to execute or inspect<select id='run-id' required></select></label>
<button id='execute-run' type='button'>Execute run</button>
<button id='progress-run' type='button'>Refresh progress</button>
<button id='attempts-run' type='button'>Load attempts</button>
<pre id='run-output' class='output'></pre>
</section>

<section>
<h2>Comparisons</h2>
<p class='hint'>Select completed runs from the same snapshot. Successful attempts are loaded
  automatically. Select the corresponding successful task attempts in both lists. Use Ctrl/Cmd
  or Shift to select multiple items.</p>
<p id='comparison-runs-hint' class='hint'></p>
<form id='comparison-form'>
  <div class='grid'>
    <label>Snapshot<select id='comparison-snapshot' required></select></label>
    <label>Model A run<select id='comparison-run-a' required></select></label>
    <label>Model B run<select id='comparison-run-b' required></select></label>
    <label>Name<input id='comparison-name' required></label>
    <label>Assignment seed<input id='comparison-seed' required></label>
  </div>
  <label>Model A successful attempts
    <select id='attempts-a' multiple required></select>
  </label>
  <label>Model B successful attempts
    <select id='attempts-b' multiple required></select>
  </label>
  <button type='submit'>Create comparison</button>
</form>
<label>Comparison ID<input id='comparison-id' placeholder='comparison UUID'></label>
<button id='freeze-comparison' type='button'>Freeze comparison</button>
<pre id='comparison-output' class='output'></pre>
</section>

<section>
<h2>Current comparisons</h2>
<p class='hint'>All comparison records currently stored in the evaluation database.</p>
<button id='refresh-comparisons' type='button'>Refresh comparisons</button>
<div id='comparisons-list'></div>
</section>

<section>
<h2>Reviewer assignments</h2>
<p class='hint'>Freeze the comparison first, then choose it below. The button creates missing
  assignments or loads the existing assignments for that reviewer.</p>
<form id='assignment-form'>
  <label>Comparison<select id='assignment-comparison' required></select></label>
  <label>Reviewer ID<input id='assignment-reviewer' required placeholder='reviewer-1'></label>
  <button type='submit'>Create or load assignments</button>
</form>
<pre id='assignment-output' class='output'></pre>
</section>

<section>
<h2>Submitted judgments</h2>
<label>Assignment<select id='judgment-assignment' required></select></label>
<button id='load-judgments' type='button'>Load judgments</button>
<label>Comparison for summary<select id='judgment-comparison' required></select></label>
<button id='load-summary' type='button'>Load summary</button>
<div id='judgments-output'></div>
<pre id='summary-output' class='output'></pre>
</section>

<section>
<h2>Exports</h2>
<form id='export-form'>
  <label>Comparison ID<input id='export-comparison' required></label>
  <label>Format<select id='export-format'><option value='jsonl'>JSONL</option>
    <option value='csv'>CSV</option></select></label>
  <label>Path visible to the application
    <input id='export-path' required value='data/exports/comparison.jsonl'></label>
  <label><input id='include-mapping' type='checkbox'> Include model mapping in JSONL export</label>
  <button type='submit'>Create export</button>
</form>
<pre id='export-output' class='output'></pre>
</section>

<script>
const state = {snapshots: [], configs: [], runs: [], comparisons: [], assignments: [], bundles: []};
const $ = (id) => document.getElementById(id);

function showStatus(text) { $('status').textContent = text; }
function showJson(id, value) {
  const element = $(id);
  element.textContent = JSON.stringify(value, null, 2);
  element.classList.add('has-content');
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.text();
  let data = {};
  try { data = body ? JSON.parse(body) : {}; } catch (error) { /* non-JSON error */ }
  if (!response.ok) {
    throw new Error(data.detail || 'Request failed (' + response.status + ').');
  }
  return data;
}

function setOptions(id, records, label) {
  const select = $(id);
  const previous = select.value;
  const selected = new Set(Array.from(select.selectedOptions).map((option) => option.value));
  select.replaceChildren();
  records.forEach((record) => {
    const option = document.createElement('option');
    option.value = record.id ?? record.assignment_id;
    option.textContent = label(record);
    if (select.multiple) option.selected = selected.has(option.value);
    select.append(option);
  });
  if (!select.multiple && Array.from(select.options).some((option) => option.value === previous)) {
    select.value = previous;
  }
}

function selectBundleManifest() {
  const bundle = state.bundles.find((record) => record.id === $('bundle-root').value);
  $('manifest-path').value = bundle ? bundle.manifest_path : '';
}

function renderComparisons(records) {
  const root = $('comparisons-list'); root.replaceChildren();
  if (!records.length) { root.textContent = 'No comparisons found.'; return; }
  const table = document.createElement('table');
  const head = document.createElement('tr');
  ['Comparison ID', 'Name', 'Status', 'Items', 'Created'].forEach((heading) => {
    const cell = document.createElement('th'); cell.textContent = heading; head.append(cell);
  });
  table.append(head);
  records.forEach((record) => {
    const row = document.createElement('tr');
    const idCell = document.createElement('td');
    const idInput = document.createElement('input');
    idInput.value = record.id; idInput.readOnly = true; idInput.size = 36;
    const copy = document.createElement('button');
    copy.type = 'button'; copy.textContent = 'Copy';
    copy.addEventListener('click', async () => {
      await navigator.clipboard.writeText(record.id);
      showStatus('Comparison ID copied.');
    });
    idCell.append(idInput, copy); row.append(idCell);
    [record.name, record.status, record.item_count, record.created_at].forEach((value) => {
      const cell = document.createElement('td'); cell.textContent = String(value ?? ''); row.append(cell);
    });
    table.append(row);
  });
  root.append(table);
}

async function refreshRecords() {
  const health = await request('/health');
  $('runner-mode').textContent = health.runner_type === 'fake'
    ? 'Fake runner enabled: runs return simulated answers without calling a model.'
    : 'Live runner: runs call the endpoint in the selected model configuration.';
  state.snapshots = await request('/api/snapshots');
  state.configs = await request('/api/model-configs');
  state.runs = await request('/api/runs');
  state.comparisons = await request('/api/comparisons');
  renderComparisons(state.comparisons);
  state.assignments = await request('/api/assignments');
  state.bundles = await request('/api/task-bundles');
  setOptions('bundle-root', state.bundles, (record) => record.name);
  selectBundleManifest();
  setOptions('automation-bundles', state.bundles, (record) => record.name);
  setOptions('automation-models', state.configs, (record) => record.display_name);
  updateAutomationPlan();
  setOptions('run-snapshot', state.snapshots,
    (record) => record.bundle_name || 'bundle not recorded');
  setOptions('comparison-snapshot', [...state.snapshots].reverse(),
    (record) => record.bundle_name || 'bundle not recorded');
  setOptions('run-config', state.configs,
    (record) => record.display_name);
  const runLabel = (record) => record.run_name;
  setOptions('run-id', state.runs, runLabel);
  setOptions('assignment-comparison', state.comparisons.filter(
    (record) => record.status === 'frozen' || record.status === 'open'),
    (record) => record.name + ' [' + record.status + '] ' + record.id);
  setOptions('judgment-assignment', state.assignments,
    (record) => record.task_id + ' [' + record.status + '] ' + record.assignment_id);
  setOptions('judgment-comparison', state.comparisons,
    (record) => record.name + ' [' + record.status + '] ' + record.id);
  showJson('snapshots-output', state.snapshots);
  showJson('configs-output', state.configs);
  await refreshComparisonRuns();
}

async function refreshComparisonRuns() {
  const runs = state.runs.filter((run) =>
    run.snapshot_id === $('comparison-snapshot').value && run.status === 'completed');
  const label = (run) => run.run_name;
  setOptions('comparison-run-a', runs, label);
  setOptions('comparison-run-b', runs, label);
  if ($('comparison-run-a').value === $('comparison-run-b').value) {
    const a = runs.find((run) => run.id === $('comparison-run-a').value);
    const b = runs.find((run) => a && run.model_config_id !== a.model_config_id) ||
      runs.find((run) => run.id !== $('comparison-run-a').value);
    $('comparison-run-b').value = b ? b.id : '';
  }
  $('comparison-runs-hint').textContent = runs.length < 2
    ? 'Complete two runs on this snapshot before creating a comparison.'
    : 'Only completed runs for the selected snapshot are shown.';
  await Promise.all([
    loadAttempts($('comparison-run-a').value, 'attempts-a'),
    loadAttempts($('comparison-run-b').value, 'attempts-b')
  ]);
}

const attemptLoads = {};
async function loadAttempts(runId, targetId) {
  const select = $(targetId);
  const loadId = (attemptLoads[targetId] || 0) + 1;
  attemptLoads[targetId] = loadId;
  select.replaceChildren();
  select.dataset.runId = '';
  if (!runId) return;
  const attempts = await request('/api/runs/' + encodeURIComponent(runId) + '/attempts');
  if (attemptLoads[targetId] !== loadId) return;
  const latest = new Map();
  attempts.filter((attempt) => attempt.status === 'succeeded').forEach((attempt) => {
    const previous = latest.get(attempt.task_id);
    if (!previous || attempt.attempt_number > previous.attempt_number) {
      latest.set(attempt.task_id, attempt);
    }
  });
  latest.forEach((attempt) => {
    const option = document.createElement('option');
    option.value = attempt.id;
    option.textContent = attempt.task_id + ' — ' + attempt.id;
    option.selected = true;
    select.append(option);
  });
  select.dataset.runId = runId;
}

function idsFromSelect(id) {
  return Array.from($(id).selectedOptions).map((option) => option.value);
}

async function submitJsonForm(event, path, fields, outputId) {
  event.preventDefault();
  try {
    const payload = {};
    fields.forEach(([key, id]) => { payload[key] = $(id).value; });
    const result = await request(path, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    showJson(outputId, result);
    showStatus('Operation completed.');
    await refreshRecords();
  } catch (error) { showStatus(error.message); }
}

$('bundle-root').addEventListener('change', selectBundleManifest);
$('snapshot-form').addEventListener('submit', (event) => submitJsonForm(
  event, '/api/snapshots/import', [
    ['bundle_root', 'bundle-root'], ['manifest_path', 'manifest-path']],
  'snapshots-output'));

$('config-form').addEventListener('submit', (event) => submitJsonForm(
  event, '/api/model-configs', [
    ['display_name', 'config-display'], ['endpoint_url', 'config-endpoint'],
    ['served_model_name', 'config-model']
  ], 'configs-output'));

$('run-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    const result = await request('/api/runs', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        snapshot_id: $('run-snapshot').value, model_config_id: $('run-config').value
      })
    });
    showJson('run-output', result);
    showStatus('Run created.');
    await refreshRecords();
    $('run-id').value = result.id;
  } catch (error) { showStatus(error.message); }
});

$('execute-run').addEventListener('click', async () => {
  try {
    const result = await request(
      '/api/runs/' + encodeURIComponent($('run-id').value) + '/execute', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({})
    });
    showJson('run-output', result); showStatus('Run execution completed.');
    await refreshRecords();
  } catch (error) { showStatus(error.message); }
});

$('progress-run').addEventListener('click', async () => {
  try {
    showJson('run-output', await request(
      '/api/runs/' + encodeURIComponent($('run-id').value) + '/progress'));
  }
  catch (error) { showStatus(error.message); }
});

$('attempts-run').addEventListener('click', async () => {
  try {
    showJson('run-output', await request(
      '/api/runs/' + encodeURIComponent($('run-id').value) + '/attempts'));
  }
  catch (error) { showStatus(error.message); }
});

$('comparison-snapshot').addEventListener('change', () =>
  refreshComparisonRuns().catch((error) => showStatus(error.message)));
$('comparison-run-a').addEventListener('change', () => loadAttempts(
  $('comparison-run-a').value, 'attempts-a').catch((error) => showStatus(error.message)));
$('comparison-run-b').addEventListener('change', () => loadAttempts(
  $('comparison-run-b').value, 'attempts-b').catch((error) => showStatus(error.message)));

$('comparison-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    const runA = $('comparison-run-a').value;
    const runB = $('comparison-run-b').value;
    if (!runA || !runB || runA === runB) {
      throw new Error('Select two different completed runs for this snapshot.');
    }
    if ($('attempts-a').dataset.runId !== runA || $('attempts-b').dataset.runId !== runB) {
      throw new Error('Wait for the selected runs to finish loading their attempts.');
    }
    if (!idsFromSelect('attempts-a').length || !idsFromSelect('attempts-b').length) {
      throw new Error('Select successful attempts for both runs.');
    }
    const result = await request('/api/comparisons', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        snapshot_id: $('comparison-snapshot').value, name: $('comparison-name').value,
        assignment_seed: $('comparison-seed').value,
        model_a_attempt_ids: idsFromSelect('attempts-a'),
        model_b_attempt_ids: idsFromSelect('attempts-b')
      })
    });
    await refreshRecords();
    $('comparison-id').value = result.comparison_id;
    $('assignment-comparison').value = result.comparison_id;
    $('judgment-comparison').value = result.comparison_id;
    $('export-comparison').value = result.comparison_id;
    $('export-path').value = 'data/exports/comparison-' + result.comparison_id + '.jsonl';
    showJson('comparison-output', result);
    showStatus(result.issues.length
      ? 'Comparison created with excluded tasks. Review the issues before freezing.'
      : 'Comparison created.');
  } catch (error) { showStatus(error.message); }
});

$('freeze-comparison').addEventListener('click', async () => {
  try {
    const comparisonId = $('comparison-id').value;
    const id = encodeURIComponent(comparisonId);
    showJson('comparison-output', await request(
      '/api/comparisons/' + id + '/freeze', {method: 'POST'}));
    await refreshRecords();
    $('assignment-comparison').value = comparisonId;
    showStatus('Comparison frozen.');
  } catch (error) { showStatus(error.message); }
});

$('assignment-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    const path = '/api/comparisons/' + encodeURIComponent($('assignment-comparison').value);
    const result = await request(
      path + '/assignments/' + encodeURIComponent($('assignment-reviewer').value));
    showJson('assignment-output', result); showStatus('Assignments loaded.');
    await refreshRecords();
  } catch (error) { showStatus(error.message); }
});

function renderJudgments(records) {
  const root = $('judgments-output'); root.replaceChildren();
  if (!records.length) { root.textContent = 'No submitted judgments.'; return; }
  const table = document.createElement('table');
  const headings = ['Assignment', 'Task', 'Reviewer', 'Version', 'Preference', 'Left', 'Right',
    'Disposition', 'Error tags', 'Comments', 'Submitted'];
  const head = document.createElement('tr');
  headings.forEach((heading) => { const cell = document.createElement('th');
    cell.textContent = heading; head.append(cell); });
  table.append(head);
  records.forEach((record) => {
    const row = document.createElement('tr');
    [record.assignment_id, record.task_id, record.reviewer_id, record.version,
      record.overall_preference,
      record.left_acceptability, record.right_acceptability, record.disposition,
      (record.error_tags || []).join(', '), record.comments || '', record.submitted_at]
      .forEach((value) => { const cell = document.createElement('td');
        cell.textContent = String(value ?? ''); row.append(cell); });
    table.append(row);
  });
  root.append(table);
}

$('load-judgments').addEventListener('click', async () => {
  try {
    const id = encodeURIComponent($('judgment-assignment').value);
    renderJudgments(await request('/api/review/assignments/' + id + '/judgments'));
    showStatus('Judgments loaded.');
  } catch (error) { showStatus(error.message); }
});

$('load-summary').addEventListener('click', async () => {
  try {
    const id = encodeURIComponent($('judgment-comparison').value);
    showJson('summary-output', await request('/api/exports/' + id + '/summary'));
  } catch (error) { showStatus(error.message); }
});

$('refresh-comparisons').addEventListener('click', () =>
  refreshRecords().then(() => showStatus('Comparisons refreshed.'))
    .catch((error) => showStatus(error.message)));

$('export-format').addEventListener('change', () => {
  const path = $('export-path').value;
  if (path.startsWith('data/exports/comparison-')) {
    $('export-path').value = path.replace(/\\.(jsonl|csv)$/, '') +
      '.' + $('export-format').value;
  }
});

$('export-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    const format = $('export-format').value;
    const result = await request(
      '/api/exports/' + encodeURIComponent($('export-comparison').value), {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        format, path: $('export-path').value,
        include_model_mapping: $('include-mapping').checked
      })
    });
    showJson('export-output', result); showStatus('Export created.');
  } catch (error) { showStatus(error.message); }
});


let automationTimer;
let automationRevision = 0;
let displayedAutomationStatus = '';

function updateAutomationPlan() {
  const bundles = idsFromSelect('automation-bundles').length;
  const models = idsFromSelect('automation-models').length;
  $('automation-plan').textContent = bundles + ' bundle, ' + models + ' models: ' +
    (bundles * models) + ' runs and ' + (bundles * models * (models - 1) / 2) + ' comparisons.';
}

function renderAutomation(job) {
  const active = ['queued', 'running', 'cancelling'].includes(job.status);
  $('automation-stop').disabled = !active || job.cancel_requested;
  const progress = job.current_run_progress;
  const count = progress
    ? ' — ' + progress.completed_tasks + '/' + progress.total_tasks + ' tasks' : '';
  $('automation-status').textContent = job.name + ': ' + job.status + ' — ' +
    job.progress.stage + count + (job.error_message ? ' — ' + job.error_message : '');
  showJson('automation-output', job);
  const results = $('automation-results');
  results.replaceChildren();
  const completed = job.progress.runs.filter((run) => run.status === 'completed').length;
  const summary = document.createElement('p');
  summary.textContent = completed + '/' + job.progress.runs.length + ' runs completed; ' +
    job.progress.comparisons.length + ' comparisons frozen.';
  results.append(summary);
  job.progress.comparisons.forEach((comparison) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = comparison.name + ' (' + comparison.status + ')';
    button.addEventListener('click', async () => {
      try {
        await refreshRecords();
        $('comparison-id').value = comparison.id;
        $('assignment-comparison').value = comparison.id;
        $('judgment-comparison').value = comparison.id;
        $('export-comparison').value = comparison.id;
        $('assignment-form').scrollIntoView({behavior: 'smooth'});
      } catch (error) { showStatus(error.message); }
    });
    results.append(button);
  });
}

async function refreshAutomation(selectedId) {
  clearTimeout(automationTimer);
  const revision = ++automationRevision;
  try {
    const jobs = await request('/api/automations');
    if (revision !== automationRevision) return;
    setOptions('automation-job', jobs, (job) =>
      job.name + ' — ' + job.status + ' — ' + job.created_at);
    if (selectedId && jobs.some((job) => job.id === selectedId)) {
      $('automation-job').value = selectedId;
    }
    $('automation-start').disabled = jobs.some((job) =>
      ['queued', 'running', 'cancelling'].includes(job.status));
    const id = $('automation-job').value;
    if (!id) {
      $('automation-status').textContent = 'No automation jobs yet.';
      $('automation-stop').disabled = true;
      return;
    }
    const job = await request('/api/automations/' + encodeURIComponent(id));
    if (revision !== automationRevision) return;
    renderAutomation(job);
    const stateKey = job.id + ':' + job.status;
    if (displayedAutomationStatus !== stateKey &&
        !['queued', 'running', 'cancelling'].includes(job.status)) {
      await refreshRecords();
    }
    displayedAutomationStatus = stateKey;
  } catch (error) { $('automation-status').textContent = error.message; }
  finally {
    if (revision === automationRevision) {
      automationTimer = setTimeout(() => refreshAutomation(), 3000);
    }
  }
}

$('automation-bundles').addEventListener('change', updateAutomationPlan);
$('automation-models').addEventListener('change', updateAutomationPlan);
$('automation-job').addEventListener('change', () => refreshAutomation($('automation-job').value));
$('automation-refresh').addEventListener('click', () => refreshAutomation());
$('automation-stop').addEventListener('click', async () => {
  try {
    const id = $('automation-job').value;
    await request('/api/automations/' + encodeURIComponent(id) + '/cancel', {method: 'POST'});
    await refreshAutomation(id);
  } catch (error) { showStatus(error.message); }
});
$('automation-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const bundles = idsFromSelect('automation-bundles');
  const models = idsFromSelect('automation-models');
  if (bundles.length !== 1 || models.length < 2) {
    showStatus('Select one bundle and at least two models.');
    return;
  }
  $('automation-start').disabled = true;
  try {
    const job = await request('/api/automations', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: $('automation-name').value, bundles,
        model_config_ids: models, swap_models: $('automation-swap').checked,
        ready_timeout_seconds: Number($('automation-timeout').value),
        assignment_seed: $('automation-seed').value})
    });
    showStatus('Automation started. You can follow its progress below.');
    await refreshAutomation(job.id);
    await refreshRecords();
  } catch (error) {
    showStatus(error.message);
    $('automation-start').disabled = false;
  }
});

$('refresh-button').addEventListener('click', () => refreshRecords().catch(
  (error) => showStatus(error.message)));
refreshRecords().then(() => refreshAutomation()).catch((error) => showStatus(error.message));
</script>
</body>
</html>"""
_REVIEW_HTML = """<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Blind Review</title>
<style>
body { font-family: system-ui, sans-serif; line-height: 1.45; margin: 2rem auto;
  max-width: 72rem; padding: 0 1rem; color: #202124; }
label { display: block; margin: .35rem 0; }
input[type='text'], textarea, select { box-sizing: border-box; font: inherit;
  max-width: 100%; padding: .45rem; }
textarea { min-height: 7rem; width: 100%; }
button { cursor: pointer; font: inherit; padding: .5rem .85rem; }
fieldset { border: 1px solid #bbb; margin: 1rem 0; padding: 1rem; }
legend { font-weight: 600; padding: 0 .35rem; }
.assignment-bar { align-items: end; display: flex; flex-wrap: wrap; gap: .5rem; }
.assignment-bar label { flex: 1 1 24rem; margin: 0; }
.assignment-bar input { width: 100%; }
.message { min-height: 1.5rem; margin: 1rem 0; }
.error { color: #a40000; }
.success { color: #176b2c; }
.inputs { background: #f6f6f6; padding: 1rem; }
.input-item { margin: .75rem 0; }
.input-item img { display: block; height: auto; max-height: 28rem; max-width: 100%; }
.prompt, .answer { white-space: pre-wrap; word-break: break-word; }
.answers { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(20rem, 1fr)); }
.answer-panel { border: 1px solid #bbb; padding: 1rem; }
.answer-panel h3 { margin-top: 0; }
.answer-panel { min-width: 0; }
.annotation-controls { display: flex; flex-wrap: wrap; gap: .75rem; align-items: center; }
.annotation-controls label { font-size: .9rem; }
.annotation-note { font-size: .9rem; color: #444; }
.annotated-figure { margin: 1rem 0; }
.annotated-image { position: relative; width: 100%; }
.annotated-image img { display: block; width: 100%; height: auto; }
.box-layer { position: absolute; inset: 0; overflow: hidden; pointer-events: none; }
.bbox-mark { position: absolute; box-sizing: border-box; border: 2px solid; }
.bbox-label { position: absolute; top: 0; left: 0; color: white; padding: 1px 4px;
  font-size: 12px; line-height: 1.4; white-space: nowrap; max-width: 15rem;
  overflow: hidden; text-overflow: ellipsis; }
.box-legend { padding-left: 1.4rem; font-size: .9rem; }
.box-legend li { margin: .25rem 0; }

.choices { display: flex; flex-wrap: wrap; gap: 1rem; }
.choices label { display: inline-block; }
.hidden { display: none; }
</style>
</head>
<body>
<h1>Blind Review</h1>
<p>Load an assignment to inspect the anonymous task and submit your judgment.</p>
<div class='assignment-bar'>
  <label for='assignment'>Assignment ID
    <select id='assignment' required></select>
  </label>
  <button id='load' type='button'>Load assignment</button>
</div>
<p id='message' class='message' role='status' aria-live='polite'></p>
<main id='review' class='hidden'>
  <section aria-labelledby='task-heading'>
    <h2 id='task-heading'>Task</h2>
    <p><strong>Category:</strong> <span id='category'></span></p>
    <h3>Prompt</h3>
    <div id='prompt' class='prompt'></div>
    <div id='inputs' class='inputs'></div>
  </section>
  <section aria-labelledby='answers-heading'>
    <h2 id='answers-heading'>Anonymous answers</h2>
    <div class='answers'>
      <article class='answer-panel'><h3>Left</h3>
        <div id='left-images'></div>
        <div id='left-answer' class='answer'></div></article>
      <article class='answer-panel'><h3>Right</h3>
        <div id='right-images'></div>
        <div id='right-answer' class='answer'></div></article>
    </div>
  </section>
  <form id='judgment-form'>
    <fieldset>
      <legend>Overall preference</legend>
      <div class='choices'>
        <label><input type='radio' name='overall_preference' value='left'> Left</label>
        <label><input type='radio' name='overall_preference' value='right'> Right</label>
        <label><input type='radio' name='overall_preference' value='tie'> Tie</label>
      </div>
    </fieldset>
    <fieldset>
      <legend>Acceptability</legend>
      <label for='left-acceptability'>Left answer
        <select id='left-acceptability' name='left_acceptability'>
          <option value=''>Select an assessment</option>
          <option value='acceptable'>Acceptable</option>
          <option value='needs_revision'>Needs revision</option>
          <option value='unacceptable'>Unacceptable</option>
        </select>
      </label>
      <label for='right-acceptability'>Right answer
        <select id='right-acceptability' name='right_acceptability'>
          <option value=''>Select an assessment</option>
          <option value='acceptable'>Acceptable</option>
          <option value='needs_revision'>Needs revision</option>
          <option value='unacceptable'>Unacceptable</option>
        </select>
      </label>
    </fieldset>
    <fieldset>
      <legend>Error tags (optional)</legend>
      <div class='choices'>
        <label><input type='checkbox' name='error_tags' value='factual_error'> Factual error</label>
        <label><input type='checkbox' name='error_tags' value='instruction_following'>
          Instruction following</label>
        <label><input type='checkbox' name='error_tags' value='relevance'> Relevance</label>
        <label><input type='checkbox' name='error_tags' value='verbosity'> Verbosity</label>
        <label><input type='checkbox' name='error_tags' value='clarity'> Clarity</label>
        <label><input type='checkbox' name='error_tags' value='formatting'> Formatting</label>
        <label><input type='checkbox' name='error_tags' value='safety'> Safety</label>
        <label><input type='checkbox' name='error_tags' value='other'> Other</label>
      </div>
    </fieldset>
    <fieldset>
      <legend>Disposition</legend>
      <div class='choices'>
        <label><input type='radio' name='disposition' value='complete' checked>
          Complete review</label>
        <label><input type='radio' name='disposition' value='skip'> Skip this task</label>
      </div>
    </fieldset>
    <label for='comments'>Comments (optional)</label>
    <textarea id='comments' name='comments'
      placeholder='Record observations for the evaluation team'></textarea>
    <p><button type='submit'>Submit judgment</button></p>
  </form>
</main>
<script>
let currentAssignmentId = null;

const message = document.getElementById('message');
const review = document.getElementById('review');
const judgmentForm = document.getElementById('judgment-form');

function showMessage(text, kind = '') {
  message.textContent = text;
  message.className = 'message ' + kind;
}

function setRequiredFields(required) {
  document.querySelectorAll(
    '[name="overall_preference"], #left-acceptability, #right-acceptability')
    .forEach((element) => { element.required = required; });
}

function renderInputs(inputs) {
  const root = document.getElementById('inputs');
  root.replaceChildren();
  if (!inputs.length) return;
  const heading = document.createElement('h3');
  heading.textContent = 'Inputs';
  root.append(heading);
  inputs.forEach((input, index) => {
    const item = document.createElement('div');
    item.className = 'input-item';
    if (input.type === 'image' && input.url) {
      const image = document.createElement('img');
      image.src = input.url;
      image.alt = 'Task image ' + (index + 1);
      item.append(image);
    } else if (input.type === 'text') {
      item.textContent = input.text || '';
      item.className += ' prompt';
    }
    root.append(item);
  });
}

function annotationColor(label) {
  const colors = ['#b91c1c', '#1d4ed8', '#047857', '#7e22ce', '#9a3412', '#0e7490'];
  let hash = 0;
  for (const char of label.toLowerCase()) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return colors[hash % colors.length];
}

function boxPercentages(box, width, height) {
  const c = box.coordinates;
  if (!Array.isArray(c) || c.length !== 4 || !c.every(Number.isFinite)) return null;
  let unit = box.unit;
  if (!unit || unit === 'auto') unit = c.every((n) => n >= 0 && n <= 1)
    ? 'normalized' : 'pixels';
  const scale = {normalized: 1, '1000': 1000, percent: 100}[unit];
  const xScale = scale || width;
  const yScale = scale || height;
  if (!(xScale > 0 && yScale > 0)) return null;
  const [x1, y1, x2, y2] = [c[0] / xScale, c[1] / yScale, c[2] / xScale, c[3] / yScale];
  if (x1 < 0 || y1 < 0 || x2 > 1 || y2 > 1 || x2 <= x1 || y2 <= y1) return null;
  return {left: x1 * 100, top: y1 * 100, width: (x2 - x1) * 100,
    height: (y2 - y1) * 100};
}

function annotationImageIndex(box, images, unassigned) {
  if (!box.image) return images.length === 1 ? 0 : Number(unassigned);
  if (Number.isInteger(box.image.index)) return box.image.index;
  if (typeof box.image.path === 'string') {
    const exact = images.findIndex((image) => image.relative_path === box.image.path);
    if (exact >= 0) return exact;
    const matches = images.map((image, index) => ({image, index})).filter(({image}) =>
      image.relative_path.split('/').pop() === box.image.path);
    return matches.length === 1 ? matches[0].index : -1;
  }
  return -1;
}

function renderAnnotatedImages(side, inputs, annotations) {
  const root = document.getElementById(side + '-images');
  root.replaceChildren();
  const images = inputs.filter((input) => input.type === 'image' && input.url);
  if (!images.length) return;
  const boxes = annotations?.boxes || [];
  const controls = document.createElement('div');
  controls.className = 'annotation-controls';
  const toggleLabel = document.createElement('label');
  const toggle = document.createElement('input');
  toggle.type = 'checkbox'; toggle.checked = true;
  toggleLabel.append(toggle, ' Show boxes');
  controls.append(toggleLabel);
  const source = document.createElement('select');
  source.value = '-1';
  if (images.length > 1) {
    const label = document.createElement('label');
    label.textContent = 'Boxes without an image reference ';
    const option = document.createElement('option');
    option.value = '-1'; option.textContent = 'Choose an image'; source.append(option);
    images.forEach((input, index) => {
      const option = document.createElement('option');
      option.value = String(index); option.textContent = 'Image ' + (index + 1);
      source.append(option);
    });
    label.append(source); controls.append(label);
  }
  const note = document.createElement('p');
  note.className = 'annotation-note';
  root.append(controls, note);
  const figures = images.map((input, index) => {
    const figure = document.createElement('figure');
    figure.className = 'annotated-figure';
    const caption = document.createElement('figcaption');
    caption.textContent = side === 'left' ? 'Left — image ' : 'Right — image ';
    caption.textContent += index + 1;
    const wrapper = document.createElement('div');
    wrapper.className = 'annotated-image';
    const image = document.createElement('img');
    image.alt = (side === 'left' ? 'Left' : 'Right') + ' answer annotations, image ' + (index + 1);
    const layer = document.createElement('div');
    layer.className = 'box-layer'; layer.setAttribute('aria-hidden', 'true');
    wrapper.append(image, layer);
    const legend = document.createElement('ol');
    legend.className = 'box-legend';
    figure.append(caption, wrapper, legend); root.append(figure);
    return {image, layer, legend, input};
  });
  function redraw() {
    let shown = 0, skipped = 0, unassigned = 0;
    figures.forEach(({layer, legend}) => {
      layer.replaceChildren(); legend.replaceChildren();
      layer.hidden = !toggle.checked;
    });
    boxes.forEach((box, index) => {
      const imageIndex = annotationImageIndex(box, images, source.value);
      const figure = figures[imageIndex];
      if (!figure) { unassigned++; return; }
      const {image, layer, legend} = figure;
      if (!image.naturalWidth) return;
      const coords = boxPercentages(box, image.naturalWidth, image.naturalHeight);
      if (!coords) { skipped++; return; }
      const label = box.label || 'Subject';
      const color = annotationColor(label);
      const mark = document.createElement('div');
      mark.className = 'bbox-mark'; mark.style.borderColor = color;
      Object.entries(coords).forEach(([key, value]) => { mark.style[key] = value + '%'; });
      const tag = document.createElement('span');
      tag.className = 'bbox-label'; tag.style.backgroundColor = color;
      tag.textContent = (index + 1) + ': ' + label;
      mark.append(tag); layer.append(mark);
      const entry = document.createElement('li');
      entry.value = index + 1;
      entry.textContent = label + ' [' + box.coordinates.join(', ') + ']';
      entry.style.color = color;
      legend.append(entry); shown++;
    });
    const messages = [boxes.length ? shown + '/' + boxes.length + ' boxes drawn.'
      : 'No usable bounding boxes found in this answer.'];
    if (skipped) messages.push(skipped + ' outside image bounds at the automatic scale; skipped.');
    if (unassigned) messages.push(unassigned + ' without a matching image; choose its source.');
    if (figures.some(({image}) => image.dataset.failed)) {
      messages.push('An image could not be loaded.');
    }
    else if (figures.some(({image}) => !image.naturalWidth)) messages.push('Loading images…');
    messages.push(...(annotations?.warnings || []));
    note.textContent = messages.join(' ');
  }
  figures.forEach(({image, input}) => {
    image.addEventListener('load', redraw);
    image.addEventListener('error', () => { image.dataset.failed = 'true'; redraw(); });
    image.src = input.url;
  });
  toggle.addEventListener('change', redraw);
  source.addEventListener('change', redraw);
  redraw();
}

function renderReview(data) {
  currentAssignmentId = data.assignment_id;
  document.getElementById('category').textContent = data.task.category || '';
  document.getElementById('prompt').textContent = data.task.prompt || '';
  document.getElementById('left-answer').textContent = data.left.answer || '';
  document.getElementById('right-answer').textContent = data.right.answer || '';
  renderInputs(data.task.inputs || []);
  renderAnnotatedImages('left', data.task.inputs || [], data.left.annotations);
  renderAnnotatedImages('right', data.task.inputs || [], data.right.annotations);
  review.classList.remove('hidden');
  judgmentForm.reset();
  document.querySelector('input[name="disposition"][value="complete"]').checked = true;
  setRequiredFields(true);
  showMessage(data.status === 'submitted'
    ? 'This assignment has already been submitted. A new submission will be recorded as a revision.'
    : 'Review loaded. Model identities are hidden.', 'success');
}

async function loadAssignmentOptions() {
  try {
    const response = await fetch('/api/review/assignments');
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Unable to load assignments.');
    const select = document.getElementById('assignment');
    select.replaceChildren();
    data.forEach((assignment) => {
      const option = document.createElement('option');
      option.value = assignment.assignment_id;
      option.textContent = assignment.task_id + ' — ' + assignment.status + ' — '
        + assignment.assignment_id;
      select.append(option);
    });
    if (!data.length) {
      const option = document.createElement('option');
      option.textContent = 'No assignments available';
      option.disabled = true;
      option.selected = true;
      select.append(option);
    }
  } catch (error) {
    showMessage(error.message, 'error');
  }
}

async function loadReview() {
  const id = document.getElementById('assignment').value;
  if (!id) {
    showMessage('Select an assignment ID.', 'error');
    return;
  }
  showMessage('Loading assignment...');
  try {
    const response = await fetch('/api/review/assignments/' + encodeURIComponent(id));
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Unable to load assignment.');
    renderReview(data);
  } catch (error) {
    review.classList.add('hidden');
    showMessage(error.message, 'error');
  }
}

async function submitJudgment(event) {
  event.preventDefault();
  if (!currentAssignmentId) return;
  const formData = new FormData(judgmentForm);
  const disposition = formData.get('disposition');
  const payload = {
    overall_preference: formData.get('overall_preference') || null,
    left_acceptability: formData.get('left_acceptability') || null,
    right_acceptability: formData.get('right_acceptability') || null,
    disposition,
    error_tags: formData.getAll('error_tags'),
    comments: formData.get('comments').trim() || null
  };
  if (disposition === 'complete' && (!payload.overall_preference ||
      !payload.left_acceptability || !payload.right_acceptability)) {
    showMessage('Select a preference and an acceptability assessment for both answers.', 'error');
    return;
  }
  showMessage('Submitting judgment...');
  try {
    const response = await fetch(
      '/api/review/assignments/' + encodeURIComponent(currentAssignmentId) + '/judgment', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Unable to submit judgment.');
    showMessage('Judgment submitted successfully (version ' + data.version + ').', 'success');
  } catch (error) {
    showMessage(error.message, 'error');
  }
}

document.getElementById('load').addEventListener('click', loadReview);
loadAssignmentOptions();
document.querySelectorAll('[name="disposition"]').forEach((element) => {
  element.addEventListener('change', () => setRequiredFields(element.value === 'complete'));
});
judgmentForm.addEventListener('submit', submitJudgment);
</script></body></html>"""
