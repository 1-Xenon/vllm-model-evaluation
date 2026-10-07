"""FastAPI application for the local operator and reviewer workflows."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import joinedload

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


def _serialize_snapshot(snapshot: TaskSnapshot) -> dict[str, Any]:
    return {
        "id": snapshot.id,
        "source_name": snapshot.source_name,
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
    return {
        "id": run.id,
        "snapshot_id": run.snapshot_id,
        "model_config_id": run.model_config_id,
        "run_label": run.run_label,
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

    @app.get("/api/model-configs")
    def model_configs_list() -> list[dict[str, Any]]:
        with session_scope(factory) as session:
            configs = session.scalars(select(ModelConfig).order_by(ModelConfig.created_at)).all()
            return [_serialize_model_config(config) for config in configs]

    @app.post("/api/snapshots/import")
    def import_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with session_scope(factory) as session:
                snapshot = import_bundle(
                    session, payload["bundle_root"], payload.get("manifest_path", "manifest.jsonl")
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
            with session_scope(factory) as session:
                asset_root = payload.get("asset_root")
                progress = execute_existing_run(
                    session,
                    run_id,
                    create_runner(settings),
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
        if asset is None:
            raise HTTPException(status_code=404, detail="media asset not found")
        root = settings.task_bundle_root.resolve()
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

<section>
<h2>Task snapshots</h2>
<form id='snapshot-form'>
  <label>Bundle root visible to the application
    <input id='bundle-root' required placeholder='/data/task-bundles/example'>
  </label>
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
    <input id='export-path' required value='/data/exports/comparison.jsonl'></label>
  <label><input id='include-mapping' type='checkbox'> Include model mapping in JSONL export</label>
  <button type='submit'>Create export</button>
</form>
<pre id='export-output' class='output'></pre>
</section>

<script>
const state = {snapshots: [], configs: [], runs: [], comparisons: [], assignments: []};
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
  select.replaceChildren();
  records.forEach((record) => {
    const option = document.createElement('option');
    option.value = record.id ?? record.assignment_id;
    option.textContent = label(record);
    select.append(option);
  });
}

async function refreshRecords() {
  state.snapshots = await request('/api/snapshots');
  state.configs = await request('/api/model-configs');
  state.runs = await request('/api/runs');
  state.comparisons = await request('/api/comparisons');
  state.assignments = await request('/api/assignments');
  setOptions('run-snapshot', state.snapshots,
    (record) => record.id + ' (' + record.task_count + ' tasks)');
  setOptions('comparison-snapshot', state.snapshots,
    (record) => record.id + ' (' + record.task_count + ' tasks)');
  setOptions('run-config', state.configs,
    (record) => record.display_name + ' [' + record.id + ']');
  const runLabel = (record) => record.run_label + ' [' + record.id + ']';
  setOptions('comparison-run-a', state.runs, runLabel);
  setOptions('comparison-run-b', state.runs, runLabel);
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
}

async function loadAttempts(runId, targetId) {
  if (!runId) return;
  const attempts = await request('/api/runs/' + encodeURIComponent(runId) + '/attempts');
  const select = $(targetId);
  select.replaceChildren();
  attempts.filter((attempt) => attempt.status === 'succeeded').forEach((attempt) => {
    const option = document.createElement('option');
    option.value = attempt.id;
    option.textContent = attempt.task_id + ' — ' + attempt.id;
    option.selected = true;
    select.append(option);
  });
  showJson('comparison-output', attempts);
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

$('comparison-run-a').addEventListener('change', () => loadAttempts(
  $('comparison-run-a').value, 'attempts-a'));
$('comparison-run-b').addEventListener('change', () => loadAttempts(
  $('comparison-run-b').value, 'attempts-b'));

$('comparison-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
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
    $('export-path').value = '/data/exports/comparison-' + result.comparison_id + '.jsonl';
    showJson('comparison-output', result); showStatus('Comparison created.');
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

$('export-format').addEventListener('change', () => {
  const path = $('export-path').value;
  if (path.startsWith('/data/exports/comparison-')) {
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

$('refresh-button').addEventListener('click', () => refreshRecords().catch(
  (error) => showStatus(error.message)));
refreshRecords().catch((error) => showStatus(error.message));
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
        <div id='left-answer' class='answer'></div></article>
      <article class='answer-panel'><h3>Right</h3>
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

function renderReview(data) {
  currentAssignmentId = data.assignment_id;
  document.getElementById('category').textContent = data.task.category || '';
  document.getElementById('prompt').textContent = data.task.prompt || '';
  document.getElementById('left-answer').textContent = data.left.answer || '';
  document.getElementById('right-answer').textContent = data.right.answer || '';
  renderInputs(data.task.inputs || []);
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
