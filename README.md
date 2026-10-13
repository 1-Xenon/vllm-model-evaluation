# vLLM Model Evaluation

This project provides a traceable, offline-capable workflow for comparing two locally hosted, stateless vision-language models. The design specification and implementation decisions are documented in:

- [`vllm_evaluation_design_spec.md`](vllm_evaluation_design_spec.md)
- [`initial_release_decisions.md`](initial_release_decisions.md)
- [`initial_release_task_breakdown.md`](initial_release_task_breakdown.md)
- [`task_manifest_schema.md`](task_manifest_schema.md)

## Phase 0 foundation

The initial implementation foundation uses:

- Python 3.12 or newer.
- FastAPI and Uvicorn for the application layer.
- SQLite with SQLAlchemy for the initial persistence layer.
- TOML configuration files with environment-variable overrides.
- Pytest for tests.
- Ruff for formatting and linting.

The current bootstrap command validates configuration and creates configured local storage directories. The local API and operator/reviewer pages are available through `vllm_evaluation.app:create_app`.

The Phase 1 persistence layer uses SQLite through SQLAlchemy. Schema changes are tracked with Alembic and should be applied with:

```bash
alembic upgrade head
```

## Local setup

Create a virtual environment and install the project with development dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

Copy `config.example.toml` to a local configuration file and adjust paths or runtime settings as needed:

```bash
cp config.example.toml config.toml
vllm-evaluation --config config.toml
```

The application does not package model weights. Sequential Qwen2.5-VL model-serving commands for the A4000 are documented in [`model-serving/README.md`](model-serving/README.md). Model endpoints remain separately hosted and configurable.

## Application server

```bash
uvicorn vllm_evaluation.app:create_app --factory --host 127.0.0.1 --port 8000
```

Open `/` for the operator dashboard, `/docs` for the operator API, and `/review` for the blind-review page. Operators can import snapshots, add model configurations, create and execute runs, build comparisons, create assignments, inspect submitted judgments, view summaries, and create JSONL/CSV exports. Reviewers can load an assignment, inspect text and image inputs, submit left/right/tie judgments, assess answer acceptability, record error tags and comments, and revisit an assignment. The application can run against the deterministic fake runner before a live endpoint is configured.

Runs use the live runner by default and call the endpoint and served model saved in the selected
model configuration. For simulated workflow testing, explicitly set `VLLM_EVAL_RUNNER_TYPE=fake`.
The dashboard displays the active runner mode. After switching from fake to live mode, restart the
app and create a new run: executing a completed run again preserves its existing answers.

## Automated evaluation

The **Automated evaluation** section runs the full workflow while retaining the manual controls:

1. Add model configurations with their exact served model IDs and endpoint URLs.
2. Select one task bundle and at least two models, and name the batch.
3. Leave **Automatically swap the local GPU model** checked for sequential local vLLM serving.
   All selected models must use the same local endpoint port, such as
   `http://127.0.0.1:8001/v1/chat/completions`. The app host must have Docker access and the
   existing NVIDIA/container setup from `model-serving/README.md`.
4. Click **Start automation**. The app imports the bundle, creates fresh runs, runs it
   on the first model, swaps models, and repeats. It waits for `/v1/models` to identify the expected
   model before sending tasks. Every model pair gets one frozen comparison for the selected bundle.

Model order follows the model list. Startup settings and the model cache come from
`model-serving/models.env`; the selected model and endpoint port override that file for each launch.
The serving script defaults to `./model-serving/serve-model.sh`; override it with
`[automation] serve_script` in a loaded TOML configuration or `VLLM_EVAL_AUTOMATION_SERVE_SCRIPT`.
Automatic swapping is intended for the app running on the Docker host. For independently hosted
model endpoints (including the containerized evaluation app), uncheck automatic swapping; those
endpoints must already serve their respective models.

Jobs run in the background and their configuration and results persist in the database. The page
shows live progress, history, errors, and buttons to open frozen comparisons for review. Closing the
page does not stop a job. **Stop automation** stops after the current request or startup command;
completed answers remain saved and the last loaded model stays running. Manual execution and other
automation jobs are blocked while a job owns the model server.

Automation uses live inference and sequential tasks. Any failed or truncated task stops the job;
it does not freeze partial comparisons. Inspect the saved run with the existing attempt controls
before starting a fresh job. Keep the app running: an app restart marks unfinished jobs interrupted,
preserving completed results rather than automatically repeating model swaps. The new
`automation_jobs` table is created on app startup, or by `alembic upgrade head` for migrated databases.

## Later-phase materials

- [`operator_guide.md`](operator_guide.md) describes the operator and review workflow.
- [`deployment.md`](deployment.md) describes containerisation, persistent storage, and offline transfer.
- [`validation_report.md`](validation_report.md) records acceptance scenarios and known limitations.

## Development checks

```bash
pytest
ruff check .
ruff format --check .
```
