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
