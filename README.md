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

The current bootstrap command only validates configuration and creates configured local storage directories. Task import, persistence models, model runners, and the web interface are implemented in later phases.

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

The application does not download models or require a GPU. Model endpoints will be supplied separately when the runner is implemented.

## Development checks

```bash
pytest
ruff check .
ruff format --check .
```
