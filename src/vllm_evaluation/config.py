"""Configuration loading with TOML files, defaults, and environment overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings for the initial application foundation."""

    app_name: str = "vllm-model-evaluation"
    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    data_root: Path = Path("./data")
    database_url: str = "sqlite:///./data/evaluation.db"
    task_bundle_root: Path = Path("./data/task-bundles")
    results_root: Path = Path("./data/results")
    exports_root: Path = Path("./data/exports")
    runner_type: str = "fake"
    runner_endpoint_url: str = ""
    runner_model_name: str = ""
    runner_timeout_seconds: float = 120.0
    runner_max_retries: int = 0
    runner_concurrency: int = 1
    runner_streaming: bool = True


_CONFIG_KEYS: dict[tuple[str, str], tuple[str, type]] = {
    ("app", "name"): ("app_name", str),
    ("app", "environment"): ("environment", str),
    ("server", "host"): ("host", str),
    ("server", "port"): ("port", int),
    ("server", "log_level"): ("log_level", str),
    ("storage", "data_root"): ("data_root", Path),
    ("storage", "database_url"): ("database_url", str),
    ("storage", "task_bundle_root"): ("task_bundle_root", Path),
    ("storage", "results_root"): ("results_root", Path),
    ("storage", "exports_root"): ("exports_root", Path),
    ("runner", "type"): ("runner_type", str),
    ("runner", "endpoint_url"): ("runner_endpoint_url", str),
    ("runner", "model_name"): ("runner_model_name", str),
    ("runner", "timeout_seconds"): ("runner_timeout_seconds", float),
    ("runner", "max_retries"): ("runner_max_retries", int),
    ("runner", "concurrency"): ("runner_concurrency", int),
    ("runner", "streaming"): ("runner_streaming", bool),
}

_ENV_KEYS: dict[str, tuple[str, type]] = {
    "VLLM_EVAL_APP_NAME": ("app_name", str),
    "VLLM_EVAL_ENVIRONMENT": ("environment", str),
    "VLLM_EVAL_HOST": ("host", str),
    "VLLM_EVAL_PORT": ("port", int),
    "VLLM_EVAL_LOG_LEVEL": ("log_level", str),
    "VLLM_EVAL_DATA_ROOT": ("data_root", Path),
    "VLLM_EVAL_DATABASE_URL": ("database_url", str),
    "VLLM_EVAL_TASK_BUNDLE_ROOT": ("task_bundle_root", Path),
    "VLLM_EVAL_RESULTS_ROOT": ("results_root", Path),
    "VLLM_EVAL_EXPORTS_ROOT": ("exports_root", Path),
    "VLLM_EVAL_RUNNER_TYPE": ("runner_type", str),
    "VLLM_EVAL_RUNNER_ENDPOINT_URL": ("runner_endpoint_url", str),
    "VLLM_EVAL_RUNNER_MODEL_NAME": ("runner_model_name", str),
    "VLLM_EVAL_RUNNER_TIMEOUT_SECONDS": ("runner_timeout_seconds", float),
    "VLLM_EVAL_RUNNER_MAX_RETRIES": ("runner_max_retries", int),
    "VLLM_EVAL_RUNNER_CONCURRENCY": ("runner_concurrency", int),
    "VLLM_EVAL_RUNNER_STREAMING": ("runner_streaming", bool),
}


def _coerce(value: object, value_type: type, *, source: str) -> object:
    if value_type is Path:
        if not isinstance(value, (str, Path)):
            raise ConfigurationError(f"{source} must be a path-like string")
        return Path(value)
    if value_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "1", "yes", "on"}:
            return True
        if isinstance(value, str) and value.lower() in {"false", "0", "no", "off"}:
            return False
        raise ConfigurationError(f"{source} must be a boolean")
    try:
        return value_type(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{source} has an invalid value: {value!r}") from exc


def _validate(settings: Settings) -> Settings:
    if not 1 <= settings.port <= 65535:
        raise ConfigurationError("server port must be between 1 and 65535")
    if settings.runner_timeout_seconds <= 0:
        raise ConfigurationError("runner timeout_seconds must be greater than zero")
    if settings.runner_max_retries < 0:
        raise ConfigurationError("runner max_retries cannot be negative")
    if settings.runner_concurrency < 1:
        raise ConfigurationError("runner concurrency must be at least one")
    if settings.runner_type not in {"fake", "openai_compatible"}:
        raise ConfigurationError("runner type must be 'fake' or 'openai_compatible'")
    return settings


def load_settings(
    config_path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Load defaults, then TOML values, then environment overrides."""

    environment = dict(os.environ if environ is None else environ)
    requested_path = config_path or environment.get("VLLM_EVAL_CONFIG")
    values: dict[str, object] = {}

    if requested_path:
        path = Path(requested_path)
        if not path.exists():
            raise ConfigurationError(f"configuration file does not exist: {path}")
        try:
            with path.open("rb") as config_file:
                document = tomllib.load(config_file)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigurationError(f"invalid TOML configuration: {path}") from exc

        for (section, key), (field_name, value_type) in _CONFIG_KEYS.items():
            section_values = document.get(section, {})
            if not isinstance(section_values, dict):
                raise ConfigurationError(f"configuration section [{section}] must be a table")
            if key in section_values:
                values[field_name] = _coerce(
                    section_values[key], value_type, source=f"[{section}] {key}"
                )

    for env_name, (field_name, value_type) in _ENV_KEYS.items():
        if env_name in environment:
            values[field_name] = _coerce(environment[env_name], value_type, source=env_name)

    return _validate(Settings(**values))


def prepare_storage(settings: Settings) -> None:
    """Create local storage directories required by the configured application."""

    for path in (
        settings.data_root,
        settings.task_bundle_root,
        settings.results_root,
        settings.exports_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

