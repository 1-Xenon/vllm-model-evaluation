"""Create configured runner implementations without persisting credentials."""

from __future__ import annotations

import os
from collections.abc import Mapping

from .config import Settings
from .fake_runner import FakeRunner
from .openai_runner import OpenAICompatibleConfig, OpenAICompatibleRunner
from .runner import ModelRunner


def create_runner(
    settings: Settings,
    *,
    environ: Mapping[str, str] | None = None,
) -> ModelRunner:
    """Build a runner from settings; secrets are resolved only in memory."""

    if settings.runner_type == "fake":
        return FakeRunner()
    if settings.runner_type == "openai_compatible":
        environment = os.environ if environ is None else environ
        token = (
            environment.get(settings.runner_auth_token_env)
            if settings.runner_auth_token_env
            else None
        )
        return OpenAICompatibleRunner(
            OpenAICompatibleConfig(
                endpoint_url=settings.runner_endpoint_url,
                model_name=settings.runner_model_name,
                timeout_seconds=settings.runner_timeout_seconds,
                default_stream=settings.runner_streaming,
                auth_token=token,
            )
        )
    raise ValueError(f"unsupported runner type: {settings.runner_type}")
