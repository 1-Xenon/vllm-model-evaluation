"""Stable runner contracts and output normalization for model execution."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from .models import AttemptStatus


class ReasoningStatus(StrEnum):
    EMITTED = "emitted"
    NOT_EMITTED = "not_emitted"
    UNSUPPORTED = "unsupported"
    UNPARSED = "unparsed"


@dataclass(frozen=True, slots=True)
class RunnerInput:
    """Model-facing input data after task assets have been loaded."""

    input_type: str
    text: str | None = None
    image_bytes: bytes | None = None
    image_media_type: str | None = None


@dataclass(frozen=True, slots=True)
class RunnerRequest:
    """A stateless request sent to one configured model endpoint."""

    request_id: str
    model_name: str
    prompt: str
    inputs: tuple[RunnerInput, ...] = ()
    system_instruction: str | None = None
    generation_settings: Mapping[str, Any] = field(default_factory=dict)
    stream: bool = True


@dataclass(frozen=True, slots=True)
class RunnerEvent:
    """Normalized event emitted by a streaming or fake runner."""

    event_type: str
    text: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RunnerResult:
    """Normalized result suitable for persistence by the execution layer."""

    status: AttemptStatus
    final_answer: str | None = None
    reasoning: str | None = None
    reasoning_status: ReasoningStatus = ReasoningStatus.NOT_EMITTED
    finish_reason: str | None = None
    token_usage: dict[str, Any] | None = None
    raw_response: dict[str, Any] | list[Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class RunnerAttempt:
    """One execution result, retaining its retry identity."""

    attempt_number: int
    result: RunnerResult


class ModelRunner(Protocol):
    """Interface implemented by fake and endpoint-backed runners."""

    def generate(self, request: RunnerRequest) -> RunnerResult:
        """Generate one result for a stateless request."""


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            part.get("text", "")
            for part in value
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def result_from_events(
    events: Iterable[RunnerEvent],
    *,
    raw_events: list[dict[str, Any]] | None = None,
    reasoning_status: ReasoningStatus | None = None,
) -> RunnerResult:
    """Combine normalized events into one persisted result."""

    final_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: str | None = None
    token_usage: dict[str, Any] | None = None
    saw_reasoning = False
    for event in events:
        if event.event_type == "content" and event.text:
            final_parts.append(event.text)
        elif event.event_type == "reasoning" and event.text:
            reasoning_parts.append(event.text)
            saw_reasoning = True
        if event.finish_reason is not None:
            finish_reason = event.finish_reason
        if event.usage is not None:
            token_usage = event.usage

    final_answer = "".join(final_parts)
    reasoning = "".join(reasoning_parts) or None
    if not final_answer and not reasoning:
        return RunnerResult(
            status=AttemptStatus.FAILED,
            reasoning_status=reasoning_status or ReasoningStatus.NOT_EMITTED,
            raw_response=raw_events,
            error_code="empty_response",
            error_message="runner emitted no answer or reasoning content",
        )

    status = AttemptStatus.TRUNCATED if finish_reason == "length" else AttemptStatus.SUCCEEDED
    return RunnerResult(
        status=status,
        final_answer=final_answer or None,
        reasoning=reasoning,
        reasoning_status=(
            reasoning_status
            or (ReasoningStatus.EMITTED if saw_reasoning else ReasoningStatus.NOT_EMITTED)
        ),
        finish_reason=finish_reason,
        token_usage=token_usage,
        raw_response=raw_events,
    )


def result_from_message(
    payload: dict[str, Any],
    *,
    reasoning_status: ReasoningStatus | None = None,
) -> RunnerResult:
    """Normalize an OpenAI-compatible non-streaming response."""

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return RunnerResult(
            status=AttemptStatus.FAILED,
            raw_response=payload,
            error_code="malformed_response",
            error_message="response did not contain a usable choices array",
        )
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return RunnerResult(
            status=AttemptStatus.FAILED,
            raw_response=payload,
            error_code="malformed_response",
            error_message="response choice did not contain a message object",
        )

    events: list[RunnerEvent] = []
    content = _extract_text(message.get("content"))
    if content:
        events.append(RunnerEvent(event_type="content", text=content))
    reasoning = message.get("reasoning_content", message.get("reasoning"))
    if isinstance(reasoning, str) and reasoning:
        events.append(RunnerEvent(event_type="reasoning", text=reasoning))
    events.append(
        RunnerEvent(
            event_type="finish",
            finish_reason=choices[0].get("finish_reason"),
            usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else None,
        )
    )
    return result_from_events(events, raw_events=[payload], reasoning_status=reasoning_status)


def execute_with_retries(
    runner: ModelRunner,
    request: RunnerRequest,
    *,
    max_retries: int,
    backoff_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> list[RunnerAttempt]:
    """Execute a request with bounded retries and distinct attempt numbers."""

    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    attempts: list[RunnerAttempt] = []
    for attempt_number in range(1, max_retries + 2):
        result = runner.generate(request)
        attempts.append(RunnerAttempt(attempt_number=attempt_number, result=result))
        if not result.retryable or result.status == AttemptStatus.SUCCEEDED:
            break
        if attempt_number <= max_retries and backoff_seconds > 0:
            sleep(backoff_seconds * (2 ** (attempt_number - 1)))
    return attempts
