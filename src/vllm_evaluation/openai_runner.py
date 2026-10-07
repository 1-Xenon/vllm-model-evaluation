"""Placeholder OpenAI-compatible HTTP runner for local VLM endpoints."""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import AttemptStatus
from .runner import (
    ModelRunner,
    RunnerEvent,
    RunnerInput,
    RunnerRequest,
    RunnerResult,
    result_from_events,
    result_from_message,
)


@dataclass(frozen=True, slots=True)
class OpenAICompatibleConfig:
    endpoint_url: str
    model_name: str
    timeout_seconds: float = 120.0
    default_stream: bool = True
    auth_token: str | None = None


def _data_url(value: RunnerInput) -> str:
    if value.image_bytes is None or not value.image_media_type:
        raise ValueError("image input requires bytes and a media type")
    encoded = base64.b64encode(value.image_bytes).decode("ascii")
    return f"data:{value.image_media_type};base64,{encoded}"


def _message_content(request: RunnerRequest) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
    for item in request.inputs:
        if item.input_type == "text" and item.text is not None:
            content.append({"type": "text", "text": item.text})
        elif item.input_type == "image":
            content.append({"type": "image_url", "image_url": {"url": _data_url(item)}})
        else:
            raise ValueError(f"unsupported runner input type: {item.input_type}")
    return content


def build_chat_payload(request: RunnerRequest, *, default_stream: bool = True) -> dict[str, Any]:
    """Build a credential-free OpenAI-compatible chat request."""

    messages: list[dict[str, Any]] = []
    if request.system_instruction:
        messages.append({"role": "system", "content": request.system_instruction})
    messages.append({"role": "user", "content": _message_content(request)})
    payload = {
        "model": request.model_name,
        "messages": messages,
        "stream": request.stream if request.stream is not None else default_stream,
    }
    payload.update(dict(request.generation_settings))
    return payload


def _stream_event(payload: dict[str, Any]) -> RunnerEvent | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        usage = payload.get("usage")
        return RunnerEvent(
            event_type="usage",
            usage=usage if isinstance(usage, dict) else None,
            raw=payload,
        )
    choice = choices[0]
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        delta = {}
    if isinstance(delta.get("reasoning_content"), str):
        return RunnerEvent(
            event_type="reasoning",
            text=delta["reasoning_content"],
            raw=payload,
        )
    if isinstance(delta.get("reasoning"), str):
        return RunnerEvent(event_type="reasoning", text=delta["reasoning"], raw=payload)
    if isinstance(delta.get("content"), str):
        return RunnerEvent(event_type="content", text=delta["content"], raw=payload)
    return RunnerEvent(
        event_type="finish",
        finish_reason=choice.get("finish_reason"),
        usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else None,
        raw=payload,
    )


def parse_stream_lines(
    lines: Iterable[bytes],
    *,
    started_at: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RunnerResult:
    """Parse Server-Sent Events from an OpenAI-compatible response."""

    events: list[RunnerEvent] = []
    raw_events: list[dict[str, Any]] = []
    first_output_at: float | None = None
    first_answer_at: float | None = None
    try:
        for line in lines:
            text = line.decode("utf-8").strip()
            if not text or not text.startswith("data:"):
                continue
            data = text[len("data:") :].strip()
            if data == "[DONE]":
                break
            payload = json.loads(data)
            if not isinstance(payload, dict):
                raise ValueError("stream event must be a JSON object")
            raw_events.append(payload)
            event = _stream_event(payload)
            if event is not None:
                events.append(event)
                observed_at = clock()
                if event.event_type in {"content", "reasoning"} and event.text:
                    first_output_at = first_output_at or observed_at
                if event.event_type == "content" and event.text:
                    first_answer_at = first_answer_at or observed_at
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return RunnerResult(
            status=AttemptStatus.FAILED,
            raw_response=raw_events,
            error_code="malformed_stream",
            error_message=str(exc),
            retryable=True,
        )
    result = result_from_events(events, raw_events=raw_events)
    if started_at is not None:
        result = replace(
            result,
            ttft_seconds=first_output_at - started_at if first_output_at is not None else None,
            first_answer_seconds=(
                first_answer_at - started_at if first_answer_at is not None else None
            ),
            total_latency_seconds=clock() - started_at,
        )
    return result


class OpenAICompatibleRunner(ModelRunner):
    """HTTP runner for a local OpenAI-compatible chat-completions endpoint."""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        if not config.endpoint_url:
            raise ValueError("endpoint_url is required")
        if not config.model_name:
            raise ValueError("model_name is required")
        if config.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.config = config

    @property
    def chat_completions_url(self) -> str:
        endpoint = self.config.endpoint_url.rstrip("/")
        return (
            endpoint if endpoint.endswith("/chat/completions") else f"{endpoint}/chat/completions"
        )

    def generate(self, request: RunnerRequest) -> RunnerResult:
        payload = build_chat_payload(request, default_stream=self.config.default_stream)
        headers = {"Content-Type": "application/json"}
        if self.config.auth_token:
            headers["Authorization"] = f"Bearer {self.config.auth_token}"
        http_request = Request(
            self.chat_completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started_at = time.monotonic()
        try:
            with urlopen(http_request, timeout=self.config.timeout_seconds) as response:
                if payload["stream"]:
                    return parse_stream_lines(response, started_at=started_at)
                response_payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(response_payload, dict):
                    raise ValueError("response must be a JSON object")
                return replace(
                    result_from_message(response_payload),
                    total_latency_seconds=time.monotonic() - started_at,
                )
        except HTTPError as exc:
            return RunnerResult(
                status=AttemptStatus.FAILED,
                error_code=f"http_{exc.code}",
                error_message=f"endpoint returned HTTP {exc.code}",
                retryable=exc.code == 429 or exc.code >= 500,
            )
        except TimeoutError:
            return RunnerResult(
                status=AttemptStatus.TIMED_OUT,
                error_code="timeout",
                error_message="endpoint request timed out",
                retryable=True,
            )
        except URLError as exc:
            return RunnerResult(
                status=AttemptStatus.FAILED,
                error_code="endpoint_unreachable",
                error_message=str(exc.reason),
                retryable=True,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            return RunnerResult(
                status=AttemptStatus.FAILED,
                error_code="malformed_response",
                error_message=str(exc),
                retryable=True,
            )
