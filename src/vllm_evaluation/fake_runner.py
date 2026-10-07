"""Deterministic runner used before live model deployments are available."""

from __future__ import annotations

from enum import StrEnum

from .models import AttemptStatus
from .runner import (
    ModelRunner,
    RunnerEvent,
    RunnerRequest,
    RunnerResult,
    result_from_events,
)


class FakeScenario(StrEnum):
    SUCCESS = "success"
    REASONING = "reasoning"
    NO_USAGE = "no_usage"
    TIMEOUT = "timeout"
    MALFORMED_STREAM = "malformed_stream"
    EMPTY_RESPONSE = "empty_response"
    TRUNCATED = "truncated"


class FakeRunner(ModelRunner):
    """Predictable runner with explicit edge-case scenarios."""

    def __init__(
        self,
        *,
        scenario: FakeScenario = FakeScenario.SUCCESS,
        answer: str = "Fake runner answer.",
        reasoning: str = "Fake runner reasoning.",
    ) -> None:
        self.scenario = scenario
        self.answer = answer
        self.reasoning = reasoning

    def generate(self, request: RunnerRequest) -> RunnerResult:
        del request
        if self.scenario == FakeScenario.TIMEOUT:
            return RunnerResult(
                status=AttemptStatus.TIMED_OUT,
                error_code="timeout",
                error_message="fake runner timeout",
                retryable=True,
            )
        if self.scenario == FakeScenario.MALFORMED_STREAM:
            return RunnerResult(
                status=AttemptStatus.FAILED,
                raw_response=[{"malformed": True}],
                error_code="malformed_stream",
                error_message="fake runner emitted malformed stream data",
                retryable=True,
            )
        if self.scenario == FakeScenario.EMPTY_RESPONSE:
            return RunnerResult(
                status=AttemptStatus.FAILED,
                raw_response=[],
                error_code="empty_response",
                error_message="fake runner emitted no content",
            )

        events = [RunnerEvent(event_type="content", text=self.answer)]
        if self.scenario == FakeScenario.REASONING:
            events.insert(0, RunnerEvent(event_type="reasoning", text=self.reasoning))
        if self.scenario != FakeScenario.NO_USAGE:
            events.append(
                RunnerEvent(
                    event_type="finish",
                    finish_reason="length" if self.scenario == FakeScenario.TRUNCATED else "stop",
                    usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                )
            )
        else:
            events.append(RunnerEvent(event_type="finish", finish_reason="stop"))

        return result_from_events(
            events,
            raw_events=[{"fake": event.event_type, "text": event.text} for event in events],
        )
