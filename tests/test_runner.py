import base64
import io
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from vllm_evaluation.fake_runner import FakeRunner, FakeScenario
from vllm_evaluation.models import AttemptStatus
from vllm_evaluation.openai_runner import (
    OpenAICompatibleConfig,
    OpenAICompatibleRunner,
    build_chat_payload,
    parse_stream_lines,
)
from vllm_evaluation.runner import (
    ReasoningStatus,
    RunnerInput,
    RunnerRequest,
    RunnerResult,
    execute_with_retries,
)


def request(*, stream: bool = True) -> RunnerRequest:
    return RunnerRequest(
        request_id="request-001",
        model_name="placeholder-model",
        prompt="Describe the image.",
        inputs=(
            RunnerInput(input_type="image", image_bytes=b"image", image_media_type="image/png"),
        ),
        stream=stream,
    )


class FakeRunnerTests(unittest.TestCase):
    def test_success_and_reasoning_are_normalized(self) -> None:
        result = FakeRunner(scenario=FakeScenario.REASONING).generate(request())

        self.assertEqual(result.status, AttemptStatus.SUCCEEDED)
        self.assertEqual(result.reasoning_status, ReasoningStatus.EMITTED)
        self.assertEqual(result.final_answer, "Fake runner answer.")
        self.assertEqual(result.reasoning, "Fake runner reasoning.")

    def test_edge_cases_are_explicit(self) -> None:
        timeout = FakeRunner(scenario=FakeScenario.TIMEOUT).generate(request())
        truncated = FakeRunner(scenario=FakeScenario.TRUNCATED).generate(request())
        empty = FakeRunner(scenario=FakeScenario.EMPTY_RESPONSE).generate(request())

        self.assertEqual(timeout.status, AttemptStatus.TIMED_OUT)
        self.assertTrue(timeout.retryable)
        self.assertEqual(truncated.status, AttemptStatus.TRUNCATED)
        self.assertEqual(empty.error_code, "empty_response")


class OpenAICompatibleRunnerTests(unittest.TestCase):
    def test_http_errors_preserve_diagnostics_and_retry_policy(self) -> None:
        cases = [
            (400, b'{"error":{"message":"Cannot decode image"}}', "Cannot decode image"),
            (400, b'{"message":"Unknown model"}', "Unknown model"),
            (400, b'{"error":"Invalid settings"}', "Invalid settings"),
            (429, b"Too many requests", "Too many requests"),
            (500, b"", ""),
            (502, b"Bad gateway \xff", "Bad gateway \ufffd"),
        ]
        runner = OpenAICompatibleRunner(
            OpenAICompatibleConfig(endpoint_url="http://localhost/v1", model_name="model")
        )
        for code, body, detail in cases:
            with self.subTest(code=code, body=body):
                error = HTTPError(runner.chat_completions_url, code, "error", {}, io.BytesIO(body))
                with patch("vllm_evaluation.openai_runner.urlopen", side_effect=error):
                    result = runner.generate(request())
                prefix = f"endpoint returned HTTP {code}"
                self.assertEqual(result.status, AttemptStatus.FAILED)
                self.assertEqual(result.error_code, f"http_{code}")
                self.assertEqual(result.error_message, f"{prefix}: {detail}" if detail else prefix)
                self.assertEqual(result.retryable, code == 429 or code >= 500)
                self.assertIsNotNone(result.total_latency_seconds)
                self.assertEqual(result.raw_response is not None, bool(body))
                self.assertTrue(error.closed)
                if code == 400 and b"Cannot decode" in body:
                    self.assertEqual(
                        result.raw_response, {"error": {"message": "Cannot decode image"}}
                    )

    def test_http_error_body_is_bounded_and_read_failure_preserves_status(self) -> None:
        runner = OpenAICompatibleRunner(
            OpenAICompatibleConfig(endpoint_url="http://localhost/v1", model_name="model")
        )
        for fails in (False, True):
            with self.subTest(read_fails=fails):
                body = io.BytesIO(b"x" * 20_000)
                error = HTTPError(runner.chat_completions_url, 400, "error", {}, body)
                with (
                    patch("vllm_evaluation.openai_runner.urlopen", side_effect=error),
                    patch.object(
                        error, "read", side_effect=OSError("read failed") if fails else body.read
                    ) as read,
                ):
                    result = runner.generate(request())
                read.assert_called_once_with(16_384)
                self.assertEqual(result.error_code, "http_400")
                self.assertEqual(result.raw_response, None if fails else {"body": "x" * 16_384})

    def test_payload_uses_data_url_and_excludes_authentication(self) -> None:
        payload = build_chat_payload(request(), default_stream=True)
        image_url = payload["messages"][0]["content"][1]["image_url"]["url"]

        self.assertEqual(image_url, "data:image/png;base64," + base64.b64encode(b"image").decode())
        self.assertNotIn("Authorization", payload)

    def test_stream_parser_preserves_reasoning_and_usage(self) -> None:
        lines = [
            b'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"answer"}}]}\n',
            (
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                b'"usage":{"completion_tokens":2}}\n'
            ),
            b"data: [DONE]\n",
        ]
        result = parse_stream_lines(lines)

        self.assertEqual(result.status, AttemptStatus.SUCCEEDED)
        self.assertEqual(result.reasoning, "think")
        self.assertEqual(result.final_answer, "answer")
        self.assertEqual(result.token_usage, {"completion_tokens": 2})

    def test_runner_requires_endpoint_and_model(self) -> None:
        with self.assertRaises(ValueError):
            OpenAICompatibleRunner(OpenAICompatibleConfig(endpoint_url="", model_name="model"))


class RetryTests(unittest.TestCase):
    def test_retries_have_distinct_attempt_numbers(self) -> None:
        class RetryThenSuccess:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, request: RunnerRequest) -> RunnerResult:
                del request
                self.calls += 1
                if self.calls == 1:
                    return RunnerResult(
                        status=AttemptStatus.TIMED_OUT,
                        error_code="timeout",
                        retryable=True,
                    )
                return RunnerResult(status=AttemptStatus.SUCCEEDED, final_answer="done")

        attempts = execute_with_retries(RetryThenSuccess(), request(), max_retries=1)

        self.assertEqual([attempt.attempt_number for attempt in attempts], [1, 2])
        self.assertEqual(attempts[-1].result.final_answer, "done")
