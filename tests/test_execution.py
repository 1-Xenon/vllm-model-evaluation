import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from vllm_evaluation.db import create_schema, session_scope
from vllm_evaluation.execution import (
    ExecutionConfig,
    calculate_tps,
    create_model_run,
    execute_run,
    progress_for_run,
)
from vllm_evaluation.fake_runner import FakeRunner, FakeScenario
from vllm_evaluation.importer import import_task_bundle
from vllm_evaluation.models import AttemptStatus, GenerationAttempt, ModelConfig, ModelRun
from vllm_evaluation.runner import RunnerResult


class ExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:", future=True)
        create_schema(self.engine)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def tearDown(self) -> None:
        self.engine.dispose()

    def _create_run(self, session, bundle_root: Path) -> ModelRun:
        snapshot = import_task_bundle(session, bundle_root=bundle_root)
        model_config = ModelConfig(
            display_name="Fake model",
            endpoint_url="http://fake.invalid/v1",
            served_model_name="fake-model",
        )
        session.add(model_config)
        session.flush()
        return create_model_run(
            session, snapshot=snapshot, model_config=model_config, run_label="A"
        )

    def test_sequential_execution_persists_outputs_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.jsonl").write_text(
                '{"task_id":"task-001","category":"text","prompt":"Summarise.",'
                '"inputs":[{"type":"text","text":"Some text."}]}\n'
            )
            with session_scope(self.factory) as session:
                run = self._create_run(session, root)
                progress = execute_run(
                    session,
                    run,
                    FakeRunner(),
                    config=ExecutionConfig(),
                )
                self.assertEqual(progress.completed_tasks, 1)
                self.assertEqual(progress.pending_tasks, 0)

                second_progress = execute_run(
                    session,
                    run,
                    FakeRunner(answer="should not overwrite"),
                    config=ExecutionConfig(),
                )
                self.assertEqual(second_progress.completed_tasks, 1)
                attempts = list(session.scalars(select(GenerationAttempt)))

        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].final_answer, "Fake runner answer.")
        self.assertEqual(attempts[0].status, AttemptStatus.SUCCEEDED.value)

    def test_retry_attempts_are_persisted_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.jsonl").write_text(
                '{"task_id":"task-001","category":"text","prompt":"Summarise.",'
                '"inputs":[{"type":"text","text":"Some text."}]}\n'
            )
            with session_scope(self.factory) as session:
                run = self._create_run(session, root)
                execute_run(
                    session,
                    run,
                    FakeRunner(scenario=FakeScenario.TIMEOUT),
                    config=ExecutionConfig(max_retries=1),
                )
                attempts = list(
                    session.scalars(
                        select(GenerationAttempt).order_by(GenerationAttempt.attempt_number)
                    )
                )

        self.assertEqual([attempt.attempt_number for attempt in attempts], [1, 2])
        self.assertTrue(
            all(attempt.status == AttemptStatus.TIMED_OUT.value for attempt in attempts)
        )

    def test_tps_is_missing_when_timing_is_unavailable(self) -> None:
        result = RunnerResult(
            status=AttemptStatus.SUCCEEDED,
            final_answer="answer",
            token_usage={"completion_tokens": 10},
        )
        self.assertIsNone(calculate_tps(result))

    def test_progress_counts_tasks_not_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.jsonl").write_text(
                '{"task_id":"task-001","category":"text","prompt":"Summarise.",'
                '"inputs":[{"type":"text","text":"Some text."}]}\n'
            )
            with session_scope(self.factory) as session:
                run = self._create_run(session, root)
                execute_run(
                    session,
                    run,
                    FakeRunner(scenario=FakeScenario.TIMEOUT),
                    config=ExecutionConfig(max_retries=1),
                )
                progress = progress_for_run(session, run)

        self.assertEqual(progress.total_tasks, 1)
        self.assertEqual(progress.completed_tasks, 1)
        self.assertEqual(progress.failed_tasks, 1)
