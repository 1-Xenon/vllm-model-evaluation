import unittest

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from vllm_evaluation.db import create_schema, session_scope
from vllm_evaluation.errors import ValidationError
from vllm_evaluation.models import (
    GenerationAttempt,
    ModelConfig,
    Task,
)
from vllm_evaluation.repositories import RunRepository, TaskSnapshotRepository


class ModelPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:", future=True)
        create_schema(self.engine)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_snapshot_run_and_attempt_relationships_persist(self) -> None:
        snapshot_repository = TaskSnapshotRepository()
        run_repository = RunRepository()

        with session_scope(self.factory) as session:
            snapshot = snapshot_repository.create(
                session,
                source_name="example.jsonl",
                manifest_hash="a" * 64,
                snapshot_hash="b" * 64,
                task_count=1,
            )
            task = Task(
                snapshot=snapshot,
                stable_task_id="task-001",
                category="captioning",
                prompt="Describe the image.",
                task_order=0,
            )
            session.add(task)
            session.flush()
            model_config = run_repository.create_model_config(
                session,
                display_name="Model A",
                endpoint_url="http://model-a.invalid/v1",
                served_model_name="placeholder-a",
            )
            run = run_repository.create_run(
                session,
                snapshot=snapshot,
                model_config=model_config,
                run_label="A",
            )
            attempt = run_repository.create_attempt(
                session,
                run=run,
                task=task,
                attempt_number=1,
            )

            self.assertEqual(attempt.run.snapshot_id, snapshot.id)
            self.assertEqual(attempt.task.stable_task_id, "task-001")

        with self.factory() as session:
            saved_attempt = session.scalar(select(GenerationAttempt))
            self.assertIsNotNone(saved_attempt)
            self.assertEqual(saved_attempt.run.run_label, "A")

    def test_attempt_cannot_cross_task_snapshots(self) -> None:
        snapshot_repository = TaskSnapshotRepository()
        run_repository = RunRepository()

        with session_scope(self.factory) as session:
            first_snapshot = snapshot_repository.create(
                session,
                source_name="first.jsonl",
                manifest_hash="1" * 64,
                snapshot_hash="2" * 64,
            )
            second_snapshot = snapshot_repository.create(
                session,
                source_name="second.jsonl",
                manifest_hash="3" * 64,
                snapshot_hash="4" * 64,
            )
            task = Task(
                snapshot=second_snapshot,
                stable_task_id="task-002",
                category="captioning",
                prompt="Describe the image.",
                task_order=0,
            )
            model_config = ModelConfig(
                display_name="Model A",
                endpoint_url="http://model-a.invalid/v1",
                served_model_name="placeholder-a",
            )
            session.add_all([task, model_config])
            session.flush()
            run = run_repository.create_run(
                session,
                snapshot=first_snapshot,
                model_config=model_config,
            )

            with self.assertRaisesRegex(ValidationError, "same task snapshot"):
                run_repository.create_attempt(
                    session,
                    run=run,
                    task=task,
                    attempt_number=1,
                )
