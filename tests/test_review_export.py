import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from vllm_evaluation.db import create_schema, session_scope
from vllm_evaluation.execution import ExecutionConfig, execute_run
from vllm_evaluation.export import comparison_summary, export_csv, export_jsonl
from vllm_evaluation.fake_runner import FakeRunner
from vllm_evaluation.importer import import_task_bundle
from vllm_evaluation.models import GenerationAttempt
from vllm_evaluation.repositories import RunRepository
from vllm_evaluation.review import (
    create_comparison,
    ensure_reviewer_assignments,
    freeze_comparison,
    reviewer_view,
    submit_judgment,
)


class ReviewAndExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:", future=True)
        create_schema(self.engine)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_blind_assignments_are_stable_and_judgment_history_is_versioned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.jsonl").write_text(
                '{"task_id":"task-001","category":"text_summarisation",'
                '"prompt":"Summarise.","inputs":[{"type":"text","text":"Text"}]}\n'
            )
            with session_scope(self.factory) as session:
                snapshot = import_task_bundle(session, bundle_root=root)
                repo = RunRepository()
                model_a = repo.create_model_config(
                    session,
                    display_name="Model A",
                    endpoint_url="http://a.invalid",
                    served_model_name="a",
                )
                model_b = repo.create_model_config(
                    session,
                    display_name="Model B",
                    endpoint_url="http://b.invalid",
                    served_model_name="b",
                )
                run_a = repo.create_run(session, snapshot=snapshot, model_config=model_a)
                run_b = repo.create_run(session, snapshot=snapshot, model_config=model_b)
                execute_run(session, run_a, FakeRunner(answer="A"), config=ExecutionConfig())
                execute_run(session, run_b, FakeRunner(answer="B"), config=ExecutionConfig())
                attempts = list(
                    session.scalars(select(GenerationAttempt).order_by(GenerationAttempt.run_id))
                )
                comparison = create_comparison(
                    session,
                    snapshot_id=snapshot.id,
                    name="smoke",
                    assignment_seed="fixed-seed",
                    model_a_attempt_ids=[attempts[0].id],
                    model_b_attempt_ids=[attempts[1].id],
                ).session
                freeze_comparison(session, comparison.id)
                first = ensure_reviewer_assignments(session, comparison.id, "reviewer-1")[0]
                second = ensure_reviewer_assignments(session, comparison.id, "reviewer-1")[0]
                self.assertEqual(first.id, second.id)
                view = reviewer_view(session, first.id)
                self.assertNotIn("Model A", str(view))
                self.assertNotIn("http://a.invalid", str(view))
                submit_judgment(
                    session,
                    first.id,
                    {"overall_preference": "left", "left_acceptability": "acceptable"},
                )
                submit_judgment(
                    session,
                    first.id,
                    {"overall_preference": "tie", "comments": "revisited"},
                )
                summary = comparison_summary(session, comparison.id)
                self.assertEqual(summary["judgment_count"], 1)
                self.assertTrue(summary["counts_are_reviewer_votes"])
                jsonl_path = export_jsonl(
                    session, root / "export.jsonl", comparison_id=comparison.id
                )
                csv_path = export_csv(session, root / "export.csv", comparison_id=comparison.id)
                self.assertIn('"record_type": "judgment"', jsonl_path.read_text())
                self.assertIn("record_type", csv_path.read_text())
