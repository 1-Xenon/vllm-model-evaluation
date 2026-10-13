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
from vllm_evaluation.models import ComparisonSession, GenerationAttempt
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

    def test_comparison_rejects_stale_selections_without_creating_empty_drafts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                '{"task_id":"task-001","category":"text","prompt":"Summarise.",'
                '"inputs":[{"type":"text","text":"Text"}]}\n'
                '{"task_id":"task-002","category":"text","prompt":"Describe.",'
                '"inputs":[{"type":"text","text":"Text"}]}\n'
            )
            with session_scope(self.factory) as session:
                snapshot = import_task_bundle(session, bundle_root=root)
                repo = RunRepository()
                model = repo.create_model_config(
                    session,
                    display_name="Model",
                    endpoint_url="http://model.invalid",
                    served_model_name="model",
                )
                runs = [
                    repo.create_run(session, snapshot=snapshot, model_config=model)
                    for _ in range(2)
                ]
                selections = []
                for run in runs:
                    execute_run(session, run, FakeRunner(), config=ExecutionConfig())
                    selections.append(
                        list(
                            session.scalars(
                                select(GenerationAttempt)
                                .where(GenerationAttempt.run_id == run.id)
                                .order_by(GenerationAttempt.task_id)
                            )
                        )
                    )
                a, b = selections
                manifest.write_text(manifest.read_text().replace("Summarise.", "Other prompt."))
                other_snapshot = import_task_bundle(session, bundle_root=root)
                cases = [
                    (other_snapshot.id, [a[0].id], [b[0].id], "different snapshot"),
                    (snapshot.id, [], [b[0].id], "both model runs"),
                    (snapshot.id, ["missing-id"], [b[0].id], "no longer exist"),
                    (snapshot.id, [a[0].id], [a[0].id], "different attempts"),
                    (snapshot.id, [a[0].id], [b[1].id], "no complete successful task pairs"),
                    (snapshot.id, [a[0].id, b[0].id], [b[1].id], "one attempt per task"),
                ]
                for snapshot_id, a_ids, b_ids, message in cases:
                    with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                        create_comparison(
                            session,
                            snapshot_id=snapshot_id,
                            name="invalid",
                            assignment_seed="1",
                            model_a_attempt_ids=a_ids,
                            model_b_attempt_ids=b_ids,
                        )
                    self.assertEqual(list(session.scalars(select(ComparisonSession))), [])
                result = create_comparison(
                    session,
                    snapshot_id=snapshot.id,
                    name="partial",
                    assignment_seed="1",
                    model_a_attempt_ids=[a[0].id, a[1].id],
                    model_b_attempt_ids=[b[0].id],
                )
                self.assertEqual(len(result.session.items), 1)
                self.assertEqual([issue.code for issue in result.issues], ["missing_pair"])

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
