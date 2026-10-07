import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from vllm_evaluation.db import create_schema, session_scope
from vllm_evaluation.importer import import_task_bundle
from vllm_evaluation.models import MediaAsset, Task, TaskInput, TaskSnapshot


class BundleImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:", future=True)
        create_schema(self.engine)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_bundle_import_creates_one_immutable_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "media").mkdir()
            (root / "media" / "example.svg").write_text("<svg />")
            (root / "manifest.jsonl").write_text(
                "{"
                '"task_id":"task-001",'
                '"category":"mixed",'
                '"prompt":"Inspect.",'
                '"inputs":['
                '{"type":"text","text":"Context"},'
                '{"type":"image","path":"media/example.svg",'
                '"transform":{"resize":{"width":320}}}]'
                "}\n"
            )

            with session_scope(self.factory) as session:
                first = import_task_bundle(session, bundle_root=root)
                second = import_task_bundle(session, bundle_root=root)

                self.assertEqual(first.id, second.id)
                self.assertEqual(len(session.scalars(select(TaskSnapshot)).all()), 1)

            with self.factory() as session:
                task = session.scalar(select(Task))
                inputs = list(session.scalars(select(TaskInput).order_by(TaskInput.ordinal)))
                asset = session.scalar(select(MediaAsset))

        self.assertEqual(task.stable_task_id, "task-001")
        self.assertEqual([item.input_type for item in inputs], ["text", "image"])
        self.assertEqual(inputs[1].transform_metadata_json, {"resize": {"width": 320}})
        self.assertEqual(asset.relative_path, "media/example.svg")
