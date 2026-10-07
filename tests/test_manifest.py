import json
import tempfile
import unittest
from pathlib import Path

from vllm_evaluation.errors import ManifestValidationError
from vllm_evaluation.manifest import (
    calculate_manifest_hash,
    load_manifest,
    resolve_assets,
)


class ManifestTests(unittest.TestCase):
    def test_jsonl_manifest_preserves_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "task_id": "task-001",
                                "category": "mixed",
                                "prompt": "Read the input.",
                                "inputs": [
                                    {"type": "text", "text": "First."},
                                    {"type": "image", "path": "media/example.svg"},
                                ],
                            }
                        )
                    ]
                )
            )

            tasks = load_manifest(path)

        self.assertEqual([item.input_type for item in tasks[0].inputs], ["text", "image"])

    def test_duplicate_task_ids_are_structured_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            record = {
                "task_id": "duplicate",
                "category": "test",
                "prompt": "Prompt",
                "inputs": [{"type": "text", "text": "Input"}],
            }
            path.write_text(f"{json.dumps(record)}\n{json.dumps(record)}\n")

            with self.assertRaises(ManifestValidationError) as context:
                load_manifest(path)

        self.assertTrue(
            any(issue.code == "duplicate_task_id" for issue in context.exception.issues)
        )

    def test_asset_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "task_id": "unsafe",
                        "category": "test",
                        "prompt": "Prompt",
                        "inputs": [{"type": "image", "path": "../secret.png"}],
                    }
                )
            )
            tasks = load_manifest(path)

            with self.assertRaises(ManifestValidationError) as context:
                resolve_assets(tasks, directory)

        self.assertTrue(
            any(issue.code == "unsafe_asset_path" for issue in context.exception.issues)
        )

    def test_asset_hash_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "media"
            media.mkdir()
            (media / "example.svg").write_text("<svg />")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "task_id": "image-001",
                        "category": "test",
                        "prompt": "Inspect.",
                        "inputs": [{"type": "image", "path": "media/example.svg"}],
                    }
                )
            )
            tasks = load_manifest(manifest)
            assets = resolve_assets(tasks, root)

        self.assertEqual(len(assets), 1)
        self.assertEqual(len(assets[0].content_hash), 64)
        self.assertEqual(calculate_manifest_hash(tasks), calculate_manifest_hash(tasks))
