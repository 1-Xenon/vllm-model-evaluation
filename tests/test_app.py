import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from fastapi import HTTPException
from sqlalchemy import select

from vllm_evaluation.app import create_app
from vllm_evaluation.models import Task


class ApplicationSmokeTests(unittest.TestCase):
    def test_default_execution_calls_selected_model_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.toml"
            config.write_text(
                "[storage]\n"
                f'data_root = "{root}"\n'
                f'database_url = "sqlite:///{root}/test.db"\n'
                f'task_bundle_root = "{root}/bundles"\n'
                f'results_root = "{root}/results"\n'
                f'exports_root = "{root}/exports"\n'
                '[runner]\nendpoint_url = "http://wrong.invalid/v1"\n'
                'model_name = "wrong-model"\n'
            )
            app = create_app(config)
            routes = {route.path: route.endpoint for route in app.routes}
            bundle = root / "bundles" / "example"
            bundle.mkdir()
            (bundle / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "task_id": "test-001",
                        "category": "text",
                        "prompt": "Describe.",
                        "inputs": [{"type": "text", "text": "A vehicle."}],
                    }
                )
                + "\n"
            )
            snapshot = routes["/api/snapshots/import"]({"bundle_root": str(bundle)})
            model = routes["/api/model-configs"](
                {
                    "display_name": "Selected model",
                    "endpoint_url": "http://selected.invalid/v1",
                    "served_model_name": "selected-model",
                }
            )
            run_payload = {"snapshot_id": snapshot["id"], "model_config_id": model["id"]}
            create_run = next(
                route.endpoint
                for route in app.routes
                if route.path == "/api/runs" and "POST" in route.methods
            )
            run = create_run(run_payload)
            response = io.BytesIO(
                b'data: {"choices":[{"delta":{"content":"A vehicle is visible."}}]}\n\n'
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                b"data: [DONE]\n\n"
            )
            with patch("vllm_evaluation.openai_runner.urlopen", return_value=response) as send:
                routes["/api/runs/{run_id}/execute"](run["id"])
            request = send.call_args.args[0]
            self.assertEqual(request.full_url, "http://selected.invalid/v1/chat/completions")
            self.assertEqual(json.loads(request.data)["model"], "selected-model")
            attempts = routes["/api/runs/{run_id}/attempts"](run["id"])
            self.assertEqual(attempts[0]["final_answer"], "A vehicle is visible.")
            self.assertEqual(attempts[0]["status"], "succeeded")

            failed_run = create_run(run_payload)
            with patch("vllm_evaluation.openai_runner.urlopen", side_effect=URLError("offline")):
                routes["/api/runs/{run_id}/execute"](failed_run["id"])
            attempts = routes["/api/runs/{run_id}/attempts"](failed_run["id"])
            self.assertEqual(attempts[0]["status"], "failed")
            self.assertIsNone(attempts[0]["final_answer"])

    def test_tasks_jsonl_bundle_is_discovered_and_imported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.toml"
            config.write_text(
                "[storage]\n"
                f'data_root = "{root}/data"\n'
                f'database_url = "sqlite:///{root}/data/test.db"\n'
                f'task_bundle_root = "{root}/data/task-bundles"\n'
                f'results_root = "{root}/data/results"\n'
                f'exports_root = "{root}/data/exports"\n'
            )
            app = create_app(config)
            parent = app.state.settings.task_bundle_root
            bundle = parent / "4w1h_test_set"
            bundle.mkdir()
            (bundle / "media").mkdir()
            (bundle / "media/example.svg").write_text("<svg />")
            task = {
                "task_id": "4w1h-001",
                "category": "image_captioning_4w1h",
                "prompt": "Describe WHO, WHEN, WHERE, WHAT and HOW.",
                "inputs": [{"type": "image", "path": "media/example.svg"}],
            }
            (bundle / "tasks.jsonl").write_text(json.dumps(task) + "\n")
            (bundle / "manifest.json").write_text('{"tasks": [{"task_id": "metadata"}]}')
            (bundle / "tasks_with_context.jsonl").write_text(
                json.dumps({**task, "task_id": "context-001"}) + "\n"
            )
            legacy = parent / "legacy"
            legacy.mkdir()
            (legacy / "manifest.jsonl").write_text(json.dumps(task) + "\n")
            (legacy / "tasks.jsonl").write_text("invalid fallback should not be selected")
            (parent / "empty").mkdir()
            routes = {route.path: route.endpoint for route in app.routes}

            bundles = routes["/api/task-bundles"]()
            self.assertEqual(
                [(item["name"], item["manifest_path"]) for item in bundles],
                [("4w1h_test_set", "tasks.jsonl"), ("legacy", "manifest.jsonl")],
            )
            payload = {"bundle_root": bundles[0]["id"]}
            snapshot = routes["/api/snapshots/import"](payload)
            self.assertEqual(snapshot["task_count"], 1)
            self.assertEqual(snapshot["source_name"], "tasks.jsonl")
            self.assertEqual(snapshot["bundle_name"], "4w1h_test_set")
            repeated = routes["/api/snapshots/import"](
                {**payload, "manifest_path": bundles[0]["manifest_path"]}
            )
            self.assertEqual(repeated["id"], snapshot["id"])
            self.assertEqual(len(routes["/api/snapshots"]()), 1)
            with app.state.session_factory() as session:
                imported = session.scalar(select(Task))
                self.assertEqual(imported.stable_task_id, task["task_id"])
                self.assertEqual(imported.prompt, task["prompt"])
            media = routes["/api/snapshots/{snapshot_id}/media/{asset_path:path}"](
                snapshot["id"], "media/example.svg"
            )
            self.assertEqual(Path(media.path), bundle / "media/example.svg")
            context = routes["/api/snapshots/import"](
                {**payload, "manifest_path": "tasks_with_context.jsonl"}
            )
            self.assertNotEqual(context["id"], snapshot["id"])
            with self.assertRaises(HTTPException) as caught:
                routes["/api/snapshots/import"]({"bundle_root": str(root)})
            self.assertEqual(caught.exception.status_code, 400)

    def test_health_endpoint_boots_with_isolated_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.toml"
            config.write_text(
                """
[storage]
data_root = "./data"
database_url = "sqlite:///./data/test.db"
task_bundle_root = "./data/task-bundles"
results_root = "./data/results"
exports_root = "./data/exports"
"""
            )
            app = create_app(config)
            health_route = next(route for route in app.routes if route.path == "/health")
            self.assertEqual(
                health_route.endpoint(), {"status": "ok", "runner_type": "openai_compatible"}
            )

    def test_reviewer_page_contains_blind_judgment_form(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.toml"
            config.write_text(
                """
[storage]
data_root = "./data"
database_url = "sqlite:///./data/test.db"
task_bundle_root = "./data/task-bundles"
results_root = "./data/results"
exports_root = "./data/exports"
"""
            )
            app = create_app(config)
            review_route = next(route for route in app.routes if route.path == "/review")
            page = review_route.endpoint()

            self.assertIn("Submit judgment", page)
            self.assertIn("overall_preference", page)
            self.assertIn("left_acceptability", page)
            self.assertIn("right_acceptability", page)
            self.assertIn("error_tags", page)
            self.assertIn("/api/review/assignments/", page)
            self.assertIn("<select id='assignment' required>", page)
            self.assertIn("fetch('/api/review/assignments')", page)
            self.assertIn("Task image", page)

    def test_operator_page_contains_workflow_and_export_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.toml"
            config.write_text(
                """
[storage]
data_root = "./data"
database_url = "sqlite:///./data/test.db"
task_bundle_root = "./data/task-bundles"
results_root = "./data/results"
exports_root = "./data/exports"
"""
            )
            app = create_app(config)
            operator_route = next(route for route in app.routes if route.path == "/")
            page = operator_route.endpoint()

            self.assertIn("Import snapshot", page)
            self.assertIn("Automated evaluation", page)
            self.assertIn("id='automation-bundles' required", page)
            self.assertNotIn("id='automation-bundles' multiple", page)
            self.assertIn("id='automation-models' multiple required", page)
            self.assertIn("Start automation", page)
            self.assertIn("Stop automation", page)
            self.assertIn("/api/automations", page)
            self.assertIn("<select id='bundle-root' required>", page)
            self.assertIn("/api/task-bundles", page)
            self.assertIn("bundle not recorded", page)
            self.assertIn("Create run", page)
            self.assertIn("record.run_name", page)
            self.assertIn("(record) => record.display_name);", page)
            self.assertIn("Create comparison", page)
            self.assertIn("id='attempts-a' multiple required", page)
            self.assertIn("id='attempts-b' multiple required", page)
            self.assertNotIn("comparison-created-by", page)
            self.assertIn("Create or load assignments", page)
            self.assertIn("id='assignment-comparison' required", page)
            self.assertIn("/api/comparisons", page)
            self.assertIn("Freeze the comparison first", page)
            self.assertIn("Load judgments", page)
            self.assertIn("<select id='judgment-assignment' required>", page)
            self.assertIn("/api/review/assignments/' + id + '/judgments", page)
            self.assertIn("Create export", page)
            self.assertIn("/api/comparisons/", page)
