import tempfile
import unittest
from pathlib import Path

from vllm_evaluation.app import create_app


class ApplicationSmokeTests(unittest.TestCase):
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
            self.assertEqual(health_route.endpoint(), {"status": "ok", "runner_type": "fake"})

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
            self.assertIn("Create run", page)
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
