import io
import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import HTTPException
from sqlalchemy import select

from vllm_evaluation.app import create_app
from vllm_evaluation.automation import (
    ACTIVE_STATUSES,
    AutomationBusy,
    AutomationCancelled,
    AutomationManager,
    AutomationRequest,
    LocalModelServer,
)
from vllm_evaluation.config import Settings, prepare_storage
from vllm_evaluation.db import (
    create_engine_from_settings,
    create_schema,
    create_session_factory,
    session_scope,
)
from vllm_evaluation.fake_runner import FakeRunner, FakeScenario
from vllm_evaluation.models import (
    AutomationJob,
    ComparisonSession,
    GenerationAttempt,
    ModelConfig,
    ModelRun,
)


class AutomationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.settings = Settings(
            data_root=self.root,
            database_url=f"sqlite:///{self.root}/test.db",
            task_bundle_root=self.root / "bundles",
            results_root=self.root / "results",
            exports_root=self.root / "exports",
        )
        prepare_storage(self.settings)
        self.engine = create_engine_from_settings(self.settings)
        create_schema(self.engine)
        self.factory = create_session_factory(self.engine)
        self.models = []
        with session_scope(self.factory) as session:
            for name in ("A", "B", "C"):
                model = ModelConfig(
                    display_name=name,
                    served_model_name=f"model-{name}",
                    endpoint_url="http://127.0.0.1:8001/v1/chat/completions",
                )
                session.add(model)
                session.flush()
                self.models.append(model)
        self.bundles = []
        for index, filename in enumerate(("tasks.jsonl", "manifest.jsonl")):
            bundle = self.settings.task_bundle_root / f"bundle-{index}"
            bundle.mkdir()
            (bundle / filename).write_text(
                "\n".join(
                    json.dumps(
                        {
                            "task_id": f"task-{index}-{task}",
                            "category": "text",
                            "prompt": "Summarise",
                            "inputs": [{"type": "text", "text": "Some text."}],
                        }
                    )
                    for task in range(2)
                )
            )
            self.bundles.append(str(bundle))
        self.server = Mock()
        self.events = []
        self.server.prepare.side_effect = lambda model, **kwargs: self.events.append(
            ("prepare", model.served_model_name)
        )

        def runner_factory(settings):
            outer = self

            class RecordingRunner(FakeRunner):
                def generate(self, request):
                    outer.events.append(("generate", request.model_name))
                    return super().generate(request)

            return RecordingRunner(answer=settings.runner_model_name)

        self.manager = AutomationManager(
            self.settings, self.factory, server=self.server, runner_factory=runner_factory
        )

    def tearDown(self):
        self.engine.dispose()
        self.directory.cleanup()

    def request(self, **overrides):
        return AutomationRequest(
            **{
                "name": "batch",
                "bundles": self.bundles[:1],
                "model_config_ids": [model.id for model in self.models],
                **overrides,
            }
        )

    def wait(self, job_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = self.manager.get(job_id)
            if result["status"] not in ACTIVE_STATUSES:
                try:
                    with self.manager.execution_slot():
                        return self.manager.get(job_id)
                except AutomationBusy:
                    pass
            time.sleep(0.01)
        self.fail("automation did not finish")

    def test_models_run_sequentially_on_one_bundle_and_all_pairs_are_frozen(self):
        result = self.wait(self.manager.start(self.request())["id"])
        self.assertEqual(result["status"], "completed", result["error_message"])
        self.assertEqual(
            self.events,
            [
                event
                for name in ("A", "B", "C")
                for event in [("prepare", f"model-{name}"), *[("generate", f"model-{name}")] * 2]
            ],
        )
        self.assertEqual(len(result["progress"]["runs"]), 3)
        self.assertEqual(len(result["progress"]["comparisons"]), 3)
        with self.factory() as session:
            self.assertEqual(len(list(session.scalars(select(GenerationAttempt)))), 6)
            comparisons = list(session.scalars(select(ComparisonSession)))
            self.assertTrue(
                all(item.status == "frozen" and len(item.items) == 2 for item in comparisons)
            )
            for comparison in comparisons:
                for item in comparison.items:
                    self.assertEqual(item.left_attempt.run.snapshot_id, comparison.snapshot_id)
                    self.assertEqual(item.right_attempt.run.snapshot_id, comparison.snapshot_id)
                    self.assertNotEqual(
                        item.left_attempt.run.model_config_id,
                        item.right_attempt.run.model_config_id,
                    )
        reloaded = AutomationManager(self.settings, self.factory, server=self.server)
        self.assertEqual(reloaded.get(result["id"])["status"], "completed")
        self.assertEqual(reloaded.list_jobs()[0]["config"]["bundles"], self.bundles[:1])

    def test_task_failure_stops_before_swapping_and_does_not_freeze(self):
        self.manager.runner_factory = lambda settings: FakeRunner(scenario=FakeScenario.TIMEOUT)
        result = self.wait(self.manager.start(self.request())["id"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("0/2 tasks succeeded", result["error_message"])
        self.assertEqual(self.server.prepare.call_count, 1)
        self.assertEqual(result["progress"]["comparisons"], [])
        with self.factory() as session:
            self.assertEqual(list(session.scalars(select(ComparisonSession))), [])

    def test_readiness_failure_preserves_job_without_executing_tasks(self):
        self.server.prepare.side_effect = TimeoutError("wrong model is loaded")
        result = self.wait(self.manager.start(self.request())["id"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("wrong model", result["error_message"])
        with self.factory() as session:
            self.assertEqual(list(session.scalars(select(GenerationAttempt))), [])

    def test_active_job_blocks_manual_execution_and_other_workers(self):
        entered, release = threading.Event(), threading.Event()

        def wait_for_model(model, **kwargs):
            entered.set()
            release.wait(5)
            if kwargs["cancelled"]():
                raise AutomationCancelled()

        self.server.prepare.side_effect = wait_for_model
        job = self.manager.start(self.request())
        self.assertTrue(entered.wait(5))
        try:
            other = AutomationManager(self.settings, self.factory, server=self.server)
            self.assertEqual(other.get(job["id"])["status"], "running")
            with self.assertRaises(AutomationBusy), other.execution_slot():
                pass
            with self.assertRaises(AutomationBusy):
                other.start(self.request())
            self.manager.cancel(job["id"])
        finally:
            release.set()
        self.assertEqual(self.wait(job["id"])["status"], "cancelled")

    def test_stop_after_current_task_keeps_completed_attempt(self):
        manager = self.manager

        class StoppingRunner(FakeRunner):
            def generate(self, request):
                manager.cancel(manager.list_jobs()[0]["id"])
                return super().generate(request)

        self.manager.runner_factory = lambda settings: StoppingRunner()
        result = self.wait(self.manager.start(self.request())["id"])
        self.assertEqual(result["status"], "cancelled")
        self.assertTrue(all(item["status"] == "cancelled" for item in result["progress"]["runs"]))
        with self.factory() as session:
            self.assertEqual(len(list(session.scalars(select(GenerationAttempt)))), 1)
            runs = list(session.scalars(select(ModelRun)))
            self.assertTrue(all(run.status == "cancelled" for run in runs))
            self.assertEqual(list(session.scalars(select(ComparisonSession))), [])

    def test_validation_rolls_back_and_releases_execution_slot(self):
        cases = [
            {"bundles": [str(self.root)]},
            {"bundles": []},
            {"bundles": self.bundles},
            {"model_config_ids": [self.models[0].id, "missing"]},
            {"model_config_ids": [self.models[0].id, self.models[0].id]},
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.manager.start(self.request(**values))
            with self.manager.execution_slot():
                pass
        with self.factory() as session:
            self.assertEqual(list(session.scalars(select(AutomationJob))), [])
            self.assertEqual(list(session.scalars(select(ModelRun))), [])
        self.manager.settings = replace(self.settings, runner_type="fake")
        with self.assertRaisesRegex(ValueError, "live runner"):
            self.manager.start(self.request())

    def test_stale_job_is_marked_interrupted_on_restart(self):
        with session_scope(self.factory) as session:
            job = AutomationJob(name="stale", status="running", config_json={}, progress_json={})
            session.add(job)
            session.flush()
        reloaded = AutomationManager(self.settings, self.factory, server=self.server)
        self.assertEqual(reloaded.get(job.id)["status"], "interrupted")

    def test_comparison_failure_rolls_back_all_comparisons(self):
        from vllm_evaluation.review import freeze_comparison

        count = 0

        def fail_second(session, comparison_id):
            nonlocal count
            count += 1
            if count == 2:
                raise ValueError("freeze failed")
            return freeze_comparison(session, comparison_id)

        with patch("vllm_evaluation.automation.freeze_comparison", side_effect=fail_second):
            result = self.wait(self.manager.start(self.request())["id"])
        self.assertEqual(result["status"], "failed")
        with self.factory() as session:
            self.assertEqual(list(session.scalars(select(ComparisonSession))), [])
            self.assertEqual(len(list(session.scalars(select(GenerationAttempt)))), 6)

    def test_app_routes_start_track_and_stop_jobs_and_block_manual_execution(self):
        config = self.root / "config.toml"
        config.write_text(
            "[storage]\n"
            f'data_root = "{self.root}"\n'
            f'database_url = "{self.settings.database_url}"\n'
            f'task_bundle_root = "{self.settings.task_bundle_root}"\n'
            f'results_root = "{self.settings.results_root}"\n'
            f'exports_root = "{self.settings.exports_root}"\n'
        )
        app = create_app(config)
        app.state.automation.server = self.server
        entered, release = threading.Event(), threading.Event()

        def prepare(model, **kwargs):
            entered.set()
            release.wait(5)
            raise AutomationCancelled()

        self.server.prepare.side_effect = prepare
        routes = {
            (route.path, method): route.endpoint for route in app.routes for method in route.methods
        }
        job = routes[("/api/automations", "POST")](self.request())
        self.assertTrue(entered.wait(5))
        try:
            listed = routes[("/api/automations", "GET")]()
            self.assertEqual(listed[0]["id"], job["id"])
            status = routes[("/api/automations/{job_id}", "GET")](job["id"])
            self.assertEqual(status["status"], "running")
            run_id = status["progress"]["runs"][0]["run_id"]
            with self.assertRaises(HTTPException) as error:
                routes[("/api/runs/{run_id}/execute", "POST")](run_id)
            self.assertEqual(error.exception.status_code, 409)
            with self.assertRaises(HTTPException) as error:
                routes[("/api/automations", "POST")](self.request())
            self.assertEqual(error.exception.status_code, 409)
            stopped = routes[("/api/automations/{job_id}/cancel", "POST")](job["id"])
            self.assertTrue(stopped["cancel_requested"])
        finally:
            release.set()
        self.assertEqual(self.wait(job["id"])["status"], "cancelled")


class LocalModelServerTests(unittest.TestCase):
    def setUp(self):
        self.server = LocalModelServer(Settings())
        self.model = ModelConfig(
            display_name="B",
            served_model_name="model-B",
            endpoint_url="http://127.0.0.1:8001/v1/chat/completions",
        )

    def test_switch_passes_explicit_model_and_waits_for_matching_identity(self):
        with (
            patch.object(self.server, "_ready", side_effect=[False, False, True]),
            patch("vllm_evaluation.automation.subprocess.run") as launch,
            patch("vllm_evaluation.automation.time.sleep"),
        ):
            launch.return_value = subprocess.CompletedProcess([], 0, stdout="container", stderr="")
            self.server.prepare(self.model, swap_models=True, timeout=10, cancelled=lambda: False)
        args = launch.call_args.args[0]
        self.assertEqual(args[-5:], ["--model", "model-B", "--port", "8001", "--detach"])
        self.assertNotIn("shell", launch.call_args.kwargs)

    def test_reuses_ready_model_and_existing_endpoint_mode_never_launches(self):
        for swap, readiness in ((True, [True]), (False, [False, True])):
            with (
                patch.object(self.server, "_ready", side_effect=readiness),
                patch("vllm_evaluation.automation.subprocess.run") as launch,
            ):
                self.server.prepare(
                    self.model, swap_models=swap, timeout=10, cancelled=lambda: False
                )
                launch.assert_not_called()

    def test_readiness_checks_model_id_and_normalizes_chat_endpoint(self):
        for name, expected in (("model-A", False), ("model-B", True)):
            reply = io.BytesIO(json.dumps({"data": [{"id": name}]}).encode())
            with patch("vllm_evaluation.automation.urlopen", return_value=reply) as get:
                self.assertEqual(self.server._ready(self.model), expected)
                self.assertEqual(get.call_args.args[0].full_url, "http://127.0.0.1:8001/v1/models")

    def test_timeout_and_launch_failure_are_explicit(self):
        with (
            patch.object(self.server, "_ready", return_value=False),
            patch("vllm_evaluation.automation.time.monotonic", side_effect=[0, 11]),
        ):
            with self.assertRaisesRegex(TimeoutError, "model-B"):
                self.server.prepare(
                    self.model, swap_models=False, timeout=10, cancelled=lambda: False
                )
        with (
            patch.object(self.server, "_ready", return_value=False),
            patch("vllm_evaluation.automation.subprocess.run") as launch,
        ):
            launch.return_value = subprocess.CompletedProcess([], 1, stdout="", stderr="GPU failed")
            with self.assertRaisesRegex(RuntimeError, "GPU failed"):
                self.server.prepare(
                    self.model, swap_models=True, timeout=10, cancelled=lambda: False
                )

    def test_swap_rejects_remote_endpoint_before_touching_docker(self):
        self.model.endpoint_url = "http://remote.invalid:8001/v1"
        with patch("vllm_evaluation.automation.subprocess.run") as launch:
            with self.assertRaisesRegex(ValueError, "local HTTP"):
                self.server.validate([self.model], True)
            launch.assert_not_called()

    def test_launcher_overrides_env_file_without_changing_manual_mode(self):
        script = Path("model-serving/serve-model.sh").resolve()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "docker.log"
            docker = root / "docker"
            docker.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$DOCKER_TEST_LOG"\n')
            docker.chmod(0o755)
            env_file = root / "models.env"
            env_file.write_text(
                f'VLLM_MODEL=env-model\nVLLM_PORT=8001\nVLLM_MODEL_CACHE="{root}/cache"\n'
            )
            environment = {
                **os.environ,
                "PATH": f"{root}:{os.environ['PATH']}",
                "VLLM_ENV_FILE": str(env_file),
                "DOCKER_TEST_LOG": str(log),
            }
            subprocess.run(
                ["bash", str(script), "--model", "selected-model", "--port", "8002", "--detach"],
                env=environment,
                check=True,
                capture_output=True,
            )
            command = log.read_text().splitlines()[-1]
            self.assertIn("--detach", command)
            self.assertIn("8002:8000", command)
            self.assertIn("selected-model", command)
            self.assertNotIn("env-model", command)
            subprocess.run(["bash", str(script)], env=environment, check=True, capture_output=True)
            command = log.read_text().splitlines()[-1]
            self.assertNotIn("--detach", command)
            self.assertIn("8001:8000", command)
            self.assertIn("env-model", command)
