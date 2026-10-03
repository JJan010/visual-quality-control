"""Deterministic worker tests; no model artifacts or GPU execution required.

The application module and its dependencies must be installed. Slow work is
controlled with threading.Event rather than a timing-dependent request race.
Run: python -m unittest discover -s tests -p test_api_worker.py -v
"""

import asyncio
import importlib
import threading
import time
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

api = importlib.import_module("visual_quality.api.app")


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.worker = api.ModelWorker()
        self.worker.metadata = {
            "backend": "test", "blur_backend": "test", "device": "cpu"
        }
        self.release_events = []
        self.tasks = []

    async def asyncTearDown(self):
        for event in self.release_events:
            event.set()
        if self.tasks:
            await asyncio.wait_for(
                asyncio.gather(*self.tasks, return_exceptions=True), timeout=5
            )
        self.worker.executor.shutdown(wait=True)

    async def eventually(self, condition):
        deadline = time.monotonic() + 5
        while not condition():
            if time.monotonic() >= deadline:
                self.fail("Timed out waiting for a controlled worker event.")
            await asyncio.sleep(0.005)

    def start(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task

    def gated_operation(self, *, fail=False):
        entered = threading.Event()
        release = threading.Event()
        self.release_events.append(release)

        def operation(data, include_visualization=False):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("Test gate was not released.")
            if fail:
                raise RuntimeError("Synthetic inference failure")
            return {"result": "ok", "thread": threading.get_ident()}

        self.worker.predict_sync = operation
        return entered, release

    async def test_load_predict_and_unload_use_one_non_main_thread(self):
        owner_threads = []
        main_thread = threading.get_ident()

        def fake_load(_):
            owner_threads.append(threading.get_ident())
            return SimpleNamespace(runtime_metadata=lambda: self.worker.metadata)

        def fake_inference(data, include_visualization=False):
            owner_threads.append(threading.get_ident())
            return "ok"

        def fake_synchronize(_):
            owner_threads.append(threading.get_ident())

        loop = asyncio.get_running_loop()
        with patch.object(api, "load_predictor", side_effect=fake_load) as loader, \
             patch.object(api.torch.cuda, "set_device"), \
             patch.object(api.torch.cuda, "synchronize", side_effect=fake_synchronize), \
             patch.dict(api.os.environ, {"VQC_LOCALIZATION_CALIBRATION": ""}):
            await asyncio.wait_for(
                loop.run_in_executor(self.worker.executor, self.worker.load, "test.json"), 5
            )
            self.worker.predict_sync = fake_inference
            self.assertEqual(await asyncio.wait_for(self.worker.predict(b"a"), 5), "ok")
            self.assertEqual(await asyncio.wait_for(self.worker.predict(b"b"), 5), "ok")
            await asyncio.wait_for(
                loop.run_in_executor(self.worker.executor, self.worker.unload), 5
            )
            loader.assert_called_once()

        self.assertEqual(len(owner_threads), 4)
        self.assertEqual(len(set(owner_threads)), 1)
        self.assertNotEqual(owner_threads[0], main_thread)
        self.assertIsNone(self.worker.predictor)
        self.assertFalse(self.worker.busy)

    async def test_busy_worker_rejects_second_request_and_health_stays_available(self):
        entered, release = self.gated_operation()
        first = self.start(self.worker.predict(b"first"))
        await self.eventually(entered.is_set)
        with self.assertRaises(api.HTTPException) as caught:
            await self.worker.predict(b"second")
        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(caught.exception.headers["Retry-After"], "1")
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(worker=self.worker)))
        response = await api.health(request)
        self.assertEqual(response["status"], "ready")
        self.assertTrue(response["busy"])
        release.set()
        self.assertEqual((await asyncio.wait_for(first, 5))["result"], "ok")
        self.assertFalse(self.worker.busy)

    async def test_cancelled_waiter_does_not_release_running_work(self):
        entered, release = self.gated_operation()
        first = self.start(self.worker.predict(b"first"))
        await self.eventually(entered.is_set)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertTrue(self.worker.busy)
        with self.assertRaises(api.HTTPException) as caught:
            await self.worker.predict(b"second")
        self.assertEqual(caught.exception.status_code, 429)
        release.set()
        await self.eventually(lambda: not self.worker.busy)
        self.assertEqual((await asyncio.wait_for(self.worker.predict(b"third"), 5))["result"], "ok")

    async def test_failure_after_waiter_cancellation_releases_worker(self):
        entered, release = self.gated_operation(fail=True)
        first = self.start(self.worker.predict(b"first"))
        await self.eventually(entered.is_set)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        release.set()
        await self.eventually(lambda: not self.worker.busy)
        self.worker.predict_sync = lambda *args: "recovered"
        self.assertEqual(await asyncio.wait_for(self.worker.predict(b"next"), 5), "recovered")

    async def test_handler_returns_500_for_inference_failure_then_recovers(self):
        def broken(*args):
            raise RuntimeError("Synthetic failure; not a CUDA fault")

        self.worker.predict_sync = broken
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(worker=self.worker)))
        upload = api.UploadFile(filename="test.png", file=BytesIO(b"payload"))
        with self.assertLogs("uvicorn.error", level="ERROR"):
            with self.assertRaises(api.HTTPException) as caught:
                await api.predict(request, upload, False)
        self.assertEqual(caught.exception.status_code, 500)
        self.assertTrue(upload.file.closed)
        self.assertFalse(self.worker.busy)
        self.worker.predict_sync = lambda *args: "recovered"
        self.assertEqual(await asyncio.wait_for(self.worker.predict(b"next"), 5), "recovered")

    async def test_executor_submission_failure_clears_busy(self):
        self.worker.executor.shutdown(wait=True)
        with self.assertRaises(RuntimeError):
            await self.worker.predict(b"unused")
        self.assertFalse(self.worker.busy)

    async def test_lifespan_loads_once_and_cleans_up_on_owner_thread(self):
        threads = []
        app = SimpleNamespace(state=SimpleNamespace())
        loop_thread = threading.get_ident()
        self.worker.predict_sync = lambda *args: threads.append(threading.get_ident())
        with patch.object(api, "ModelWorker", return_value=self.worker), \
             patch.object(self.worker, "load", side_effect=lambda _: threads.append(threading.get_ident())) as load, \
             patch.object(self.worker, "unload", side_effect=lambda: threads.append(threading.get_ident())) as unload, \
             patch.dict(api.os.environ, {"VQC_CONFIG": "test.json"}):
            async with api.lifespan(app):
                self.assertIs(app.state.worker, self.worker)
                await asyncio.wait_for(self.worker.predict(b"one"), 5)
                await asyncio.wait_for(self.worker.predict(b"two"), 5)
            load.assert_called_once_with("test.json")
            unload.assert_called_once()
        self.assertEqual(len(threads), 4)
        self.assertEqual(len(set(threads)), 1)
        self.assertNotEqual(threads[0], loop_thread)
        with self.assertRaises(RuntimeError):
            self.worker.executor.submit(lambda: None)

    async def test_startup_failure_prevents_serving_and_closes_executor(self):
        app = SimpleNamespace(state=SimpleNamespace())
        with patch.object(api, "ModelWorker", return_value=self.worker), \
             patch.object(self.worker, "load", side_effect=RuntimeError("Synthetic load failure")), \
             patch.object(self.worker, "unload") as unload, \
             patch.dict(api.os.environ, {"VQC_CONFIG": "test.json"}):
            with self.assertRaisesRegex(RuntimeError, "Synthetic load failure"):
                async with api.lifespan(app):
                    self.fail("The application served after failed initialization.")
            unload.assert_called_once()
        self.assertFalse(hasattr(app.state, "worker"))
        with self.assertRaises(RuntimeError):
            self.worker.executor.submit(lambda: None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
