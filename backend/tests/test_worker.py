"""모델별 worker pool lifecycle 회귀."""

from __future__ import annotations

import sys
import threading
import types
import zipfile
from collections import OrderedDict, deque
from unittest.mock import Mock

import pytest

from app import config
from app.inference.worker import InferenceWorker, _ModelLane, _model_worker_main


class _FakeProc:
    def __init__(self, alive: bool = False) -> None:
        self._alive = alive
        self.pid = 4321

    def start(self) -> None:
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout=None) -> None:
        return None

    def terminate(self) -> None:
        self._alive = False


class _FakeEvent:
    def __init__(self) -> None:
        self.set_called = False

    def set(self) -> None:
        self.set_called = True

    def is_set(self) -> bool:
        return self.set_called


class _FakeLane:
    def __init__(self, alive: bool = False) -> None:
        self.alive = alive
        self.starts = 0
        self.stops = 0

    def is_alive(self) -> bool:
        return self.alive

    def start(self) -> None:
        self.starts += 1
        self.alive = True

    def stop(self) -> None:
        self.stops += 1
        self.alive = False


def _pool(lanes: dict[str, _FakeLane]) -> InferenceWorker:
    worker = object.__new__(InferenceWorker)
    worker._pool_lock = threading.RLock()
    worker._lanes = lanes
    worker._desired_models = set(lanes)
    worker._running = True
    worker._enabled = True
    worker._aggregate_lock = threading.Lock()
    worker._aggregates = {}
    worker._aggregate_ids_by_source = {}
    worker._ready_results = deque()
    return worker


def test_is_alive_requires_every_desired_model_lane():
    worker = _pool(
        {
            "yolo26n-pose.pt": _FakeLane(alive=True),
            "yolo26x-pose.pt": _FakeLane(alive=False),
        }
    )

    assert worker.is_alive() is False
    worker._lanes["yolo26x-pose.pt"].alive = True
    assert worker.is_alive() is True


def test_start_after_stop_restarts_desired_lanes():
    lane = _FakeLane(alive=True)
    worker = _pool({"yolo26n-pose.pt": lane})

    worker.stop()
    assert worker._running is False
    assert lane.stops == 1

    worker.start()
    assert worker._running is True
    assert lane.starts == 1
    assert worker.is_alive() is True


def test_model_lane_respawn_recreates_queues():
    lane = object.__new__(_ModelLane)
    lane.model_name = "yolo26n-pose.pt"
    lane.device = None
    lane._batch_max = 8
    lane._batch_timeout_sec = 0.008
    lane._min_imgsz = 320
    lane._pending_lock = threading.Lock()
    lane._pending = OrderedDict()
    lane._proc = None
    lane.in_q = object()
    lane.out_q = object()
    lane._stop_event = None

    made_queues: list[object] = []
    made_processes: list[dict] = []

    def _queue(maxsize):
        value = types.SimpleNamespace(maxsize=maxsize)
        made_queues.append(value)
        return value

    def _process(**kwargs):
        made_processes.append(kwargs)
        return _FakeProc()

    lane._ctx = types.SimpleNamespace(
        Queue=_queue,
        Event=_FakeEvent,
        Process=_process,
    )
    old_in, old_out = lane.in_q, lane.out_q

    lane.start()

    assert lane.in_q is not old_in
    assert lane.out_q is not old_out
    assert len(made_queues) == 2
    assert made_processes[0]["args"][1] is lane.in_q
    assert made_processes[0]["args"][2] is lane.out_q
    assert lane.is_alive() is True


def test_configure_models_stops_unused_lane_releasing_vram():
    nano = _FakeLane(alive=True)
    extra = _FakeLane(alive=True)
    worker = _pool({"yolo26n-pose.pt": nano, "yolo26x-pose.pt": extra})

    worker.configure_models({"yolo26x-pose.pt"})

    assert set(worker._lanes) == {"yolo26x-pose.pt"}
    assert nano.stops == 1
    assert extra.stops == 0


@pytest.fixture
def run_model_loader(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "YOLO_WEIGHTS_DIR", tmp_path)
    monkeypatch.setitem(sys.modules, "torch", Mock())
    stopped = threading.Event()
    stopped.set()

    def run(loader, device="cpu"):
        monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=loader))
        _model_worker_main("yolo26n-pose.pt", None, None, stopped, device, 1, 0, 320)

    return run


@pytest.mark.parametrize("download_succeeds", [True, False])
def test_truncated_preset_is_removed_before_one_retry(tmp_path, run_model_loader, caplog, download_succeeds):
    path = tmp_path / "yolo26n-pose.pt"
    path.write_bytes(b"partial")
    attempts = []
    model = Mock()

    def load(filename):
        attempts.append((filename, path.exists()))
        if path.exists():
            raise EOFError("incomplete weights")
        if not download_succeeds:
            path.write_bytes(b"interrupted again")
            raise ConnectionError("download unavailable")
        return model

    run_model_loader(load)

    assert attempts == [(str(path), True), (str(path), False)]
    if download_succeeds:
        model.to.assert_called_once_with("cpu")
    else:
        assert "Model load failed" in caplog.text


@pytest.mark.parametrize("error", ["CUDA out of memory", "invalid device"])
def test_device_errors_keep_cached_weights(tmp_path, run_model_loader, error):
    path = tmp_path / "yolo26n-pose.pt"
    path.write_bytes(b"cached")
    model = Mock()
    model.to.side_effect = [RuntimeError(error), None]
    loader = Mock(return_value=model)

    run_model_loader(loader, device="cuda:0")

    assert path.read_bytes() == b"cached"
    devices = [call.args[0] for call in model.to.call_args_list]
    assert devices == (["cuda:0", "cpu"] if error == "CUDA out of memory" else ["cuda:0"])


def test_oom_while_reading_weights_retries_on_cpu_with_same_file(tmp_path, run_model_loader):
    path = tmp_path / "yolo26n-pose.pt"
    path.write_bytes(b"cached")
    model = Mock()
    loader = Mock(side_effect=[RuntimeError("CUDA out of memory"), model])

    run_model_loader(loader, device="cuda:0")

    assert path.read_bytes() == b"cached"
    model.to.assert_called_once_with("cpu")


def test_complete_preset_is_kept_when_load_fails_for_other_reasons(tmp_path, run_model_loader, caplog):
    # 다 받은 공식 가중치는 ZIP 이다 — 다른 까닭으로 로드가 실패해도 지우고 다시 받지 않는다.
    path = tmp_path / "yolo26n-pose.pt"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("data.pkl", b"weights")
    loader = Mock(side_effect=RuntimeError("unsupported model format"))

    run_model_loader(loader)

    assert path.exists()
    assert loader.call_count == 1
    assert "Model load failed" in caplog.text
