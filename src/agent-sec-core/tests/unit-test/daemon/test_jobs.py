"""Tests for daemon background job scheduling."""

import asyncio
import contextlib
import sys
import threading

import pytest
from agent_sec_cli.daemon.jobs import (
    JobManager,
    JobStatus,
    PeriodicBackgroundJob,
)
from agent_sec_cli.daemon.jobs.base import next_cycle_start
from agent_sec_cli.daemon.jobs.prompt_preload import (
    _PROMPT_PRELOAD_CHILD_MODULE,
    PromptModelPreloadJob,
    _download_prompt_model_sync,
    _preload_prompt_model_sync,
    _run_preload_child_process,
)
from agent_sec_cli.daemon.jobs.registry import register_default_jobs
from agent_sec_cli.daemon.runtime import PromptScanRuntimeState


class RecordingPeriodicJob(PeriodicBackgroundJob):
    """Periodic job used by scheduling tests."""

    name = "recording-periodic-job"

    def __init__(self, interval_seconds: float) -> None:
        super().__init__(interval_seconds=interval_seconds)
        self.run_count = 0
        self.started = asyncio.Event()

    async def run_once(self) -> None:
        """Record one scheduled run."""
        self.run_count += 1
        self.started.set()


def test_next_cycle_start_uses_start_time_interval_boundaries():
    assert next_cycle_start(100.0, 103.0, 10.0) == 110.0
    assert next_cycle_start(100.0, 110.0, 10.0) == 110.0


def test_next_cycle_start_skips_missed_interval_boundaries():
    assert next_cycle_start(100.0, 112.0, 10.0) == 120.0
    assert next_cycle_start(100.0, 125.0, 10.0) == 130.0


def test_next_cycle_start_rejects_invalid_interval():
    with pytest.raises(ValueError, match="interval_seconds must be positive"):
        next_cycle_start(100.0, 101.0, 0.0)


def test_job_status_omits_unset_optional_periodic_fields():
    status = JobStatus(name="job", state="stopped")

    assert status.to_dict() == {
        "name": "job",
        "state": "stopped",
        "last_error": None,
        "last_tick_at": None,
    }


def test_periodic_background_job_runs_and_reports_interval():
    async def scenario():
        job = RecordingPeriodicJob(interval_seconds=3600.0)
        await job.start()
        try:
            await asyncio.wait_for(job.started.wait(), timeout=0.5)
            status = job.status().to_dict()
            run_count = job.run_count
        finally:
            await job.stop()
        return status, run_count

    status, run_count = asyncio.run(scenario())

    assert run_count == 1
    assert status["name"] == "recording-periodic-job"
    assert status["state"] == "running"
    assert status["interval_seconds"] == 3600.0
    assert "last_started_at" in status
    assert "next_run_at" in status


def test_register_default_jobs_respects_prompt_preload_env(monkeypatch):
    prompt_state = PromptScanRuntimeState()

    disabled_manager = JobManager()
    monkeypatch.setenv("AGENT_SEC_DAEMON_PROMPT_PRELOAD", "0")
    register_default_jobs(disabled_manager, prompt_state)

    enabled_manager = JobManager()
    monkeypatch.setenv("AGENT_SEC_DAEMON_PROMPT_PRELOAD", "1")
    register_default_jobs(enabled_manager, prompt_state)

    assert [job["name"] for job in disabled_manager.status()] == [
        "skill-ledger-activation"
    ]
    assert [job["name"] for job in enabled_manager.status()] == [
        "skill-ledger-activation",
        "prompt-model-preload",
    ]


def test_prompt_model_preload_job_updates_runtime_state(monkeypatch):
    prompt_state = PromptScanRuntimeState()
    child_calls: list[str] = []
    calls: list[tuple[str, str]] = []

    async def fake_child_preload(mode: str) -> None:
        child_calls.append(mode)
        assert prompt_state.status == "downloading"

    def fake_preload(state, mode: str, probe_text: str) -> None:
        calls.append((mode, probe_text))
        assert state.status == "loading"
        state.model = "fake-model"
        state.status = "loading"

    monkeypatch.setattr(
        "agent_sec_cli.daemon.jobs.prompt_preload._run_preload_child_process",
        fake_child_preload,
    )
    monkeypatch.setattr(
        "agent_sec_cli.daemon.jobs.prompt_preload._preload_prompt_model_sync",
        fake_preload,
    )

    async def scenario():
        job = PromptModelPreloadJob(prompt_state, probe_text="probe")
        await job.start()
        status = await _wait_for_job_state(job, {"completed", "error"})
        await job.stop()
        return status

    status = asyncio.run(scenario())

    assert child_calls == ["strict"]
    assert calls == [("strict", "probe")]
    assert status["state"] == "completed"
    assert status["last_error"] is None
    assert prompt_state.status == "ready"
    assert prompt_state.model == "fake-model"
    assert prompt_state.loaded is True
    assert prompt_state.last_error is None
    assert prompt_state.last_started_at is not None
    assert prompt_state.last_finished_at is not None


def test_prompt_model_preload_job_marks_prompt_degraded_on_failure(monkeypatch):
    prompt_state = PromptScanRuntimeState()

    async def fake_child_preload(_mode: str) -> None:
        pass

    def fake_preload(_state, _mode: str, _probe_text: str) -> None:
        raise RuntimeError("forced preload failure")

    monkeypatch.setattr(
        "agent_sec_cli.daemon.jobs.prompt_preload._run_preload_child_process",
        fake_child_preload,
    )
    monkeypatch.setattr(
        "agent_sec_cli.daemon.jobs.prompt_preload._preload_prompt_model_sync",
        fake_preload,
    )

    async def scenario():
        job = PromptModelPreloadJob(prompt_state)
        await job.start()
        status = await _wait_for_job_state(job, {"completed", "error"})
        await job.stop()
        return status

    status = asyncio.run(scenario())

    assert status["state"] == "error"
    assert status["last_error"] == "forced preload failure"
    assert prompt_state.status == "degraded"
    assert prompt_state.loaded is False
    assert prompt_state.last_error == "forced preload failure"
    assert prompt_state.last_finished_at is not None


def test_prompt_model_preload_job_marks_prompt_degraded_on_child_failure(monkeypatch):
    prompt_state = PromptScanRuntimeState()

    async def fake_child_preload(_mode: str) -> None:
        raise RuntimeError("forced child failure")

    def fake_preload(_state, _mode: str, _probe_text: str) -> None:
        raise AssertionError("main preload should not run after child failure")

    monkeypatch.setattr(
        "agent_sec_cli.daemon.jobs.prompt_preload._run_preload_child_process",
        fake_child_preload,
    )
    monkeypatch.setattr(
        "agent_sec_cli.daemon.jobs.prompt_preload._preload_prompt_model_sync",
        fake_preload,
    )

    async def scenario():
        job = PromptModelPreloadJob(prompt_state)
        await job.start()
        status = await _wait_for_job_state(job, {"completed", "error"})
        await job.stop()
        return status

    status = asyncio.run(scenario())

    assert status["state"] == "error"
    assert status["last_error"] == "forced child failure"
    assert prompt_state.status == "degraded"
    assert prompt_state.loaded is False
    assert prompt_state.last_error == "forced child failure"
    assert prompt_state.last_finished_at is not None


def test_prompt_model_preload_job_cancel_during_child_preload(monkeypatch):
    prompt_state = PromptScanRuntimeState()
    child_started = asyncio.Event()
    child_cancelled = False

    async def fake_child_preload(_mode: str) -> None:
        nonlocal child_cancelled
        child_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            child_cancelled = True
            raise

    def fake_preload(_state, _mode: str, _probe_text: str) -> None:
        raise AssertionError("main preload should not run after cancellation")

    monkeypatch.setattr(
        "agent_sec_cli.daemon.jobs.prompt_preload._run_preload_child_process",
        fake_child_preload,
    )
    monkeypatch.setattr(
        "agent_sec_cli.daemon.jobs.prompt_preload._preload_prompt_model_sync",
        fake_preload,
    )

    async def scenario():
        job = PromptModelPreloadJob(prompt_state)
        await job.start()
        await asyncio.wait_for(child_started.wait(), timeout=0.5)
        await job.stop()
        return job.status().to_dict()

    status = asyncio.run(scenario())

    assert child_cancelled is True
    assert status["state"] == "stopped"
    assert prompt_state.status == "stopped"
    assert prompt_state.loaded is False
    assert prompt_state.last_error is None
    assert prompt_state.last_finished_at is not None


def test_prompt_model_preload_job_cancel_during_main_preload(monkeypatch):
    prompt_state = PromptScanRuntimeState()
    preload_started = threading.Event()
    preload_finished = threading.Event()
    release_preload = threading.Event()

    async def fake_child_preload(_mode: str) -> None:
        pass

    def fake_preload(state, _mode: str, _probe_text: str) -> None:
        state.status = "loading"
        preload_started.set()
        try:
            release_preload.wait(timeout=1.0)
        finally:
            preload_finished.set()

    monkeypatch.setattr(
        "agent_sec_cli.daemon.jobs.prompt_preload._run_preload_child_process",
        fake_child_preload,
    )
    monkeypatch.setattr(
        "agent_sec_cli.daemon.jobs.prompt_preload._preload_prompt_model_sync",
        fake_preload,
    )

    async def scenario():
        job = PromptModelPreloadJob(prompt_state)
        await job.start()
        for _attempt in range(50):
            if preload_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert preload_started.is_set()

        await job.stop()
        prompt_snapshot = prompt_state.to_dict()
        release_preload.set()
        for _attempt in range(50):
            if preload_finished.is_set():
                break
            await asyncio.sleep(0.01)
        assert preload_finished.is_set()
        return job.status().to_dict(), prompt_snapshot

    status, prompt_snapshot = asyncio.run(scenario())

    assert status["state"] == "stopped"
    assert prompt_snapshot["status"] == "stopped"
    assert prompt_snapshot["loaded"] is False
    assert prompt_snapshot["last_error"] is None
    assert prompt_snapshot["last_finished_at"] is not None


def test_prompt_preload_child_process_is_terminated_on_cancel(monkeypatch):
    process_started = asyncio.Event()
    subprocess_args = []

    class FakeProcess:
        def __init__(self) -> None:
            self.returncode = None
            self.terminated = False
            self.killed = False

        async def communicate(self):
            process_started.set()
            await asyncio.Event().wait()
            return b"", b""

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        async def wait(self) -> int:
            return self.returncode

    fake_process = FakeProcess()

    async def fake_create_subprocess_exec(*args, **_kwargs):
        subprocess_args.append(args)
        return fake_process

    monkeypatch.setattr(
        "agent_sec_cli.daemon.jobs.prompt_preload.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    async def scenario():
        task = asyncio.create_task(_run_preload_child_process("strict"))
        await asyncio.wait_for(process_started.wait(), timeout=0.5)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert fake_process.terminated is True
    assert fake_process.killed is False
    assert subprocess_args == [
        (sys.executable, "-m", _PROMPT_PRELOAD_CHILD_MODULE, "strict")
    ]


def test_prompt_preload_child_process_is_terminated_on_timeout(monkeypatch):
    process_started = asyncio.Event()
    subprocess_args = []

    class FakeProcess:
        def __init__(self) -> None:
            self.returncode = None
            self.terminated = False
            self.killed = False

        async def communicate(self):
            process_started.set()
            await asyncio.Event().wait()
            return b"", b""

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        async def wait(self) -> int:
            return self.returncode

    fake_process = FakeProcess()

    async def fake_create_subprocess_exec(*args, **_kwargs):
        subprocess_args.append(args)
        return fake_process

    monkeypatch.setenv(
        "AGENT_SEC_DAEMON_PROMPT_PRELOAD_DOWNLOAD_TIMEOUT_SECONDS",
        "0.01",
    )
    monkeypatch.setattr(
        "agent_sec_cli.daemon.jobs.prompt_preload.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    async def scenario():
        with pytest.raises(RuntimeError, match="timed out after 0.01s"):
            await _run_preload_child_process("strict")

    asyncio.run(scenario())

    assert process_started.is_set()
    assert fake_process.terminated is True
    assert fake_process.killed is False
    assert subprocess_args == [
        (sys.executable, "-m", _PROMPT_PRELOAD_CHILD_MODULE, "strict")
    ]


def test_prompt_model_download_sync_suppresses_warmup_output(monkeypatch, capsys):
    calls = []

    class FakePromptScanner:
        def __init__(self, mode):
            calls.append(("init", mode.value))

        def warmup(self):
            print("download progress on stdout")
            print("download progress on stderr", file=sys.stderr)
            calls.append(("warmup",))

        def scan(self, text, source=None):
            raise AssertionError("download-only child preload should not scan")

    monkeypatch.setattr(
        "agent_sec_cli.prompt_scanner.scanner.PromptScanner",
        FakePromptScanner,
    )

    _download_prompt_model_sync("strict")
    captured = capsys.readouterr()

    assert calls == [
        ("init", "strict"),
        ("warmup",),
    ]
    assert captured.out == ""
    assert captured.err == ""


def test_prompt_model_preload_sync_does_not_redirect_daemon_stdio(monkeypatch, capsys):
    prompt_state = PromptScanRuntimeState()
    calls = []
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    class FakePromptScanner:
        def __init__(self, mode):
            calls.append(("init", mode.value))

        def warmup(self):
            raise AssertionError("daemon preload should not run download warmup")

        def scan(self, text, source=None):
            assert sys.stdout is original_stdout
            assert sys.stderr is original_stderr
            print("daemon stdout remains visible")
            print("daemon stderr remains visible", file=sys.stderr)
            calls.append(("scan", text, source))

    monkeypatch.setattr(
        "agent_sec_cli.prompt_scanner.scanner.PromptScanner",
        FakePromptScanner,
    )

    _preload_prompt_model_sync(prompt_state, "strict", "probe")
    captured = capsys.readouterr()

    assert calls == [
        ("init", "strict"),
        ("scan", "probe", "daemon-startup"),
    ]
    assert captured.out == "daemon stdout remains visible\n"
    assert captured.err == "daemon stderr remains visible\n"
    assert prompt_state.model == "LLM-Research/Llama-Prompt-Guard-2-86M"
    assert prompt_state.status == "loading"


async def _wait_for_job_state(
    job: PromptModelPreloadJob,
    target_states: set[str],
) -> dict:
    for _attempt in range(50):
        status = job.status().to_dict()
        if status["state"] in target_states:
            return status
        await asyncio.sleep(0.01)
    return job.status().to_dict()
