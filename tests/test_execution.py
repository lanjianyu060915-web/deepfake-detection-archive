"""Execution changes batch cost only and retries uncommitted CUDA OOM work."""

import json
from unittest.mock import Mock

import pytest
import torch

from deepfake_detection_archive.execution import (
    ExecutionPolicy,
    cpu_resources,
    execution_signature,
)


def fake_policy(monkeypatch, tmp_path, mode="adaptive", batch=8):
    monkeypatch.setattr(
        "deepfake_detection_archive.execution.execution_signature",
        lambda *_: {"device": "cuda:0", "mode": mode},
    )
    monkeypatch.setattr(
        "deepfake_detection_archive.execution.resource_snapshot",
        lambda *_: {"cpu": {"threads": 1}, "cuda": {"free": 10**10}},
    )
    monkeypatch.setattr(torch, "set_num_threads", Mock())
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", Mock())
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_: 200)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: 50)
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (10**10, 2 * 10**10))
    policy = ExecutionPolicy(
        torch.device("cuda"), mode, batch, max(8, batch), tmp_path, resume=False
    )
    policy.profiled = True
    return policy


def test_reference_never_profiles_or_retries(monkeypatch, tmp_path):
    policy = fake_policy(monkeypatch, tmp_path, mode="reference")
    call = Mock(side_effect=torch.cuda.OutOfMemoryError("actual CUDA OOM"))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        policy.score(call)
    assert call.call_count == 1
    assert policy.batch_size == 8


def test_adaptive_halves_only_uncommitted_work(monkeypatch, tmp_path):
    policy = fake_policy(monkeypatch, tmp_path)
    calls = []

    def score(batch):
        calls.append(batch)
        if batch > 2:
            raise torch.cuda.OutOfMemoryError("injected OOM")
        return 0.25

    assert policy.score(score) == 0.25
    assert calls == [8, 4, 2]
    assert policy.score(score) == 0.25
    assert calls == [8, 4, 2, 2]
    policy.finish()
    saved = json.loads((tmp_path / "execution.json").read_text())
    assert saved["successful_files"] == 2
    assert len(saved["reductions"]) == 2


@pytest.mark.parametrize("error", [RuntimeError("bad kernel"), ValueError("bad input")])
def test_other_errors_are_not_retried(monkeypatch, tmp_path, error):
    policy = fake_policy(monkeypatch, tmp_path)
    call = Mock(side_effect=error)
    with pytest.raises(type(error), match=str(error)):
        policy.score(call)
    assert call.call_count == 1


def test_minimum_batch_failure_aborts(monkeypatch, tmp_path):
    policy = fake_policy(monkeypatch, tmp_path, batch=2)
    call = Mock(side_effect=torch.cuda.OutOfMemoryError("injected OOM"))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        policy.score(call)
    assert [args.args[0] for args in call.call_args_list] == [2, 1]


def test_cgroup_v2_nested_limits_and_affinity(monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    group = tmp_path / "cgroup"
    proc.mkdir()
    (group / "job").mkdir(parents=True)
    (proc / "cgroup").write_text("0::/job\n")
    (proc / "mountinfo").write_text(f"1 0 0:1 / {group} rw - cgroup2 cgroup rw\n")
    (group / "cpu.max").write_text("150000 100000")
    (group / "memory.max").write_text("1073741824")
    (group / "job/cpu.max").write_text("max 100000")
    (group / "job/memory.max").write_text("max")
    monkeypatch.setattr("os.sched_getaffinity", lambda _: {0, 1, 2, 3})
    result = cpu_resources(proc)
    assert result["threads"] == 1
    assert result["quota_cpus"] == 1.5
    assert result["memory_limit"] == 1073741824


def test_cgroup_v1_limits(monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    cpu = tmp_path / "cpu"
    memory = tmp_path / "memory"
    proc.mkdir()
    for path in (cpu, memory):
        (path / "job").mkdir(parents=True)
    (proc / "cgroup").write_text("2:cpu,cpuacct:/job\n3:memory:/job\n")
    (proc / "mountinfo").write_text(
        f"1 0 0:1 / {cpu} rw - cgroup cgroup rw,cpu,cpuacct\n"
        f"2 0 0:2 / {memory} rw - cgroup cgroup rw,memory\n"
    )
    (cpu / "job/cpu.cfs_quota_us").write_text("200000")
    (cpu / "job/cpu.cfs_period_us").write_text("100000")
    (memory / "job/memory.limit_in_bytes").write_text("536870912")
    monkeypatch.setattr("os.sched_getaffinity", lambda _: {0, 1, 2, 3})
    result = cpu_resources(proc)
    assert result["threads"] == 2
    assert result["memory_limit"] == 536870912


def test_profile_is_bounded_and_does_not_return_warmup_score(monkeypatch, tmp_path):
    policy = fake_policy(monkeypatch, tmp_path)
    policy.profiled = False
    calls = []

    def score(batch):
        calls.append(batch)
        return len(calls) / 10

    assert policy.score(score) == 0.4
    assert calls == [1, 2, 8, 8]
    assert len(policy.profile) == 3
    assert policy.successful_files == 1
    assert policy.score(score) == 0.5
    assert calls[-1] == 8
    assert len(calls) == 5


def test_profile_deadline_is_checked_before_next_call(monkeypatch, tmp_path):
    policy = fake_policy(monkeypatch, tmp_path)
    policy.profiled = False
    clock = [0.0]
    monkeypatch.setattr(
        "deepfake_detection_archive.execution.time.monotonic", lambda: clock[0]
    )
    batches = []

    def score(batch):
        batches.append(batch)
        clock[0] += 11
        return 0.25

    assert policy.score(score) == 0.25
    assert batches == [1, 1]
    assert len(policy.profile) == 1


def test_resume_reuses_reduced_plan_without_reprofiling(monkeypatch, tmp_path):
    policy = fake_policy(monkeypatch, tmp_path)
    policy.batch_size = 2
    policy.score(lambda _: 0.25)
    policy.finish()
    prefix = (tmp_path / "execution.json").read_bytes()
    resumed = ExecutionPolicy(
        torch.device("cuda"), "adaptive", 8, 8, tmp_path, resume=True
    )
    assert resumed.batch_size == 2
    assert resumed.profiled
    resumed.finish()
    assert (tmp_path / "execution.json").read_bytes() == prefix
    call = Mock(return_value=0.5)
    assert resumed.score(call) == 0.5
    call.assert_called_once_with(2)


def test_constructor_and_finish_never_overwrite_rejected_resume(monkeypatch, tmp_path):
    fake_policy(monkeypatch, tmp_path).finish()
    assert not (tmp_path / "execution.json").exists()
    policy = fake_policy(monkeypatch, tmp_path)
    policy.score(lambda _: 0.25)
    prefix = (tmp_path / "execution.json").read_bytes()
    with pytest.raises(ValueError, match="identity differs"):
        ExecutionPolicy(torch.device("cuda"), "adaptive", 4, 8, tmp_path, resume=True)
    assert (tmp_path / "execution.json").read_bytes() == prefix


def test_profile_oom_uses_last_fitting_batch(monkeypatch, tmp_path):
    policy = fake_policy(monkeypatch, tmp_path)
    policy.profiled = False
    batches = []

    def score(batch):
        batches.append(batch)
        if batch > 1:
            raise torch.cuda.OutOfMemoryError("injected profile OOM")
        return 0.25

    assert policy.score(score) == 0.25
    assert batches == [1, 2, 1]
    assert policy.batch_size == 1
    assert len(policy.reductions) == 1


def test_oom_retry_count_is_bounded(monkeypatch, tmp_path):
    policy = fake_policy(monkeypatch, tmp_path, batch=64)
    call = Mock(side_effect=torch.cuda.OutOfMemoryError("injected OOM"))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        policy.score(call)
    assert [args.args[0] for args in call.call_args_list] == [64, 32, 16, 8]
    assert len(policy.reductions) == 3


@pytest.mark.parametrize("phase", ["profile", "inference"])
def test_oom_allocator_diagnostic_survives_retry_and_minimum_failure(
    monkeypatch, tmp_path, phase
):
    policy = fake_policy(monkeypatch, tmp_path, batch=2)
    policy.profiled = phase != "profile"
    message = "CUDA out of memory. Tried to allocate 512.00 MiB (GPU 0)."

    def score(_batch):
        raise torch.cuda.OutOfMemoryError(message)

    with pytest.raises(torch.cuda.OutOfMemoryError, match=r"512\.00 MiB"):
        policy.score(score)
    saved = json.loads((tmp_path / "execution.json").read_text())
    assert saved["reductions"][0]["oom_message"] == message
    if phase == "profile":
        assert saved["profile"][0]["oom_message"] == message


def test_tf32_driver_override_changes_resume_identity(monkeypatch):
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    disabled = execution_signature("cpu", "reference")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "1")
    enabled = execution_signature("cpu", "reference")
    assert disabled["nvidia_tf32_override"] == "0"
    assert enabled["nvidia_tf32_override"] == "1"
    assert disabled != enabled
