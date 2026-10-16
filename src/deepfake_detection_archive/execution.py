"""Bounded FP32 execution planning; sampling and file-level aggregation stay fixed."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch

from .tracking import atomic_json


RESERVE_BYTES = 256 * 1024 * 1024
RESERVE_FRACTION = 0.15
PROFILE_ATTEMPTS = 3
PROFILE_SECONDS = 10.0
OOM_REDUCTIONS = 3


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return None


def _cgroup_dirs(proc: Path) -> Iterator[tuple[str, Path]]:
    """Resolve this process's hierarchy, including inherited parent limits."""
    memberships = (_read(proc / "cgroup") or "").splitlines()
    mounts = (_read(proc / "mountinfo") or "").splitlines()
    for row in memberships:
        _, controllers, membership = row.split(":", 2)
        for mount in mounts:
            before, after = mount.split(" - ", 1)
            fields, filesystem = before.split(), after.split()
            if filesystem[0] not in {"cgroup", "cgroup2"}:
                continue
            if controllers and not set(controllers.split(",")) & set(
                filesystem[2].split(",")
            ):
                continue
            if not controllers and filesystem[0] != "cgroup2":
                continue
            root = Path(fields[3])
            member = Path(membership)
            if not member.is_relative_to(root):
                continue
            mountpoint = Path(fields[4])
            current = mountpoint / member.relative_to(root)
            while True:
                yield controllers, current
                if current == mountpoint:
                    break
                current = current.parent


def cpu_resources(proc: Path = Path("/proc/self")) -> dict[str, Any]:
    affinity = len(os.sched_getaffinity(0))
    quotas: list[float] = []
    limits: list[int] = []
    for controllers, directory in _cgroup_dirs(proc):
        if not controllers:
            cpu = _read(directory / "cpu.max")
            if cpu is not None:
                quota, period = cpu.split()
                if quota != "max":
                    quotas.append(int(quota) / int(period))
            memory = _read(directory / "memory.max")
        else:
            quota_text = _read(directory / "cpu.cfs_quota_us")
            period_text = _read(directory / "cpu.cfs_period_us")
            if (
                quota_text is not None
                and period_text is not None
                and int(quota_text) > 0
            ):
                quotas.append(int(quota_text) / int(period_text))
            memory = _read(directory / "memory.limit_in_bytes")
        if memory is not None and memory != "max" and int(memory) < 2**60:
            limits.append(int(memory))
    quota_cpus = min(quotas) if quotas else None
    memory_info = {
        row.split(":", 1)[0]: int(row.split()[1]) * 1024
        for row in (_read(proc.parent / "meminfo") or "").splitlines()
        if row.startswith(("MemTotal:", "MemAvailable:"))
    }
    return {
        "affinity_cpus": affinity,
        "quota_cpus": quota_cpus,
        "threads": max(1, math.floor(min(affinity, quota_cpus or affinity))),
        "memory_limit": min(limits) if limits else None,
        "memory_total": memory_info.get("MemTotal"),
        "memory_available": memory_info.get("MemAvailable"),
        "workers": 1,
    }


def execution_signature(device: str | torch.device, mode: str) -> dict[str, Any]:
    """Stable resume identity; changing free memory never changes the identity."""
    device = torch.device(device)
    thread_budget = cpu_resources()["threads"]
    signature: dict[str, Any] = {
        "schema_version": 1,
        "mode": mode,
        "device": str(device),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "precision": "fp32",
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "cpu_thread_budget": thread_budget,
        "cpu_threads": (
            torch.get_num_threads()
            if mode == "reference"
            else min(torch.get_num_threads(), thread_budget)
        ),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        signature.update(
            visible_index=(
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            ),
            device_name=properties.name,
            compute_capability=[properties.major, properties.minor],
            total_memory=properties.total_memory,
            cudnn=torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
        )
    return signature


def resource_snapshot(device: torch.device) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"cpu": cpu_resources()}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        snapshot["cuda"] = {
            "free": free,
            "total": total,
            "model_allocated": torch.cuda.memory_allocated(device),
            "model_reserved": torch.cuda.memory_reserved(device),
            "load_peak_allocated": torch.cuda.max_memory_allocated(device),
            "load_peak_reserved": torch.cuda.max_memory_reserved(device),
            "reserve": max(RESERVE_BYTES, math.ceil(free * RESERVE_FRACTION)),
        }
    return snapshot


class ExecutionPolicy:
    """Profile whole uncommitted files; steady state scores each file once."""

    def __init__(  # noqa: PLR0913
        self,
        device: torch.device,
        mode: str,
        batch_size: int,
        units: int,
        output_dir: Path,
        *,
        resume: bool,
    ) -> None:
        self.device = device
        self.mode = mode
        self.requested_batch = batch_size
        self.batch_size = batch_size if mode == "reference" else min(batch_size, units)
        self.path = output_dir / "execution.json"
        self.signature = execution_signature(device, mode)
        self.resources = resource_snapshot(device)
        self.previous_threads = torch.get_num_threads()
        self.profiled = mode == "reference" or device.type != "cuda"
        self.profile: list[dict[str, Any]] = []
        self.reductions: list[dict[str, Any]] = []
        self.successful_files = 0
        self.formal_seconds = 0.0
        self.peak_allocated = 0
        self.peak_reserved = 0
        if resume and self.path.exists():
            saved = json.loads(self.path.read_text())
            if (
                saved["signature"] != self.signature
                or saved["requested_batch"] != self.requested_batch
            ):
                raise ValueError("execution plan identity differs; use a new output")
            self.batch_size = saved["effective_batch"]
            self.profiled = saved["profiled"]
            self.profile = saved["profile"]
            self.reductions = saved["reductions"]
            self.successful_files = saved["successful_files"]
            self.formal_seconds = saved["formal_seconds"]
            self.peak_allocated = saved["peak_allocated"]
            self.peak_reserved = saved["peak_reserved"]
        if mode == "adaptive":
            torch.set_num_threads(
                min(self.previous_threads, self.resources["cpu"]["threads"])
            )
        self.started = False
        self.resources_end: dict[str, Any] | None = None

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "signature": self.signature,
            "resources": self.resources,
            "resources_end": self.resources_end,
            "requested_batch": self.requested_batch,
            "effective_batch": self.batch_size,
            "cpu_threads": torch.get_num_threads(),
            "precision": "fp32",
            "profiled": self.profiled,
            "profile": self.profile,
            "profile_seconds": sum(item["seconds"] for item in self.profile),
            "profile_bounds": {
                "attempts": PROFILE_ATTEMPTS,
                "seconds_before_next_attempt": PROFILE_SECONDS,
                "single_call_interruptible": False,
            },
            "oom_max_reductions": OOM_REDUCTIONS,
            "reductions": self.reductions,
            "successful_files": self.successful_files,
            "formal_seconds": self.formal_seconds,
            "peak_allocated": self.peak_allocated,
            "peak_reserved": self.peak_reserved,
        }
        atomic_json(self.path, payload)

    def _peaks(self) -> None:
        if self.device.type == "cuda":
            self.peak_allocated = max(
                self.peak_allocated, torch.cuda.max_memory_allocated(self.device)
            )
            self.peak_reserved = max(
                self.peak_reserved, torch.cuda.max_memory_reserved(self.device)
            )

    def _profile(self, score: Callable[[int], float]) -> None:
        started = time.monotonic()
        limit = self.batch_size
        candidate = 1
        chosen = 1
        for attempt in range(PROFILE_ATTEMPTS):
            if time.monotonic() - started >= PROFILE_SECONDS:
                break
            free, _ = torch.cuda.mem_get_info(self.device)
            margin = max(RESERVE_BYTES, math.ceil(free * RESERVE_FRACTION))
            resident = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            before = time.monotonic()
            oom = False
            oom_message: str | None = None
            try:
                score(candidate)
                torch.cuda.synchronize(self.device)
            except torch.cuda.OutOfMemoryError as error:
                oom_message = str(error)
                error.__traceback__ = None
                oom = True
            seconds = time.monotonic() - before
            peak = torch.cuda.max_memory_allocated(self.device)
            self._peaks()
            self.profile.append(
                {
                    "batch": candidate,
                    "seconds": seconds,
                    "peak_allocated": peak,
                    "free_before": free,
                    "reserve": margin,
                    "oom": oom,
                    "oom_message": oom_message,
                    "fits_reserve": (
                        not oom and max(0, peak - resident) <= max(0, free - margin)
                    ),
                }
            )
            if oom:
                torch.cuda.empty_cache()
                self.reductions.append(
                    {
                        "phase": "profile",
                        "oom_message": oom_message,
                        "old_batch": candidate,
                        "new_batch": chosen,
                        "attempt": attempt + 1,
                    }
                )
                self._write()
                if candidate == 1:
                    raise torch.cuda.OutOfMemoryError(
                        f"profile batch 1 cannot fit: {oom_message}"
                    )
                break
            if max(0, peak - resident) > max(0, free - margin):
                if candidate == 1:
                    raise MemoryError(
                        "profile batch 1 leaves insufficient CUDA reserve"
                    )
                break
            chosen = candidate
            if candidate == limit:
                break
            cost = max(1, peak - resident)
            estimated = max(1, int((free - margin) / cost * candidate))
            next_batch = min(limit, estimated, candidate * 2 if attempt == 0 else limit)
            if next_batch <= candidate:
                break
            candidate = next_batch
        self.batch_size = chosen
        self.profiled = True
        self._write()

    def score(self, call: Callable[[int], float]) -> float:
        if not self.started:
            self.started = True
            self._write()
        if not self.profiled:
            self._profile(call)
        started = time.monotonic()
        for attempt in range(OOM_REDUCTIONS + 1):
            oom = False
            oom_message: str | None = None
            try:
                result = call(self.batch_size)
            except torch.cuda.OutOfMemoryError as error:
                oom_message = str(error)
                if (
                    self.mode == "reference"
                    or self.device.type != "cuda"
                    or self.batch_size == 1
                    or attempt == OOM_REDUCTIONS
                ):
                    self._peaks()
                    self._write()
                    raise
                error.__traceback__ = None
                oom = True
            if not oom:
                self.successful_files += 1
                self.formal_seconds += time.monotonic() - started
                self._peaks()
                return result
            old = self.batch_size
            self.batch_size = max(1, old // 2)
            torch.cuda.empty_cache()
            self.reductions.append(
                {
                    "phase": "inference",
                    "oom_message": oom_message,
                    "old_batch": old,
                    "new_batch": self.batch_size,
                    "attempt": attempt + 1,
                }
            )
            self._write()
        raise AssertionError("bounded execution loop exhausted")

    def finish(self) -> None:
        if self.started:
            self.resources_end = resource_snapshot(self.device)
            self._peaks()
            self._write()
        if self.mode == "adaptive":
            torch.set_num_threads(self.previous_threads)
