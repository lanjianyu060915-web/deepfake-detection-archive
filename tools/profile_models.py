"""CUDA engineering profile using random weights and synthetic inputs only.

No downloaded data, pretrained weights, detection scores, or EER are produced.
Successful synthetic optimization steps demonstrate execution, not model quality.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import timm
import torch
from torch import Tensor, nn

from deepfake_detection_archive.models import (
    AASISTLite,
    LogMelCNN,
    MediaFeatureClassifier,
    MediaFeatureConfig,
)


GIB = 1024**3
ARCHITECTURES = ("aasist_l_v1", "logmel_cnn_v1", "media_feature_head_v1", "resnet18")


def device_snapshot() -> dict[str, object]:
    """Read optional device-level telemetry without requiring NVML Python bindings."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi not found"}
    result = subprocess.run(  # noqa: S603 -- fixed read-only query
        [
            executable,
            "--query-gpu=name,memory.used,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode:
        return {"available": False, "reason": result.stderr.strip()[:300]}
    return {
        "available": True,
        "csv_name_used_mib_temperature_c_utilization": result.stdout.strip(),
    }


def construct(architecture: str) -> tuple[nn.Module, tuple[int, ...], bool]:
    if architecture == "aasist_l_v1":
        return AASISTLite(), (64600,), True
    if architecture == "logmel_cnn_v1":
        return LogMelCNN(), (64600,), True
    if architecture == "media_feature_head_v1":
        return MediaFeatureClassifier(MediaFeatureConfig(512, 64, 0.1)), (8, 512), True
    model = timm.create_model("resnet18", pretrained=False, num_classes=0)
    model.requires_grad_(False)
    return model, (3, 224, 224), False


def finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"non-finite {name}")


def profile_case(  # noqa: PLR0915 -- ordered measurement and cleanup instrumentation
    architecture: str,
    batch_size: int,
    *,
    deadline: float,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, Any]:
    """Measure a fresh model/input batch, including two verified optimizer steps."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    case_start = time.perf_counter()
    model, input_shape, training = construct(architecture)
    model = model.cuda().train(training)
    inputs = torch.randn((batch_size, *input_shape), device="cuda")
    targets = (torch.arange(batch_size, device="cuda") % 2).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) if training else None
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    measured_steps = 0
    measured_seconds = 0.0
    optimizer_steps = 0
    gradient_count = 0
    parameter_updated = False
    output_shape: list[int] = []
    for step in range(3):
        if time.perf_counter() > deadline or time.perf_counter() - case_start > 60:
            break
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=dtype):
            outputs = cast(Tensor, model(inputs))
            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
                if training
                else outputs.float().square().mean()
            )
        finite(outputs, "outputs")
        finite(loss, "loss")
        output_shape = list(outputs.shape)
        if optimizer is not None:
            scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            scaler.unscale_(optimizer)
            gradient_count = 0
            monitored: nn.Parameter | None = None
            for parameter in model.parameters():
                if parameter.grad is not None:
                    finite(parameter.grad, "gradients")
                    gradient_count += 1
                    if monitored is None and bool((parameter.grad != 0).any()):
                        monitored = parameter
            if monitored is None:
                raise ValueError("no nonzero gradient found")
            before = monitored.detach().clone()
            scaler.step(optimizer)
            scaler.update()
            if not bool(torch.equal(before, monitored)):
                parameter_updated = True
                optimizer_steps += 1
            else:
                raise ValueError(
                    "optimizer failed to change a nonzero-gradient parameter"
                )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - step_start
        if step > 0:  # First step warms kernels and optimizer state.
            measured_steps += 1
            measured_seconds += elapsed
    result: dict[str, Any] = {
        "architecture": architecture,
        "batch_size": batch_size,
        "input_shape": [batch_size, *input_shape],
        "output_shape": output_shape,
        "parameters": parameter_count,
        "training": training,
        "precision": str(dtype),
        "seed": seed,
        "status": "passed" if measured_steps == 2 else "incomplete_time_budget",
        "measured_steps": measured_steps,
        "warmup_steps": int(bool(output_shape)),
        "optimizer_steps_including_warmup": optimizer_steps,
        "parameter_update_verified": parameter_updated if training else None,
        "finite_outputs": bool(output_shape),
        "finite_gradient_tensors": gradient_count if training else None,
        "measured_seconds_including_finiteness_checks": measured_seconds,
        "observations_per_second": batch_size * measured_steps / measured_seconds
        if measured_seconds
        else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "case_wall_seconds": time.perf_counter() - case_start,
        "device_after": device_snapshot(),
    }
    return result


def code_evidence(root: Path) -> dict[str, object]:
    files = [
        Path(__file__).resolve(),
        *sorted((root / "src/deepfake_detection_archive/models").rglob("*.py")),
    ]
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    executable = shutil.which("git")
    revision = "unavailable"
    if executable:
        revision = subprocess.run(  # noqa: S603 -- read-only revision query
            [executable, "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    return {"git_head": revision, "source_sha256": hashes}


def run(args: argparse.Namespace) -> dict[str, Any]:  # noqa: PLR0915
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback for this CUDA profile")
    torch.set_num_threads(2)
    started = time.perf_counter()
    deadline = started + args.max_seconds
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    budget = min(7 * GIB, total - GIB, free - GIB // 2)
    if budget < GIB:
        raise RuntimeError("insufficient free GPU memory for profile")
    torch.cuda.set_per_process_memory_fraction(budget / total)
    dtype = torch.bfloat16 if args.precision == "bfloat16" else torch.float16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support requested bfloat16 precision")
    root = Path(__file__).resolve().parents[1]
    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "synthetic_cuda_engineering_profile_only",
        "detection_performance": "not_evaluated",
        "weights": "random_initialization",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": total,
        "allocator_limit_bytes": budget,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_before": device_snapshot(),
        "seed": args.seed,
        "command_args": vars(args)
        | {"output": str(args.output), "summary": str(args.summary)},
        "code": code_evidence(root),
        "cases": [],
        "model_stops": {},
        "throughput_scope": "model resident synthetic tensors; includes numerical checks; excludes decoding/I/O",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for architecture in args.models:
        max_batch = 32768 if architecture == "media_feature_head_v1" else 2048
        candidates = args.batches or [
            2**power for power in range(max_batch.bit_length())
        ]
        previous_peak = 0
        previous_batch = 1
        for batch_size in candidates:
            if time.perf_counter() >= deadline:
                report["model_stops"][architecture] = "global_time_budget"
                break
            if previous_peak and previous_peak * batch_size / previous_batch > budget:
                report["model_stops"][architecture] = (
                    "next_batch_predicted_above_memory_budget"
                )
                break
            try:
                result = profile_case(
                    architecture,
                    batch_size,
                    deadline=deadline,
                    dtype=dtype,
                    seed=args.seed,
                )
            except (torch.OutOfMemoryError, ValueError, RuntimeError) as error:
                result = {
                    "architecture": architecture,
                    "batch_size": batch_size,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}"[:1500],
                }
            finally:
                gc.collect()
                torch.cuda.empty_cache()
            report["cases"].append(result)
            (args.output / f"{architecture}-batch-{batch_size}.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(result), flush=True)
            if result["status"] != "passed":
                report["model_stops"][architecture] = result["status"]
                break
            previous_peak = result["peak_reserved_bytes"]
            previous_batch = batch_size
            if result["case_wall_seconds"] > 30:
                report["model_stops"][architecture] = "case_latency_guard"
                break
        if architecture not in report["model_stops"]:
            report["model_stops"][architecture] = "batch_search_bound"
    torch.cuda.empty_cache()
    report["elapsed_seconds"] = time.perf_counter() - started
    report["device_after"] = device_snapshot()
    report["final_allocated_bytes"] = torch.cuda.memory_allocated()
    report["final_reserved_bytes"] = torch.cuda.memory_reserved()
    (args.output / "profile.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/profiles/cuda-models")
    )
    parser.add_argument("--max-seconds", type=int, default=540)
    parser.add_argument(
        "--precision", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument(
        "--models", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES)
    )
    parser.add_argument("--batches", nargs="+", type=int)
    parser.add_argument(
        "--summary", type=Path, default=Path("docs/research/model-profile.json")
    )
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 540:
        parser.error("--max-seconds must lie in [1, 540]")
    if args.batches and (
        any(value < 1 for value in args.batches)
        or args.batches != sorted(set(args.batches))
    ):
        parser.error("--batches must be positive and strictly increasing")
    report = run(args)
    summary = {
        **{key: value for key, value in report.items() if key != "cases"},
        "interpretation": "随机输入/随机权重的 CUDA 工程实测；没有鉴伪 EER 或真实数据泛化证据。",
        "model_summaries": {},
    }
    for architecture in args.models:
        cases = [
            case for case in report["cases"] if case["architecture"] == architecture
        ]
        passed = [case for case in cases if case["status"] == "passed"]
        summary["model_summaries"][architecture] = {
            "tested_batches": [case["batch_size"] for case in cases],
            "passed_batches": [case["batch_size"] for case in passed],
            "largest_passed_batch": passed[-1] if passed else None,
            "fastest_passed_batch": max(
                passed, key=lambda case: case["observations_per_second"]
            )
            if passed
            else None,
            "failures": [case for case in cases if case["status"] != "passed"],
        }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
