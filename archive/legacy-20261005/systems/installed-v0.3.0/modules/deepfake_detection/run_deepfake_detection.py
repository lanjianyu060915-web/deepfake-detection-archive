from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from pyinstaller_pyz_runtime import PyInstallerArchiveError, install_importer


DEFAULT_INSTALL_ROOT = Path(
    os.environ.get(
        "DEEPFAKE_INSTALL_ROOT",
        str(Path.home() / "AppData" / "Local" / "Programs" / "DeepfakeDetectionSystem"),
    )
)
DEFAULT_TORCH_HOME = Path(
    os.environ.get(
        "DEEPFAKE_TORCH_HOME",
        str(Path(__file__).resolve().parents[2] / "runtime" / "cache" / "deepfake" / "torch"),
    )
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="调用项目已有多域融合深度伪造检测模块并输出冻结 JSON 契约。"
    )
    parser.add_argument("--query", required=True, help="待检测的单张图片")
    parser.add_argument("--output", required=True, help="统一 detection_result.json 输出路径")
    parser.add_argument(
        "--install-root",
        default=str(DEFAULT_INSTALL_ROOT),
        help="已有深度伪造识别系统的安装目录",
    )
    parser.add_argument(
        "--torch-home",
        default=str(DEFAULT_TORCH_HOME),
        help="包含官方 MobileNetV3-Small 校验权重的隔离 Torch 缓存根",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "auto", "cuda"),
        default="cpu",
        help="原模块设备参数；当前已验证环境为 cpu",
    )
    return parser


def _base_payload(query: Path, runtime_seconds: float) -> dict[str, Any]:
    return {
        "schema_version": "3.0",
        "module": "deepfake",
        "status": "error",
        "query_image": str(query),
        "is_deepfake": None,
        "score": None,
        "label": None,
        "runtime_seconds": max(0.0, float(runtime_seconds)),
        "details": {},
        "error_message": None,
        "method": {
            "name": "多域融合单图深伪风险甄别",
            "spatial_model": "MobileNetV3-Small",
            "frequency_model": "FrequencyStatDetector 87维统计特征 + LogisticRegression",
            "fusion_method": "weighted_score",
        },
        "metrics": {},
        "explanation": {},
        "warnings": [],
        "limitations": [
            "模型输出为深伪风险筛查分数，不是系统准确率或司法鉴定结论。",
            "三个模块的分数属于不同任务，不能相加或解释为综合概率。",
        ],
        "runtime": {"elapsed_seconds": max(0.0, float(runtime_seconds))},
        "provenance": {},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path_value: Any) -> str | None:
    """Hash an existing runtime resource without changing it."""

    if not path_value:
        return None
    path = Path(str(path_value)).expanduser().resolve(strict=False)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_resources(install_root: Path, torch_home: Path) -> tuple[Path, Path]:
    executable = install_root / "深度伪造识别系统.exe"
    spatial_model = (
        install_root
        / "models"
        / "spatial_mobilenet_finetuned_gpu"
        / "spatial_mobilenet_finetuned_gpu.pt"
    )
    frequency_model = (
        install_root
        / "models"
        / "frequency_stat_formal"
        / "frequency_stat_detector.joblib"
    )
    fusion_config = install_root / "models" / "fusion_formal" / "fusion_config.json"
    base_weight = torch_home / "hub" / "checkpoints" / "mobilenet_v3_small-047dcff4.pth"
    required = {
        "PyInstaller 入口": executable,
        "空间域模型": spatial_model,
        "频域模型": frequency_model,
        "融合配置": fusion_config,
        "MobileNetV3-Small 官方基础权重": base_weight,
    }
    missing = [f"{label}：{path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError("鉴伪运行资源缺失；" + "；".join(missing))
    return executable, base_weight


def _configure_installed_runtime(
    install_root: Path,
    executable: Path,
    torch_home: Path,
    writable_root: Path,
) -> list[Any]:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"鉴伪交付包要求 Python 3.12，当前为 {sys.version_info.major}.{sys.version_info.minor}。"
        )
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.setdefault("MPLCONFIGDIR", str(writable_root / "runtime" / "matplotlib"))
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    sys.dont_write_bytecode = True
    install_importer(executable, install_root)
    if str(install_root) not in sys.path:
        sys.path.insert(0, str(install_root))
    handles: list[Any] = []
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for directory in (install_root, install_root / "torch" / "lib"):
            if directory.is_dir():
                handles.append(os.add_dll_directory(str(directory)))
    return handles


def _native_details(native: dict[str, Any], install_root: Path, base_weight: Path) -> dict[str, Any]:
    native_run = Path(str(native.get("run_directory", ""))).resolve(strict=False)
    model_paths = native.get("model_paths", {})
    model_paths = model_paths if isinstance(model_paths, dict) else {}
    fusion_config_path = native.get("fusion_config_path")
    return {
        "module_source": str(install_root),
        "native_status": native.get("status"),
        "native_label": native.get("final_label"),
        "score_semantics": "原模块空间域与频域分支按配置权重融合后的FAKE风险分数；不是系统准确率。",
        "decision_threshold": native.get("decision_threshold"),
        "spatial_weight": native.get("spatial_weight"),
        "frequency_weight": native.get("frequency_weight"),
        "actual_device": native.get("actual_device"),
        "media_type": native.get("media_type"),
        "input_sha256": native.get("input_sha256"),
        "valid_frame_count": native.get("valid_frame_count"),
        "invalid_frame_count": native.get("invalid_frame_count"),
        "spatial_score": native.get("spatial_score_summary", {}).get("median"),
        "frequency_score": native.get("frequency_score_summary", {}).get("median"),
        "fusion_score": native.get("fusion_score_summary", {}).get("median"),
        "candidate_frame_count": native.get("candidate_frame_count"),
        "model_paths": model_paths,
        "model_sha256": {
            "spatial": _sha256_file(model_paths.get("spatial")),
            "frequency": _sha256_file(model_paths.get("frequency")),
            "fusion_config": _sha256_file(fusion_config_path),
            "official_base_weight": _sha256_file(base_weight),
        },
        "model_versions": native.get("model_versions", {}),
        "fusion_config_path": fusion_config_path,
        "fusion_config_hash": native.get("fusion_config_hash"),
        "media_metadata": native.get("media_metadata", {}),
        "timing": native.get("timing", {}),
        "frame_results": native.get("frame_results", []),
        "keyframe_paths": native.get("keyframe_paths", []),
        "native_runtime_seconds": native.get("total_elapsed_seconds"),
        "native_run_directory": str(native_run),
        "native_report_paths": native.get("report_paths", {}),
        "official_base_weight": str(base_weight),
        "warnings": list(native.get("warnings", [])),
        "disclaimer": native.get("disclaimer"),
    }


def _map_native_result(
    native: dict[str, Any],
    query: Path,
    runtime_seconds: float,
    install_root: Path,
    base_weight: Path,
) -> dict[str, Any]:
    payload = _base_payload(query, runtime_seconds)
    native_status = str(native.get("status", "ERROR"))
    details = _native_details(native, install_root, base_weight)
    payload["details"] = details
    payload["warnings"] = list(details.get("warnings") or [])
    payload["runtime"] = {
        "elapsed_seconds": max(0.0, float(runtime_seconds)),
        "actual_device": details.get("actual_device"),
        "native_elapsed_seconds": details.get("native_runtime_seconds"),
        "timing": details.get("timing", {}),
    }
    payload["provenance"] = {
        "input_sha256": details.get("input_sha256"),
        "model_sha256": details.get("model_sha256", {}),
        "fusion_config_sha256": details.get("fusion_config_hash"),
        "model_paths": details.get("model_paths", {}),
        "native_run_directory": details.get("native_run_directory"),
        "native_report_paths": details.get("native_report_paths", {}),
    }
    payload["metrics"] = {
        "spatial_score": details.get("spatial_score"),
        "frequency_score": details.get("frequency_score"),
        "spatial_weight": details.get("spatial_weight"),
        "frequency_weight": details.get("frequency_weight"),
        "fusion_score": details.get("fusion_score"),
        "decision_threshold": details.get("decision_threshold"),
        "decision_margin": (
            float(details["fusion_score"]) - float(details["decision_threshold"])
            if details.get("fusion_score") is not None
            and details.get("decision_threshold") is not None
            else None
        ),
        "valid_frame_count": details.get("valid_frame_count"),
        "invalid_frame_count": details.get("invalid_frame_count"),
        "candidate_frame_count": details.get("candidate_frame_count"),
        "media_metadata": details.get("media_metadata", {}),
    }
    if native_status == "SUCCESS":
        native_label = str(native.get("final_label", "")).upper()
        if native_label not in {"FAKE", "REAL"}:
            payload["error_message"] = f"原模块 SUCCESS 但返回未知标签：{native_label!r}"
            return payload
        score = native.get("final_fake_score")
        if score is None:
            payload["error_message"] = "原模块 SUCCESS 但没有返回融合伪造风险评分。"
            return payload
        payload.update(
            {
                "status": "success",
                "is_deepfake": native_label == "FAKE",
                "score": float(score),
                "label": "deepfake" if native_label == "FAKE" else "real",
                "error_message": None,
            }
        )
        payload["explanation"] = {
            "decision_rule": "fusion_score >= decision_threshold",
            "fusion_formula": "spatial_weight × spatial_score + frequency_weight × frequency_score",
            "decision": "疑似AI换脸" if native_label == "FAKE" else "未检出明显换脸风险",
            "boundary_band_enabled": False,
            "boundary_band_reason": "当前版本没有经独立验证的决策边界带，仅报告实际阈值差值。",
        }
        return payload
    if native_status in {"LOW_CONFIDENCE", "INSUFFICIENT_EVIDENCE"}:
        payload.update(
            {
                "status": "success",
                "score": (
                    float(native["final_fake_score"])
                    if native.get("final_fake_score") is not None
                    else None
                ),
                "label": native_status.lower(),
                "error_message": None,
            }
        )
        payload["explanation"] = {
            "decision": "证据不足，不能输出真假结论",
            "boundary_band_enabled": False,
            "boundary_band_reason": "当前版本没有经独立验证的决策边界带。",
        }
        return payload
    payload["error_message"] = str(
        native.get("error_message") or f"原鉴伪模块返回状态 {native_status}。"
    )
    return payload


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    started = time.perf_counter()
    query = Path(args.query).expanduser().resolve(strict=False)
    output = Path(args.output).expanduser().resolve(strict=False)
    install_root = Path(args.install_root).expanduser().resolve(strict=False)
    torch_home = Path(args.torch_home).expanduser().resolve(strict=False)
    payload = _base_payload(query, 0.0)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        if not query.is_file():
            raise RuntimeError(f"query 不存在或不是文件：{query}")
        executable, base_weight = _validate_resources(install_root, torch_home)
        _dll_handles = _configure_installed_runtime(
            install_root,
            executable,
            torch_home,
            output.parent,
        )
        from src.pipeline import DeepfakeDetectionPipeline

        native_output_root = output.parent / "visualizations" / "deepfake"
        pipeline = DeepfakeDetectionPipeline(
            project_root=install_root,
            device=args.device,
            output_root=native_output_root,
            writable_root=output.parent,
        )
        native = pipeline.detect_media(query)
        payload = _map_native_result(
            native,
            query,
            time.perf_counter() - started,
            install_root,
            base_weight,
        )
    except (PyInstallerArchiveError, OSError, RuntimeError, ValueError, ImportError) as exc:
        payload = _base_payload(query, time.perf_counter() - started)
        payload["error_message"] = f"{type(exc).__name__}: {exc}"
    payload["runtime_seconds"] = time.perf_counter() - started
    payload["runtime"] = {
        **(payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}),
        "elapsed_seconds": payload["runtime_seconds"],
    }
    try:
        _write_json(output, payload)
    except OSError as exc:
        print(f"无法写入鉴伪结果 JSON：{output}；{exc}", file=sys.stderr)
        return 2, payload
    return (0 if payload["status"] == "success" else 1), payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    returncode, payload = run(args)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
