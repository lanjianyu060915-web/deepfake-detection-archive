"""Fetch pinned upstream assets, verify hashes, convert locally, and probe CUDA.

No training occurs. Random inputs only check engineering compatibility of actual
pretrained encoders. No detection-performance claim follows from these probes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import shutil
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModel

from deepfake_detection_archive.models import FrozenAudioEncoder, FrozenVisionEncoder


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
CACHE = ROOT / "cache/pretrained"
MODELS = ROOT / "models/pretrained"
SOURCES: dict[str, dict[str, Any]] = {
    "resnet18-a1": {
        "repository": "timm/resnet18.a1_in1k",
        "revision": "491b427b45c94c7fb0e78b5474cc919aff584bbf",
        "files": {
            "model.safetensors": {
                "bytes": 46807446,
                "sha256": "80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612",
            },
        },
    },
    "hubert-base-ls960": {
        "repository": "facebook/hubert-base-ls960",
        "revision": "dba3bb02fda4248b6e082697eee756de8fe8aa8a",
        "files": {
            "pytorch_model.bin": {
                "bytes": 377569754,
                "sha256": "062249fffb353eab67547a2fbc129f7c31a2f459faf641b19e8fb007cc5c48ad",
            },
            "config.json": {
                "sha256": "56be398848bbd9cbc720172b0d45b2f01cac7652f2968f0f31f5faf9f2986acc",
            },
            "preprocessor_config.json": {
                "sha256": "4a93853b74278b7c769d07f5a861e5d12ceb5db2bced5620d335f87238cb9e86",
            },
        },
    },
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    actual = {"sha256": digest(path), "bytes": path.stat().st_size}
    if expected.get("sha256", actual["sha256"]) != actual["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}")
    if expected.get("bytes", actual["bytes"]) != actual["bytes"]:
        raise ValueError(f"size mismatch: {path}")
    return actual


def fetch(url: str, target: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if target.exists():
        return verify(target, expected)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "deepfake-detection-resource-preparation/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 -- fixed HTTPS URLs
    with urllib.request.urlopen(request, timeout=90) as response:  # noqa: S310
        append = offset > 0 and response.status == 206
        with partial.open("ab" if append else "wb") as stream:
            count = offset if append else 0
            announced = count
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                count += len(chunk)
                if count - announced >= 20 * 1024 * 1024:
                    print(f"download {target.name}: {count} bytes", flush=True)
                    announced = count
    actual = verify(partial, expected)
    partial.replace(target)
    return actual


def download(names: tuple[str, ...] | None = None) -> None:
    manifest = EVIDENCE / "upstream-resources.json"
    records: dict[str, Any] = (
        json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    )
    for name, source in SOURCES.items():
        if names is not None and name not in names:
            continue
        base = f"https://huggingface.co/{source['repository']}/resolve/{source['revision']}"
        files = {}
        for filename, expected in source["files"].items():
            url = f"{base}/{filename}"
            files[filename] = {
                "url": url,
                **fetch(url, CACHE / name / filename, expected),
            }
        card_path = EVIDENCE / f"{name}-model-card.md"
        card = fetch(f"{base}/README.md", card_path, {})
        if "license: apache-2.0" not in card_path.read_text(encoding="utf-8"):
            raise ValueError(
                f"model card does not declare expected Apache-2.0 license: {name}"
            )
        records[name] = {
            "repository": source["repository"],
            "revision": source["revision"],
            "files": files,
            "license": "Apache-2.0",
            "model_card": {"url": f"{base}/README.md", **card},
        }
    (EVIDENCE / "upstream-resources.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )


def convert() -> None:
    records: dict[str, Any] = {
        "_provenance": {
            "script_sha256": digest(Path(__file__)),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "timm", "transformers", "safetensors")
            },
        }
    }
    for name, source in SOURCES.items():
        for filename, expected in source["files"].items():
            verify(CACHE / name / filename, expected)
        target = MODELS / name
        target.mkdir(parents=True, exist_ok=True)
        if name == "resnet18-a1":
            state = load_file(CACHE / name / "model.safetensors", device="cpu")
            removed = {
                key: list(state.pop(key).shape) for key in ("fc.weight", "fc.bias")
            }
            checkpoint = target / "encoder.pt"
            torch.save(state, checkpoint)
            records[name] = {
                "conversion": "Remove exactly fc.weight and fc.bias, save bare feature-only state_dict.",
                "removed_keys": removed,
                "num_classes": 0,
                "files": {"encoder.pt": verify(checkpoint, {})},
            }
        else:
            state = torch.load(
                CACHE / name / "pytorch_model.bin",
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(state, dict) or any(
                not isinstance(value, torch.Tensor) for value in state.values()
            ):
                raise ValueError("HuBERT checkpoint must be a bare tensor state_dict")
            state = {key: value.contiguous() for key, value in state.items()}
            save_file(state, target / "model.safetensors", metadata={"format": "pt"})
            for filename in ("config.json", "preprocessor_config.json"):
                shutil.copyfile(CACHE / name / filename, target / filename)
            records[name] = {
                "conversion": "torch.load(weights_only=True), tensor.contiguous(), safetensors format=pt; configs byte-preserved.",
                "raw_state_keys": len(state),
                "files": {
                    path.name: verify(path, {}) for path in sorted(target.iterdir())
                },
            }
        print(json.dumps({"converted": name, **records[name]}), flush=True)
    (EVIDENCE / "converted-resources.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )


def download_hubert_xet() -> None:
    """Use the installed official Hub/Xet client for the larger pinned resource."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    name = "hubert-base-ls960"
    source = SOURCES[name]
    for filename, expected in source["files"].items():
        target = CACHE / name / filename
        if target.exists():
            verify(target, expected)
            continue
        downloaded = Path(
            hf_hub_download(
                repo_id=source["repository"],
                filename=filename,
                revision=source["revision"],
                cache_dir=CACHE / "huggingface-hub",
                token=False,
            )
        )
        actual = verify(downloaded, expected)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_suffix(target.suffix + ".xet-verified")
        shutil.copyfile(downloaded, staged)
        staged.replace(target)
        print(json.dumps({"downloaded_via_xet": filename, **actual}), flush=True)


def probe() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA probe requires GPU access; no silent CPU fallback")
    torch.set_num_threads(2)
    torch.manual_seed(20261007)
    converted = json.loads(
        (EVIDENCE / "converted-resources.json").read_text(encoding="utf-8")
    )
    records: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "purpose": "Pretrained resource compatibility on synthetic inputs only; detection quality not evaluated.",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "seed": 20261007,
        "script_sha256": digest(Path(__file__)),
        "resources": {},
    }
    for name in SOURCES:
        target = MODELS / name
        loading_info: dict[str, Any] = {}
        if name == "resnet18-a1":
            encoder: torch.nn.Module = FrozenVisionEncoder(
                "resnet18",
                target / "encoder.pt",
                expected_sha256=converted[name]["files"]["encoder.pt"]["sha256"],
            )
            loading_info = {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "strict": True,
            }
            shape: tuple[int, ...] = (1, 3, 224, 224)
        else:
            original_loader = AutoModel.from_pretrained

            def capture_loader(*args: Any, **kwargs: Any) -> Any:
                loaded, info = original_loader(*args, **kwargs)  # noqa: B023 -- invoked synchronously inside this iteration
                loading_info.update(info)  # noqa: B023
                return loaded, info

            hashes = {
                filename: entry["sha256"]
                for filename, entry in converted[name]["files"].items()
            }
            with patch.object(AutoModel, "from_pretrained", side_effect=capture_loader):
                encoder = FrozenAudioEncoder(target, expected_hashes=hashes)
            shape = (1, 16000)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        encoder = encoder.cuda().eval()
        inputs = torch.randn(shape, device="cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = encoder(inputs)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(outputs).all()):
            raise ValueError(f"non-finite CUDA outputs: {name}")
        records["resources"][name] = {
            "loading_info": loading_info,
            "input_shape": list(shape),
            "output_shape": list(outputs.shape),
            "dtype": str(outputs.dtype),
            "forward_seconds": time.perf_counter() - started,
            "parameters": sum(value.numel() for value in encoder.parameters()),
            "all_parameters_frozen": all(
                not value.requires_grad for value in encoder.parameters()
            ),
            "finite_outputs": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        print(json.dumps({"probed": name, **records["resources"][name]}), flush=True)
        del encoder, inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()
    (EVIDENCE / "loading-verification.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "download",
            "download-resnet",
            "download-hubert-xet",
            "convert",
            "probe",
        ),
    )
    phase = parser.parse_args().phase
    {
        "download": download,
        "download-resnet": lambda: download(("resnet18-a1",)),
        "download-hubert-xet": download_hubert_xet,
        "convert": convert,
        "probe": probe,
    }[phase]()


if __name__ == "__main__":
    main()
