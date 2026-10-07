"""Synthetic CUDA BF16 trainer integration; no detection-quality inference."""

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from deepfake_detection_archive.models import (
    LogMelCNN,
    LogMelConfig,
    MediaFeatureClassifier,
    MediaFeatureConfig,
    load_checkpoint,
    save_checkpoint,
)
from deepfake_detection_archive.models.checkpoint import sha256_file
from deepfake_detection_archive.tracking import atomic_json, code_identity, utc_now
from deepfake_detection_archive.training import TrainConfig, train_model


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "outputs/profiles/trainer-bf16"
SEED = 20261007


def probe(architecture, model, train_inputs, dev_inputs):
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    fixture = OUTPUT / "fixtures" / f"{architecture}.pt"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train": train_inputs, "dev": dev_inputs, "labels": labels}, fixture)
    initial = save_checkpoint(OUTPUT / "initial" / f"{architecture}.pt", model)
    original = {key: parameter.detach().clone() for key, parameter in model.named_parameters()}
    observed_dtypes = []

    def observe(module, args, output):
        if module.training:
            observed_dtypes.append(str(output.dtype))

    hook = model.register_forward_hook(observe)
    training = [
        (train_inputs[:1], labels[:1]),
        (train_inputs[1:3], labels[1:3]),
        (train_inputs[3:], labels[3:]),
    ]
    config = TrainConfig(
        max_epochs=1, device="cuda", amp=True, amp_dtype="bfloat16",
        accumulation_steps=2, seed=SEED, max_grad_norm=1.0,
    )
    result = train_model(
        model, training, [(dev_inputs, labels.clone())],
        config=config, output_dir=OUTPUT / architecture,
        provenance={
            "data_sha256": sha256_file(fixture),
            "weights_sha256": initial["sha256"],
            "code_sha256": code_identity()["code_sha256"],
            "dataset_kind": "synthetic_random_tensors",
            "purpose": "trainer_cuda_bfloat16_integration_only",
        },
    )
    hook.remove()
    run = json.loads(result.run_path.read_text())
    history = json.loads(result.history_path.read_text())
    assert observed_dtypes == ["torch.bfloat16"] * 3, observed_dtypes
    assert run["grad_scaler_enabled"] is False
    assert history[0]["optimizer_updates"] == 2
    assert history[0]["skipped_updates"] == 0
    assert history[0]["train_samples"] == history[0]["dev_samples"] == 4
    model.cpu()
    changed = sum(
        not torch.equal(parameter.detach(), original[name])
        for name, parameter in model.named_parameters()
    )
    assert changed > 0
    restored, descriptor = load_checkpoint(
        result.best_checkpoint, expected_sha256=result.checkpoint_sha256
    )
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name]), name
        assert bool(torch.isfinite(tensor).all()), name
    with torch.inference_mode():
        torch.testing.assert_close(model(dev_inputs), restored(dev_inputs), rtol=0, atol=0)
    return {
        "architecture": architecture,
        "fixture_path": str(fixture.relative_to(ROOT)),
        "fixture_sha256": sha256_file(fixture),
        "training_config": asdict(config),
        "observed_training_output_dtypes": observed_dtypes,
        "grad_scaler_enabled": run["grad_scaler_enabled"],
        "optimizer_updates": history[0]["optimizer_updates"],
        "skipped_updates": history[0]["skipped_updates"],
        "changed_parameter_tensors": changed,
        "train_samples": 4,
        "dev_samples": 4,
        "checkpoint_path": str(result.best_checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": descriptor["sha256"],
        "checkpoint_finite": True,
        "restored_state_matches_trained_model": True,
        "restored_cpu_predictions_match": True,
        "seconds": run["seconds"],
        "peak_cuda_allocated_bytes": run["peak_cuda_allocated_bytes"],
        "peak_cuda_reserved_bytes": run["peak_cuda_reserved_bytes"],
        "history_path": str(result.history_path.relative_to(ROOT)),
        "history_sha256": sha256_file(result.history_path),
        "synthetic_dev_eer": result.best_dev_eer,
        "performance_claim": "none; synthetic EER tests control flow only",
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this execution environment")
    assert torch.cuda.is_bf16_supported(including_emulation=False)
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED + 1)
    logmel = LogMelCNN(LogMelConfig(n_fft=256, hop_length=128, n_mels=32, channels=8, hidden_dim=16))
    results = [probe(
        "logmel_cnn_v1", logmel,
        torch.randn(4, 4096, generator=generator),
        torch.randn(4, 4096, generator=generator),
    )]
    torch.manual_seed(SEED)
    head = MediaFeatureClassifier(MediaFeatureConfig(input_dim=32, hidden_dim=16))
    results.append(probe(
        "media_feature_head_v1", head,
        torch.randn(4, 3, 32, generator=generator),
        torch.randn(4, 3, 32, generator=generator),
    ))
    hardware = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    report = {
        "schema_version": 1, "created_at": utc_now(),
        "purpose": "Actual train_model CUDA BF16 synthetic integration, not real detection quality",
        "dataset_kind": "synthetic_random_tensors", "seed": SEED,
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "hardware": hardware, "native_cuda_bfloat16_supported": True,
        "code": code_identity(),
        "training_source_sha256": sha256_file(ROOT / "src/deepfake_detection_archive/training.py"),
        "probe_script_sha256": sha256_file(Path(__file__)),
        "command": "UV_CACHE_DIR=/tmp/aigc-uv-cache uv run --no-sync python outputs/profiles/trainer-bf16/run_probe.py",
        "results": results,
        "scope_limit": "Two small synthetic CPU-generated datasets; no real data, weights, holdout, or competition evaluation",
    }
    atomic_json(OUTPUT / "report.json", report)
    atomic_json(ROOT / "docs/research/trainer-profile.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
