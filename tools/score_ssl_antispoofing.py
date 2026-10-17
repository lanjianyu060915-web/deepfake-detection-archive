"""Score audio with the official SSL_Anti-spoofing LA checkpoint, without fairseq.

The AASIST back-end is the pinned upstream ``model.py`` (TakHemlata/SSL_Anti-
spoofing, MIT) executed as published except for its fairseq import and SSLModel
class. The XLS-R front-end is transformers ``Wav2Vec2Model`` loaded from the
checkpoint's fine-tuned fairseq keys renamed one to one; only the unused
pre-training heads are dropped and loading is strict. Input follows upstream
evaluation: 16 kHz mono, the first 64600 samples or whole-clip tiling, no
normalisation, fp32. Upstream scores are the bonafide logit; records store
``-bonafide`` so a higher score means fake, as in the project contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

from deepfake_detection_archive.contracts import ScoreRecord
from deepfake_detection_archive.data.manifest import SampleRecord, load_manifest
from deepfake_detection_archive.data.split import statistical_groups
from deepfake_detection_archive.media import AudioConfig, decode_audio


CUT = 64600
PREFIX = "ssl_model.model."
DROPPED = ("quantizer.", "project_q.", "final_proj.")
UPSTREAM_COMMIT = "4acaa61dcef5f7610f43aa4d0b29c4559b970cd2"
UPSTREAM_MODEL_SHA256 = (
    "08b2b99b9cc0e90732746471325185f2eb144795ee35338e0a02951015a856c6"
)
RULES = (
    (r"^feature_extractor\.conv_layers\.(\d+)\.0\.", r"feature_extractor.conv_layers.\1.conv."),
    (
        r"^feature_extractor\.conv_layers\.(\d+)\.2\.1\.",
        r"feature_extractor.conv_layers.\1.layer_norm.",
    ),
    (r"^post_extract_proj\.", "feature_projection.projection."),
    (r"^layer_norm\.", "feature_projection.layer_norm."),
    (
        r"^encoder\.pos_conv\.0\.weight_g$",
        "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    ),
    (
        r"^encoder\.pos_conv\.0\.weight_v$",
        "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
    ),
    (r"^encoder\.pos_conv\.0\.bias$", "encoder.pos_conv_embed.conv.bias"),
    (r"^encoder\.layers\.(\d+)\.self_attn_layer_norm\.", r"encoder.layers.\1.layer_norm."),
    (r"^encoder\.layers\.(\d+)\.self_attn\.", r"encoder.layers.\1.attention."),
    (
        r"^encoder\.layers\.(\d+)\.fc1\.",
        r"encoder.layers.\1.feed_forward.intermediate_dense.",
    ),
    (r"^encoder\.layers\.(\d+)\.fc2\.", r"encoder.layers.\1.feed_forward.output_dense."),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fairseq_to_hf(key: str) -> str | None:
    """Map one fairseq XLS-R key to transformers; None drops pre-training heads."""
    if key.startswith(DROPPED):
        return None
    if key == "mask_emb":
        return "masked_spec_embed"
    for pattern, replacement in RULES:
        if re.search(pattern, key):
            return re.sub(pattern, replacement, key)
    return key


def pad_or_tile(wave: NDArray[np.float32], max_len: int = CUT) -> NDArray[np.float32]:
    """Upstream ``data_utils_SSL.pad``: head crop, or tile the whole clip."""
    if wave.ndim != 1 or wave.size == 0:
        raise ValueError("expected a nonempty mono waveform")
    if wave.shape[0] >= max_len:
        return wave[:max_len]
    repeats = max_len // wave.shape[0] + 1
    return np.tile(wave, repeats)[:max_len]


class SSLModel(nn.Module):
    """transformers XLS-R with the attribute layout the checkpoint expects."""

    config: Wav2Vec2Config | None = None

    def __init__(self, device: object) -> None:
        super().__init__()
        if SSLModel.config is None:
            raise RuntimeError("SSLModel.config must be set before building")
        self.model = Wav2Vec2Model(SSLModel.config)
        self.device = device
        self.out_dim = 1024

    def extract_feat(self, input_data: torch.Tensor) -> torch.Tensor:
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]
        hidden: torch.Tensor = self.model(input_data).last_hidden_state
        return hidden


def upstream_model_class(source: Path, expected_sha256: str) -> type[nn.Module]:
    """Execute pinned upstream model.py with SSLModel swapped for the HF port."""
    actual = sha256_file(source)
    if actual != expected_sha256:
        raise ValueError(f"upstream model.py sha256 {actual} != {expected_sha256}")
    text = source.read_text(encoding="utf-8")
    start = text.index("class SSLModel")
    end = text.index("#---------AASIST back-end")
    text = text[:start] + text[end:]
    text = text.replace("import fairseq\n", "")
    namespace: dict[str, Any] = {"__name__": "ssl_antispoofing_upstream", "SSLModel": SSLModel}
    exec(compile(text, str(source), "exec"), namespace)  # noqa: S102 -- pinned MIT source
    model_class: type[nn.Module] = namespace["Model"]
    return model_class


def build_model(
    weights: Path, upstream: Path, config: Path, device: str
) -> tuple[nn.Module, dict[str, Any]]:
    """Build the detector and load the checkpoint strictly through the key map."""
    SSLModel.config = Wav2Vec2Config.from_dict(
        json.loads(config.read_text(encoding="utf-8"))
    )
    model = upstream_model_class(upstream, UPSTREAM_MODEL_SHA256)(None, device)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    mapped: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    for key, value in state.items():
        target = key
        if key.startswith(PREFIX):
            renamed = fairseq_to_hf(key[len(PREFIX) :])
            if renamed is None:
                dropped.append(key)
                continue
            target = PREFIX + renamed
        if target in mapped:
            raise ValueError(f"two checkpoint keys map to {target}")
        mapped[target] = value
    expected = model.state_dict()
    mismatched = [k for k, v in mapped.items() if k in expected and expected[k].shape != v.shape]
    if mismatched:
        raise ValueError(f"shape mismatch: {mismatched[:5]}")
    model.load_state_dict(mapped, strict=True)
    model.to(device).eval()
    report = {
        "weights_sha256": sha256_file(weights),
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_model_sha256": UPSTREAM_MODEL_SHA256,
        "xlsr_config_sha256": sha256_file(config),
        "checkpoint_keys": len(state),
        "dropped_pretraining_keys": dropped,
        "loaded_keys": len(mapped),
    }
    return model, report


def score_row(
    record: SampleRecord,
    groups: dict[str, str],
    score: float | None,
    elapsed: float,
    error: Exception | None,
) -> ScoreRecord:
    return ScoreRecord(
        sample_id=record.sample_id,
        modality="audio",
        score=score,
        status="ok" if error is None else "error",
        elapsed_seconds=max(elapsed, 0.0),
        label=record.label,
        group_id=groups.get(record.sample_id, "unknown"),
        source=record.source,
        content_type=record.content_type,
        generator=record.generator,
        error_code=None if error is None else type(error).__name__,
        error_message=None if error is None else (str(error) or type(error).__name__),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--upstream-model", type=Path, required=True)
    parser.add_argument("--xlsr-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--backend", choices=("soundfile", "ffmpeg"), default="soundfile")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, report = build_model(
        args.weights, args.upstream_model, args.xlsr_config, args.device
    )
    records = [r for r in load_manifest(args.manifest) if r.status == "available"]
    if any(r.split == "holdout" for r in records):
        sys.exit("refusing to score holdout records")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "mapping.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    groups = statistical_groups(records)
    audio = AudioConfig(sample_rate=16000, backend=args.backend)
    rows: list[ScoreRecord] = []
    started = time.monotonic()
    with (args.output / "logits.jsonl").open("w", encoding="utf-8") as raw:
        for offset in range(0, len(records), args.batch_size):
            waves: list[NDArray[np.float32]] = []
            decoded: list[tuple[SampleRecord, float]] = []
            for record in records[offset : offset + args.batch_size]:
                begin = time.monotonic()
                try:
                    wave = decode_audio(args.data_root / record.path, audio).waveform
                    waves.append(pad_or_tile(np.asarray(wave, dtype=np.float32)))
                    decoded.append((record, time.monotonic() - begin))
                except Exception as exc:  # noqa: BLE001 -- per-file failure accounting
                    rows.append(score_row(record, groups, None, time.monotonic() - begin, exc))
            if not waves:
                continue
            forward = time.monotonic()
            with torch.inference_mode():
                batch = torch.from_numpy(np.stack(waves)).to(args.device)
                logits = model(batch).float().cpu()
            share = (time.monotonic() - forward) / len(waves)
            for (record, seconds), logit in zip(decoded, logits, strict=True):
                spoof, bonafide = float(logit[0]), float(logit[1])
                raw.write(
                    json.dumps(
                        {"sample_id": record.sample_id, "spoof": spoof, "bonafide": bonafide}
                    )
                    + "\n"
                )
                rows.append(score_row(record, groups, -bonafide, seconds + share, None))
    with (args.output / "scores.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "samples": len(rows),
                "failures": sum(r.status != "ok" for r in rows),
                "seconds": round(time.monotonic() - started, 1),
                "loaded_keys": report["loaded_keys"],
            }
        )
    )


if __name__ == "__main__":
    main()
