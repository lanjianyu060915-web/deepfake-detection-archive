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
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from deepfake_detection_archive.contracts import ScoreRecord
from deepfake_detection_archive.data.manifest import SampleRecord, load_manifest
from deepfake_detection_archive.data.split import statistical_groups
from deepfake_detection_archive.media import AudioConfig, MediaError, decode_audio
from deepfake_detection_archive.models.ssl_antispoofing import build_model, pad_or_tile


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


def main() -> None:  # noqa: PLR0912, PLR0915 -- single research batch workflow
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--upstream-model", type=Path, required=True)
    parser.add_argument("--xlsr-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("soundfile", "ffmpeg"), default="soundfile"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, report = build_model(
        args.weights, args.upstream_model, args.xlsr_config, args.device
    )
    records = load_manifest(args.manifest)
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
                    if record.status != "available":
                        raise MediaError(record.status, record.failure_reason)  # noqa: TRY301 -- row accounting
                    if record.modality != "audio":
                        raise MediaError("wrong_modality", "expected audio")  # noqa: TRY301 -- row accounting
                    wave = decode_audio(args.data_root / record.path, audio).waveform
                    waves.append(pad_or_tile(np.asarray(wave, dtype=np.float32)))
                    decoded.append((record, time.monotonic() - begin))
                except Exception as exc:  # per-file failure accounting
                    rows.append(
                        score_row(record, groups, None, time.monotonic() - begin, exc)
                    )
            if not waves:
                continue
            forward = time.monotonic()
            try:
                with torch.inference_mode():
                    batch = torch.from_numpy(np.stack(waves)).to(args.device)
                    logits = model(batch).float().cpu()
            except Exception as exc:  # retain every decoded row on model failure
                share = (time.monotonic() - forward) / len(waves)
                for record, seconds in decoded:
                    rows.append(score_row(record, groups, None, seconds + share, exc))
                continue
            share = (time.monotonic() - forward) / len(waves)
            for index, (record, seconds) in enumerate(decoded):
                try:
                    logit = logits[index]
                    spoof, bonafide = float(logit[0]), float(logit[1])
                    row = score_row(record, groups, -bonafide, seconds + share, None)
                    raw.write(
                        json.dumps(
                            {
                                "sample_id": record.sample_id,
                                "spoof": spoof,
                                "bonafide": bonafide,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    rows.append(row)
                except Exception as exc:  # malformed model row stays in denominator
                    rows.append(score_row(record, groups, None, seconds + share, exc))
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
