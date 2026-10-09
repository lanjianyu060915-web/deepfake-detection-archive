"""Decode the imported ASV train/dev corpus and inspect two old PCM candidates.

Run after prepare_asvspoof2019.py; retains its official splits and media hashes.
No evaluation data, archive rehash, or new near-duplicate search is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve  # type: ignore[import-untyped]

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    load_manifest,
    write_manifest,
)
from deepfake_detection_archive.media import AudioConfig, MediaError, decode_audio


CANDIDATES = (
    ("LA_D_1032020", "LA_D_1073474"),
    ("LA_D_1062379", "LA_T_1023667"),
)
AUDIO = AudioConfig(backend="soundfile")


def decode_record(record: SampleRecord, root: Path) -> SampleRecord:
    try:
        audio = decode_audio(root / record.path, AUDIO)
    except (MediaError, OSError) as exc:
        return replace(
            record,
            status="decode_failed",
            decode_status="failed",
            failure_reason=str(exc),
        )
    return replace(
        record,
        status="available",
        decode_status="ok",
        failure_reason="unknown",
        sample_rate=audio.original_sample_rate,
        duration_seconds=audio.original_duration_seconds,
    )


def compare_pcm(
    first: NDArray[np.float32], second: NDArray[np.float32]
) -> dict[str, object]:
    """Compare all lags with full-signal energy normalization, not tiny overlaps."""
    left = first.astype(np.float64) - first.mean(dtype=np.float64)
    right = second.astype(np.float64) - second.mean(dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    correlation = fftconvolve(left, right[::-1], mode="full")
    peak = int(np.argmax(np.abs(correlation)))
    lag = peak - (len(second) - 1)
    overlap = max(0, min(len(first), lag + len(second)) - max(0, lag))
    return {
        "sample_counts": [len(first), len(second)],
        "pcm_float32_sha256": [
            hashlib.sha256(wave.astype("<f4").tobytes()).hexdigest()
            for wave in (first, second)
        ],
        "identical_pcm": bool(np.array_equal(first, second)),
        "max_absolute_normalized_cross_correlation": (
            float(abs(correlation[peak]) / denominator) if denominator else None
        ),
        "peak_lag_samples": lag,
        "peak_overlap_samples": overlap,
        "interpretation": (
            "Exact PCM equality confirms identical waveforms; low correlation "
            "does not support waveform duplication, but cannot establish "
            "unrelated source utterances or speaker identities."
        ),
    }


def prepare(  # noqa: PLR0913, PLR0917 -- paths and explicit selection
    manifest: Path,
    root: Path,
    sources: Path,
    output_dir: Path,
    workers: int = 8,
    exclude_train: tuple[str, ...] = (),
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    records = load_manifest(manifest)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        decoded = list(pool.map(partial(decode_record, root=root), records))
    decoded_path = output_dir / "manifest-decoded.jsonl"
    digest = write_manifest(decoded, decoded_path)
    destination = output_dir / "sources.jsonl"
    if sources.resolve() != destination.resolve():
        shutil.copyfile(sources, destination)
    by_id = {record.sample_id: record for record in decoded}
    comparisons: list[dict[str, object]] = []
    for first, second in CANDIDATES:
        if not all(
            name in by_id and by_id[name].decode_status == "ok"
            for name in (first, second)
        ):
            continue
        comparisons.append(
            {
                "sample_ids": [first, second],
                **compare_pcm(
                    decode_audio(root / by_id[first].path, AUDIO).waveform,
                    decode_audio(root / by_id[second].path, AUDIO).waveform,
                ),
            }
        )
    failures = [
        {"sample_id": r.sample_id, "reason": r.failure_reason}
        for r in decoded
        if r.decode_status != "ok"
    ]
    if any(sid not in by_id or by_id[sid].split != "train" for sid in exclude_train):
        raise ValueError("exclude-train must identify an existing train recording")
    selected = [record for record in decoded if record.sample_id not in exclude_train]
    training_digest = write_manifest(selected, output_dir / "manifest-training.jsonl")
    elapsed = time.monotonic() - started
    summary: dict[str, object] = {
        "manifest": str(decoded_path),
        "manifest_sha256": digest,
        "audio": asdict(AUDIO),
        "workers": workers,
        "total": len(decoded),
        "split_label_counts": dict(
            sorted(Counter(f"{r.split}/label={r.label}" for r in decoded).items())
        ),
        "decode_failures": failures,
        "elapsed_seconds": elapsed,
        "files_per_second": len(decoded) / elapsed,
        "candidate_waveform_diagnostics": comparisons,
        "selection": "Official splits; optional explicit train-only exclusions.",
        "training_manifest_sha256": training_digest,
        "training_exclusions": list(exclude_train),
    }
    (output_dir / "decode-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--exclude-train", action="append", default=[])
    args = parser.parse_args()
    summary = prepare(
        args.manifest,
        args.root,
        args.sources,
        args.output_dir,
        args.workers,
        tuple(args.exclude_train),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
