"""Import paired music and speech+music clips for external diagnostic evaluation."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
from scipy.signal import resample_poly  # type: ignore[import-untyped]

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    SourceRecord,
    file_sha256,
    write_manifest,
)
from deepfake_detection_archive.tracking import atomic_json


SOURCE = "ai-openbmat-external"
URL = "https://zenodo.org/records/20031232"
RATE = 16000
SECONDS = 4.0
FADE_MARGIN = 0.1


def _intervals(metadata: dict[str, Any]) -> list[tuple[float, float, set[str]]]:
    intervals = []
    crossfade = float(metadata["config"]["crossfade_ms"]) / 1000
    start = 0.0
    for segment in metadata["segments"]:
        duration = float(segment["duration"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("segment duration must be positive and finite")
        end = start + duration
        intervals.append((start, end, {s["type"] for s in segment["sources"]}))
        start = end - crossfade
    return intervals


def _select(human: dict[str, Any], ai: dict[str, Any]) -> dict[str, tuple[int, float]]:
    selected: dict[str, tuple[int, float]] = {}
    longest: dict[str, float] = {}
    for index, (real, fake) in enumerate(
        zip(_intervals(human), _intervals(ai), strict=False)
    ):
        if real[2] == fake[2] == {"music"}:
            bucket = "music"
        elif all({"music", "speech"} <= item[2] for item in (real, fake)):
            bucket = "mixed"
        else:
            continue
        start, end = max(real[0], fake[0]), min(real[1], fake[1])
        duration = end - start
        if duration < SECONDS + 2 * FADE_MARGIN:
            continue
        if duration > longest.get(bucket, 0.0):
            longest[bucket] = duration
            selected[bucket] = (index, (start + end - SECONDS) / 2)
    return selected


def _clip(raw: Path, destination: Path, start: float) -> None:
    with sf.SoundFile(raw) as audio:
        rate = audio.samplerate
        offset, count = round(start * rate), round(SECONDS * rate)
        if offset < 0 or offset + count > len(audio):
            raise ValueError("selected clip exceeds actual audio duration")
        audio.seek(offset)
        samples = audio.read(count, dtype="float32", always_2d=True).mean(axis=1)
    if not np.isfinite(samples).all():
        raise ValueError("nonfinite audio")
    divisor = math.gcd(rate, RATE)
    samples = resample_poly(samples, RATE // divisor, rate // divisor)
    peak = float(np.max(np.abs(samples)))
    if peak == 0:
        raise ValueError("silent selected clip")
    sf.write(destination, samples * (0.95 / peak), RATE, subtype="PCM_16")


def _metadata_provenance(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "reference_structure": metadata.get("reference_structure"),
        "config": metadata["config"],
        "segments": [
            {
                **{key: value for key, value in segment.items() if key != "sources"},
                "sources": [
                    {**source, "file": Path(source["file"]).name}
                    for source in segment["sources"]
                ],
            }
            for segment in metadata["segments"]
        ],
    }


def _record(  # noqa: PLR0913, PLR0917
    pair: str,
    side: str,
    bucket: str,
    start: float,
    raw: Path,
    output: Path,
) -> tuple[SampleRecord, str | None]:
    record = SampleRecord(
        sample_id=f"ai-openbmat:{pair}:{side}:{bucket}",
        path=f"media/{pair}_{side}_{bucket}.wav",
        modality="audio",
        label=int(side == "ai"),
        source=SOURCE,
        content_type=bucket,
        generator="Suno" if side == "ai" else "unknown",
        generator_version="3.5" if side == "ai" else "unknown",
        original_id=f"ai-openbmat:{pair}:{side}",
        group_id=f"ai-openbmat-pair:{pair}",
        derivation_group_id=f"ai-openbmat-pair:{pair}",
        license_id="CC-BY-4.0",
        split="dev",
    )
    try:
        _clip(raw, output / record.path, start)
    except (OSError, ValueError, sf.LibsndfileError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        return replace(
            record, status="decode_failed", decode_status="failed", failure_reason=error
        ), error
    return replace(
        record,
        sha256=file_sha256(output / record.path),
        container="wav",
        codec="pcm_s16le",
        duration_seconds=SECONDS,
        sample_rate=RATE,
        decode_status="ok",
    ), None


def prepare(raw_dir: Path, output: Path) -> dict[str, Any]:
    pairs = sorted(
        {path.stem for path in (raw_dir / "human").glob("gen_*.json")}
        | {
            path.stem.removesuffix("_ai")
            for path in (raw_dir / "ai").glob("gen_*_ai.json")
        }
    )
    if not pairs:
        raise ValueError("no paired metadata found")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("output must be empty")
    (output / "media").mkdir()
    records: list[SampleRecord] = []
    exclusions: Counter[str] = Counter()
    excluded_pairs: list[dict[str, str]] = []
    with (output / "provenance.jsonl").open("w", encoding="utf-8") as stream:
        for pair in pairs:
            stems = {"human": pair, "ai": f"{pair}_ai"}
            metadata = {
                side: json.loads((raw_dir / side / f"{stems[side]}.json").read_text())
                for side in ("human", "ai")
            }
            if (
                metadata["human"]["reference_structure"]["file"]
                != metadata["ai"]["reference_structure"]["file"]
            ):
                raise ValueError(f"reference_structure.file differs for pair {pair}")
            selected = _select(metadata["human"], metadata["ai"])
            origins = {
                side: _metadata_provenance(data) for side, data in metadata.items()
            }
            raw_paths = {
                side: raw_dir / side / f"{stems[side]}.wav" for side in metadata
            }
            raw_hashes = {
                side: file_sha256(path) if path.is_file() else "unknown"
                for side, path in raw_paths.items()
            }
            for bucket in ("mixed", "music"):
                if bucket not in selected:
                    exclusions[f"{bucket}:no_eligible_common_segment"] += 1
                    excluded_pairs.append({"pair_id": pair, "bucket": bucket})
                    continue
                index, start = selected[bucket]
                for side, raw in raw_paths.items():
                    record, error = _record(pair, side, bucket, start, raw, output)
                    records.append(record)
                    stream.write(
                        json.dumps(
                            {
                                "sample_id": record.sample_id,
                                "pair_id": pair,
                                "bucket": bucket,
                                "segment_index": index,
                                "start_seconds": start,
                                "duration_seconds": SECONDS,
                                "raw_sha256": raw_hashes[side],
                                "clip_sha256": record.sha256,
                                "pair_metadata": origins,
                                "error": error,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    write_manifest(records, output / "manifest.jsonl")
    for bucket in ("mixed", "music"):
        write_manifest(
            [record for record in records if record.content_type == bucket],
            output / f"{bucket}-dev-manifest.jsonl",
        )
    source = SourceRecord(
        source=SOURCE,
        url=URL,
        version="20031232",
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        downloaded_at=datetime.now(UTC).isoformat(),
        allowed_uses=("research",),
        license_status="unresolved",
        archive_sha256="unknown",
    )
    (output / "sources.jsonl").write_text(json.dumps(asdict(source)) + "\n")
    summary = {
        "source": SOURCE,
        "url": URL,
        "pair_count": len(pairs),
        "sample_count": len(records),
        "status": dict(Counter(r.status for r in records)),
        "bucket_counts": dict(Counter(r.content_type for r in records)),
        "exclusions": dict(exclusions),
        "excluded_pairs": excluded_pairs,
        "split": "dev",
        "competition_approved": False,
        "license_status": "unresolved",
        "archive_sha256": "unknown",
        "archive_verification": "range-extracted subset; whole archive not verified",
        "labeling_limit": "Labels from upstream; Suno Extend human prompt prefix "
        "removal unverified; cannot certify generated content in every selected "
        "4-second AI clip. External exploration only, not training or a formal gold standard.",
        "source_relation": "Human BAF/Epidemic Sound, AIME MTG-Jamendo; "
        "track/artist overlap unverified.",
        "processing": {
            "seconds": SECONDS,
            "sample_rate": RATE,
            "channels": "mean",
            "resampling": "scipy.signal.resample_poly",
            "peak": 0.95,
            "subtype": "PCM_16",
            "fade_margin_seconds": FADE_MARGIN,
        },
        "grouping_limit": "Pair groups do not prove artist or material independence. "
        "Before training, connect pairs through reused music, speech and reference materials.",
    }
    atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.raw_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
