"""Prepare split-preserving music/real-speech mixtures and pure replay controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf  # type: ignore[import-untyped]

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    SourceRecord,
    file_sha256,
    load_manifest,
    load_sources,
    write_manifest,
)
from deepfake_detection_archive.data.split import validate_splits
from deepfake_detection_archive.tracking import atomic_json


RATE = 16000
COUNT = RATE * 4
SOURCE = "aime-asv-mixture-research"


def _crop(path: Path, *, music: bool) -> tuple[npt.NDArray[np.float64], float]:
    with sf.SoundFile(path) as audio:
        if audio.samplerate != RATE or audio.channels != 1:
            raise ValueError("expected 16 kHz mono audio")
        if len(audio) < COUNT or (music and len(audio) != COUNT):
            raise ValueError("music must be four seconds; speech at least four seconds")
        offset = (len(audio) - COUNT) // 2
        audio.seek(offset)
        samples = audio.read(COUNT, dtype="float64")
    if not np.isfinite(samples).all() or not np.any(samples):
        raise ValueError("nonfinite or zero-RMS audio")
    return samples, offset / RATE


def _mix(
    music: npt.NDArray[np.float64], speech: npt.NDArray[np.float64], snr: float
) -> tuple[npt.NDArray[np.float64], dict[str, float]]:
    music_rms = float(np.sqrt(np.mean(music**2)))
    speech_rms = float(np.sqrt(np.mean(speech**2)))
    music_gain = 10 ** (snr / 20) / music_rms
    speech_gain = 1 / speech_rms
    combined = music * music_gain + speech * speech_gain
    peak = float(np.max(np.abs(combined)))
    if peak == 0:
        raise ValueError("mixture has zero RMS")
    gain = 0.95 / peak
    return combined * gain, {
        "music_rms": music_rms,
        "speech_rms": speech_rms,
        "music_gain": music_gain * gain,
        "speech_gain": speech_gain * gain,
        "peak_gain": gain,
    }


def _inputs(
    music_manifest: Path, speech_manifest: Path, untrained_manifest: Path | None
) -> tuple[list[SampleRecord], dict[str, list[SampleRecord]], set[str], set[str]]:
    main = load_manifest(music_manifest)
    if any(row.split not in {"train", "dev", "holdout"} for row in main):
        raise ValueError("main music needs assigned research splits")
    extra = load_manifest(untrained_manifest) if untrained_manifest else []
    if any(row.split != "dev" for row in extra):
        raise ValueError("untrained music must use dev only")
    merged = {row.sample_id: row for row in main}
    for row in extra:
        if row.sample_id in merged and row != merged[row.sample_id]:
            raise ValueError(f"conflicting music metadata: {row.sample_id}")
        merged[row.sample_id] = row
    music = list(merged.values())
    speech = load_manifest(speech_manifest)
    for rows in (music, speech):
        errors = validate_splits(rows, require_class_coverage=False)
        if errors:
            raise ValueError("; ".join(errors))
    selected = [row for row in music if row.split in {"train", "dev"}]
    if any(
        row.modality != "audio" or row.label is None or row.group_id == "unknown"
        for row in selected
    ):
        raise ValueError("music needs audio labels and known groups")
    pools = {
        split: sorted(
            (
                row
                for row in speech
                if row.split == split
                and row.label == 0
                and row.status == "available"
                and row.decode_status == "ok"
                and row.duration_seconds is not None
                and row.duration_seconds >= 4
            ),
            key=lambda row: row.sample_id,
        )
        for split in {row.split for row in selected}
    }
    if any(not pool for pool in pools.values()):
        raise ValueError("each music split needs eligible real speech")
    if any(
        row.modality != "audio" or row.identity_id == "unknown"
        for pool in pools.values()
        for row in pool
    ):
        raise ValueError("speech needs audio modality and known speaker identity")
    return selected, pools, {r.sample_id for r in main}, {r.sample_id for r in extra}


def _allowed_uses(
    music: list[SampleRecord],
    speakers: dict[tuple[str, int], SampleRecord],
    sources: dict[str, SourceRecord],
) -> tuple[str, ...]:
    if not music:
        raise ValueError("no train/dev music selected")
    used_sources: dict[str, SourceRecord] = {}
    for row in [*music, *speakers.values()]:
        source = sources.get(row.source)
        if source is None:
            raise ValueError(f"missing source metadata: {row.source}")
        if row.license_id == "unknown" or row.license_id != source.license_id:
            raise ValueError(f"sample/source license mismatch: {row.sample_id}")
        if source.license_status != "approved" or "training" not in source.allowed_uses:
            raise ValueError(f"source training permission not approved: {row.source}")
        used_sources[row.source] = source
    if (
        len({row.license_id for row in music}) != 1
        or len({row.license_id for row in speakers.values()}) != 1
    ):
        raise ValueError("mixtures require a single music/speech license pair")
    return tuple(
        sorted(
            set.intersection(
                *(set(source.allowed_uses) for source in used_sources.values())
            )
        )
    )


def prepare(  # noqa: PLR0913
    *,
    music_manifest: Path,
    music_root: Path,
    music_sources: Path,
    speech_manifest: Path,
    speech_root: Path,
    speech_sources: Path,
    output: Path,
    untrained_manifest: Path | None = None,
    seed: int = 42,
    snr_db: tuple[float, float] = (-5, 5),
) -> dict[str, Any]:
    if len(set(snr_db)) != 2 or not all(math.isfinite(snr) for snr in snr_db):
        raise ValueError("two distinct finite SNR values required")
    music, pools, main_ids, extra_ids = _inputs(
        music_manifest, speech_manifest, untrained_manifest
    )
    sources = load_sources(music_sources)
    for name, source in load_sources(speech_sources).items():
        if name in sources and sources[name] != source:
            raise ValueError(f"conflicting source: {name}")
        sources[name] = source
    speakers: dict[tuple[str, int], SampleRecord] = {}
    for row in music:
        for index, snr in enumerate(snr_db):
            digest = hashlib.sha256(f"{seed}:{row.group_id}:{snr:g}".encode()).digest()
            pool = pools[row.split]
            speakers[row.sample_id, index] = pool[
                int.from_bytes(digest, "big") % len(pool)
            ]
    allowed_uses = _allowed_uses(music, speakers, sources)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("output must be empty")
    (output / "music").symlink_to(music_root.resolve(), target_is_directory=True)
    (output / "media").mkdir()
    mixed: list[SampleRecord] = []
    control: list[SampleRecord] = []
    provenance: list[dict[str, Any]] = []
    for row in sorted(music, key=lambda item: item.sample_id):
        for index, snr in enumerate(snr_db):
            speaker = speakers[row.sample_id, index]
            sample_id = f"{row.sample_id}:speech-mix:{index}"
            filename = hashlib.sha256(sample_id.encode()).hexdigest() + ".wav"
            record = replace(
                row,
                sample_id=sample_id,
                path=f"media/{filename}",
                source=SOURCE,
                content_type="music+speech",
                parent_id=row.sample_id,
                identity_id=speaker.identity_id,
                sha256="unknown",
                license_id=f"{row.license_id} AND {speaker.license_id}",
            )
            details: dict[str, Any] = {
                "sample_id": sample_id,
                "music_id": row.sample_id,
                "music_sha256": row.sha256,
                "speech_id": speaker.sample_id,
                "speech_sha256": speaker.sha256,
                "speaker_id": speaker.identity_id,
                "split": row.split,
                "snr_db": snr,
            }
            try:
                if row.status != "available" or row.decode_status != "ok":
                    raise ValueError(f"music unavailable: {row.failure_reason}")  # noqa: TRY301 -- per-sample failure accounting
                audio, music_offset = _crop(music_root / row.path, music=True)
                voice, speech_offset = _crop(speech_root / speaker.path, music=False)
                waveform, gains = _mix(audio, voice, snr)
                sf.write(output / record.path, waveform, RATE, subtype="PCM_16")
                details.update(gains)
                details.update(
                    music_crop_seconds=music_offset, speech_crop_seconds=speech_offset
                )
                record = replace(
                    record,
                    sha256=file_sha256(output / record.path),
                    duration_seconds=4.0,
                    sample_rate=RATE,
                    container="wav",
                    codec="pcm_s16le",
                    decode_status="ok",
                )
            except (OSError, ValueError, sf.LibsndfileError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                record = replace(
                    record,
                    status="decode_failed",
                    decode_status="failed",
                    failure_reason=error,
                )
                details["error"] = error
            details["sha256"] = record.sha256
            mixed.append(record)
            provenance.append(details)
            if row.split == "train" and row.sample_id in main_ids:
                control.append(
                    replace(
                        row,
                        sample_id=f"{row.sample_id}:replay:{index}",
                        path=f"music/{row.path}",
                        parent_id=row.sample_id,
                    )
                )
    for name, records in {
        "mixed-manifest": [r for r in mixed if r.parent_id in main_ids],
        "mixed-dev-manifest": [
            r for r in mixed if r.parent_id in main_ids and r.split == "dev"
        ],
        "control-train-manifest": control,
        **(
            {
                "mixed-untrained-dev-manifest": [
                    r for r in mixed if r.parent_id in extra_ids
                ]
            }
            if untrained_manifest
            else {}
        ),
    }.items():
        write_manifest(records, output / f"{name}.jsonl")
    sources[SOURCE] = SourceRecord(
        source=SOURCE,
        url=sources[music[0].source].url,
        version="rms-mix-v1",
        license_id=mixed[0].license_id,
        license_url=sources[music[0].source].license_url,
        downloaded_at="derived",
        allowed_uses=allowed_uses,
        license_status="approved",
    )
    (output / "sources.jsonl").write_text(
        "".join(json.dumps(asdict(s)) + "\n" for s in sources.values())
    )
    (output / "provenance.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in provenance)
    )
    summary: dict[str, Any] = {
        "counts": dict(Counter(f"{r.split}:{r.label}" for r in mixed)),
        "status_counts": dict(Counter(r.status for r in mixed)),
        "control_count": len(control),
        "seed": seed,
        "snr_db": snr_db,
        "speech_pool_counts": {s: len(p) for s, p in pools.items()},
        "used_speakers": {
            s: sorted({r.identity_id for r in mixed if r.split == s}) for s in pools
        },
        "input_manifest_sha256": {
            "music": file_sha256(music_manifest),
            "speech": file_sha256(speech_manifest),
            **(
                {"untrained": file_sha256(untrained_manifest)}
                if untrained_manifest
                else {}
            ),
        },
        "processing": "4s centre crop; full-clip RMS music/speech SNR; symmetric peak 0.95; 16k mono PCM16",
        "competition_approved": False,
        "license_limit": "Research training only; AIME per-track/artist rights remain unconfirmed; source license intersection applies.",
    }
    atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "music-manifest",
        "music-root",
        "music-sources",
        "speech-manifest",
        "speech-root",
        "speech-sources",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--untrained-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snr-db", nargs=2, type=float, default=(-5, 5))
    print(
        json.dumps(prepare(**vars(parser.parse_args())), ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
