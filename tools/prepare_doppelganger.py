"""Import a fixed CC0 environmental subset and its published synthetic twins.

This is a new research split, not the upstream retrieval benchmark. No model is
trained or scored. Competition use remains subject to source confirmation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
from scipy.signal import resample_poly  # type: ignore[import-untyped]

from deepfake_detection_archive.data.audit import audit_manifest
from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    SourceRecord,
    file_sha256,
    write_manifest,
)
from deepfake_detection_archive.data.split import assign_splits, connected_groups
from deepfake_detection_archive.tracking import atomic_json


REPO = "elliottash/doppelganger"
REVISION = "eb6170777c1ee5ff68982769b003ab98c3ebf78b"
SYNTHETIC_REVISION = "801be753d84a470a8f94e233e7024a8d306f9842"
REAL_REPO = "Fhrozen/FSD50k"
REAL_REVISION = "ccf1acaff12f3f4a4c10052dddafbb6d22152c9b"
METADATA_URL = (
    f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/mseb/test.csv"
)
METADATA_SHA256 = "fdbd6985f824a01eeb5c5d3b837e699a2558323599a6f390189b99978ac4abdc"
SOURCE_INFO_SHA256 = "ced921d00c4f7ace86b92c259689a1a40a8452808e93235021165ad2d8b77e6c"
EVENTS = ("ANML", "BIRD", "FIRE", "FOOT", "RAIN", "WATR", "WTHR", "WIND")
# Evidence/reasons: docs/experiments/nonspeech-20261011/source-review.json.
# Exclude both members; uncertain recordings are never relabelled as fake.
EXCLUDED_SOURCE_IDS = {
    "18443",
    "134742",
    "135034",
    "146932",
    "156946",
    "176769",
    "181896",
    "188041",
    "188042",
    "196382",
    "209661",
    "210314",
    "233261",
    "256452",
    "263042",
    "275154",
    "328842",
    "352609",
    "366892",
    "395684",
    "397100",
}
SOURCE = "doppelganger-fsd50k-cc0-environment-research"
LICENSE = "CC0-1.0 AND CC-BY-4.0 AND LicenseRef-Stability-AI-Community"
RATE = 16000
SAMPLES = 64600


def _download(url: str, path: Path) -> None:
    """Bounded retries; partial bytes never become a usable media file."""
    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            with (
                urllib.request.urlopen(url, timeout=60) as response,  # noqa: S310 -- pinned HTTPS inputs
                temporary.open("wb") as stream,
            ):
                while block := response.read(1024 * 1024):
                    stream.write(block)
                    if stream.tell() > 100 * 1024**2:
                        raise ValueError("source file exceeds 100 MiB")
            temporary.replace(path)
        except (OSError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
        else:
            return
        finally:
            temporary.unlink(missing_ok=True)


def select_pairs(
    raw: bytes, source_info_raw: bytes, per_event: int, seed: int
) -> list[dict[str, str]]:
    if hashlib.sha256(raw).hexdigest() != METADATA_SHA256:
        raise ValueError("metadata differs from the pinned revision")
    if per_event < 1:
        raise ValueError("per_event must be positive")
    if hashlib.sha256(source_info_raw).hexdigest() != SOURCE_INFO_SHA256:
        raise ValueError("FSD source metadata differs from the pinned revision")
    source_info = json.loads(source_info_raw)
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    selected: list[dict[str, str]] = []
    for event in EVENTS:
        candidates = [
            row
            for row in rows
            if row["event_id"] == event
            and row["source_clip_id"] not in EXCLUDED_SOURCE_IDS
            and row["real_license_url"].removeprefix("https:").removeprefix("http:")
            == "//creativecommons.org/publicdomain/zero/1.0/"
        ]
        candidates.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['pair_id']}".encode()
            ).hexdigest()
        )
        selected.extend(candidates[:per_event])
    if not selected:
        raise ValueError("no eligible CC0 environmental pairs")
    seen: set[str] = set()
    for row in selected:
        pair_id = row["pair_id"]
        if not pair_id.isdigit() or pair_id in seen or not row["real_uploader"]:
            raise ValueError("invalid/duplicate pair ID or missing uploader")
        seen.add(pair_id)
        original = source_info[row["source_clip_id"]]
        if (
            original["license"] != row["real_license_url"]
            or original["uploader"] != row["real_uploader"]
            or original["title"] != row["real_title"]
        ):
            raise ValueError("FSD attribution or license disagrees with pair metadata")
        row["original_description"] = original["description"]
        row["original_tags_json"] = json.dumps(original["tags"], ensure_ascii=False)
        if (
            row["source_dataset"] != "FSD50K"
            or row["real_repo_id"] != REAL_REPO
            or row["real_revision"] != REAL_REVISION
            or row["synthetic_repo_id"] != REPO
            or row["synthetic_revision"] != SYNTHETIC_REVISION
            or row["synthetic_license"] != "Stability AI Community License"
        ):
            raise ValueError("unexpected source, revision or license")
        for kind, repo, revision in (
            ("real", REAL_REPO, REAL_REVISION),
            ("synthetic", REPO, SYNTHETIC_REVISION),
        ):
            path = row[f"{kind}_audio_path"]
            if Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError("unsafe source path")
            expected = (
                f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
            )
            if row[f"{kind}_audio_url"] != expected:
                raise ValueError("audio URL does not match pinned source")
    return selected


def _prepare_one(
    row: dict[str, str], label: int, output: Path
) -> tuple[SampleRecord, dict[str, Any]]:
    kind = "real" if label == 0 else "synthetic"
    pair_id = row["pair_id"]
    sample_id = f"doppelganger:{pair_id}:{label}"
    relative = f"media/{kind}/{pair_id}.wav"
    raw_path = output / f"raw/{kind}/{pair_id}.wav"
    provenance: dict[str, Any] = {
        **row,
        "sample_id": sample_id,
        "label": label,
        "raw_path": raw_path.relative_to(output).as_posix(),
    }
    record = SampleRecord(
        sample_id=sample_id,
        path=relative,
        modality="audio",
        label=label,
        source=SOURCE,
        content_type="environment",
        generator="unknown" if label == 0 else "stable-audio-open-1.0",
        group_id=f"freesound-uploader:{row['real_uploader'].casefold()}",
        original_id=f"freesound:{row['source_clip_id']}",
        parent_id="unknown" if label == 0 else f"doppelganger:{pair_id}:0",
        derivation_group_id=f"freesound:{row['source_clip_id']}",
        license_id=LICENSE,
    )
    try:
        _download(row[f"{kind}_audio_url"], raw_path)
        provenance["raw_sha256"] = file_sha256(raw_path)
        provenance["raw_bytes"] = raw_path.stat().st_size
        info = sf.info(raw_path)
        waveform, rate = sf.read(raw_path, dtype="float32", always_2d=True)
        if len(waveform) == 0 or not np.isfinite(waveform).all():
            raise ValueError("empty or non-finite source audio")  # noqa: TRY301 -- record sample failure
        provenance["raw_decode_status"] = "ok"
        provenance["original"] = {
            "sample_rate": rate,
            "channels": info.channels,
            "frames": len(waveform),
            "duration_seconds": len(waveform) / rate,
            "format": info.format,
            "subtype": info.subtype,
        }
        if label == 0 and len(waveform) < 5 * rate:
            return replace(
                record,
                status="excluded",
                failure_reason="real_shorter_than_upstream_5s_conditioning_window",
            ), provenance
        # The published conditioning pipeline center-crops the real audio.
        # Both labels get the same downmix/resampling/center crop/PCM encoding.
        mono = waveform.mean(axis=1)
        divisor = math.gcd(rate, RATE)
        mono = resample_poly(mono, RATE // divisor, rate // divisor)
        if len(mono) < SAMPLES:
            return replace(
                record,
                status="excluded",
                failure_reason="source_shorter_than_64600_samples_at_16khz",
            ), provenance
        start = (len(mono) - SAMPLES) // 2
        segment = mono[start : start + SAMPLES]
        peak = float(np.max(np.abs(segment)))
        provenance["pre_normalization_peak"] = peak
        provenance["pre_normalization_rms"] = float(np.sqrt(np.mean(segment**2)))
        if peak == 0:
            return replace(
                record, status="excluded", failure_reason="silent_center_segment"
            ), provenance
        # The generator peak-normalizes its outputs. Apply the same final gain
        # rule to both labels, avoiding a known one-sided normalization shortcut.
        segment = segment * (0.95 / peak)
        provenance["normalized_rms"] = float(np.sqrt(np.mean(segment**2)))
        destination = output / relative
        sf.write(destination, segment, RATE, subtype="PCM_16")
        return replace(
            record,
            sha256=file_sha256(destination),
            container="wav",
            codec="pcm_s16le",
            duration_seconds=SAMPLES / RATE,
            sample_rate=RATE,
            decode_status="ok",
        ), provenance
    except (OSError, ValueError, sf.LibsndfileError) as exc:
        provenance["error"] = f"{type(exc).__name__}: {exc}"
        return replace(
            record,
            status="decode_failed" if raw_path.exists() else "missing",
            decode_status="failed" if raw_path.exists() else "unknown",
            failure_reason=provenance["error"],
        ), provenance


def prepare(  # noqa: PLR0913 -- CLI input paths and import parameters
    metadata: Path,
    source_info: Path,
    output: Path,
    *,
    per_event: int = 40,
    seed: int = 42,
    workers: int = 4,
) -> dict[str, Any]:
    started = time.monotonic()
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    raw = metadata.read_bytes()
    source_info_raw = source_info.read_bytes()
    pairs = select_pairs(raw, source_info_raw, per_event, seed)
    output.mkdir(parents=True, exist_ok=False)
    (output / "upstream-test.csv").write_bytes(raw)
    (output / "fsd50k-eval-info.json").write_bytes(source_info_raw)
    for kind in ("real", "synthetic"):
        (output / "raw" / kind).mkdir(parents=True)
        (output / "media" / kind).mkdir(parents=True)
    tasks = [(row, label) for row in pairs for label in (0, 1)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(lambda task: _prepare_one(task[0], task[1], output), tasks)
        )
    records = [item[0] for item in results]
    # Keep pair balance. Failed/short entries and their partners remain in the
    # inventory, but neither partner enters the training manifest.
    ineligible = {
        record.derivation_group_id for record in records if record.status != "available"
    }
    records = [
        replace(record, status="excluded", failure_reason="paired_source_unavailable")
        if record.status == "available" and record.derivation_group_id in ineligible
        else record
        for record in records
    ]
    eligible = [record for record in records if record.status == "available"]
    write_manifest(records, output / "inventory.jsonl")
    with (output / "provenance.jsonl").open("w", encoding="utf-8") as stream:
        for _, provenance in results:
            stream.write(json.dumps(provenance, ensure_ascii=False) + "\n")
    split_records = assign_splits(eligible, seed=seed)
    by_id = {record.sample_id: record for record in split_records}
    inventory = [by_id.get(record.sample_id, record) for record in records]
    inventory_hash = write_manifest(inventory, output / "inventory.jsonl")
    manifest_hash = write_manifest(split_records, output / "manifest.jsonl")
    source = SourceRecord(
        source=SOURCE,
        url=f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}",
        version=REVISION,
        license_id=LICENSE,
        license_url="https://stability.ai/community-license-agreement",
        downloaded_at=datetime.now(UTC).isoformat(),
        allowed_uses=("training", "research"),
        license_status="approved",
        archive_sha256=METADATA_SHA256,
    )
    (output / "sources.jsonl").write_text(
        json.dumps(asdict(source), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    audit = audit_manifest(split_records, output, {SOURCE: source})
    atomic_json(output / "audit.json", audit.to_dict())
    summary: dict[str, Any] = {
        "source": SOURCE,
        "metadata_sha256": METADATA_SHA256,
        "source_info_sha256": SOURCE_INFO_SHA256,
        "excluded_source_ids": sorted(EXCLUDED_SOURCE_IDS),
        "inventory_sha256": inventory_hash,
        "manifest_sha256": manifest_hash,
        "seed": seed,
        "max_pairs_per_event": per_event,
        "selected_pairs": len(pairs),
        "selected_events": dict(Counter(row["event_id"] for row in pairs)),
        "status_counts": dict(Counter(record.status for record in inventory)),
        "failure_reasons": dict(
            Counter(r.failure_reason for r in inventory if r.status != "available")
        ),
        "eligible_samples": len(split_records),
        "split_label_counts": dict(
            Counter(f"{r.split}/{r.label}" for r in split_records)
        ),
        "groups": len(set(connected_groups(split_records).values())),
        "raw_bytes": sum(item[1].get("raw_bytes", 0) for item in results),
        "audit_passed": audit.passed,
        "source_decode_coverage": sum(
            p.get("raw_decode_status") == "ok" for _, p in results
        )
        / len(records),
        "normalized_coverage": len(split_records) / len(records),
        "preprocessing": "mono, polyphase 16kHz, center 64600 samples, peak 0.95, PCM16 WAV",
        "training_license_scope": "internal noncommercial detection research only",
        "competition_approved": False,
        "generator_weights_revision": "unknown",
        "upstream_split": "mseb/test; re-partitioned for binary research pilot",
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(output / "summary.json", summary)
    if not audit.passed:
        raise ValueError("prepared manifest did not pass audit; see audit.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--source-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-event", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.metadata,
                args.source_info,
                args.output,
                per_event=args.per_event,
                seed=args.seed,
                workers=args.workers,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
