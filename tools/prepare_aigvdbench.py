"""Register the AIGVDBench first batch as isolated dev and holdout manifests.

Input is the per-file download log written while fetching selected members of
the pinned AIGVDBench release: real clips from the official val/test splits and
fakes from generators absent from current training and ViF-Bench. Real clips
and the named open-source derivatives group by the shared official Video_id's
YouTube ID, including different clips from the same parent. Closed-source
numeric filenames have no confirmed mapping and stay singleton groups. Identical bytes are merged
into one group, and a group must not span dev and holdout. Failed downloads stay
in the manifest as missing so coverage keeps them in the denominator. No media
is decoded, scored or copied here.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any

from deepfake_detection_archive.data.manifest import (
    UNKNOWN,
    SampleRecord,
    SourceRecord,
    write_manifest,
)


SOURCE = "aigvdbench-first-batch"
REPOSITORY = "https://huggingface.co/datasets/AIGVDBench/AIGVDBench"
LICENSE_ID = "CC-BY-4.0 AND LicenseRef-YouTube-underlying-unverified"
SPLITS = {"dev": "dev", "holdout": "holdout"}
YOUTUBE_DERIVED_GENERATORS = {
    "Open-Sora",
    "AccVideo",
    "AnimateDiff",
    "EasyAnimate",
    "IPOC",
    "Pyramid-Flow",
    "RepVideo",
    "VideoCrafter",
    "SEINE",
    "SVD",
}


def _find(parent: dict[str, str], key: str) -> str:
    while parent[key] != key:
        parent[key] = parent[parent[key]]
        key = parent[key]
    return key


def _groups(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Union declared groups with identical bytes; return member -> group id."""
    parent: dict[str, str] = {}

    def union(left: str, right: str) -> None:
        parent.setdefault(left, left)
        parent.setdefault(right, right)
        a, b = _find(parent, left), _find(parent, right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    by_hash: dict[str, str] = {}
    for row in rows:
        member = row["member_key"]
        union(member, row["declared_group"])
        digest = row.get("sha256")
        if row["status"] == "ok" and digest:
            if digest in by_hash:
                union(member, by_hash[digest])
            else:
                by_hash[digest] = member
    return {row["member_key"]: _find(parent, row["member_key"]) for row in rows}


def _declared_group(row: dict[str, Any]) -> str:
    if row["label"] == 0:
        return f"{SOURCE}:youtube:{row['youtube_id']}"
    if row.get("paired_real"):
        return f"{SOURCE}:youtube:{str(row['paired_real'])[:11]}"
    if row["generator"] in YOUTUBE_DERIVED_GENERATORS:
        return f"{SOURCE}:youtube:{Path(row['member']).name[:11]}"
    return f"{SOURCE}:{row['generator']}:{Path(row['member']).stem}"


def _relative_path(row: dict[str, Any]) -> str:
    name = Path(row["member"]).name
    if row["label"] == 0:
        return f"real/{name}"
    return f"fake/{row['generator']}/{name}"


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        if row.get("role") not in SPLITS or row.get("label") not in (0, 1):
            raise ValueError(f"invalid role or label for {row.get('member')}")
        if row["label"] == 0 and not row.get("youtube_id"):
            raise ValueError(f"real clip without youtube_id: {row['member']}")
        row["member_key"] = f"{row['zip']}::{row['member']}"
        row["declared_group"] = _declared_group(row)
    keys = Counter(row["member_key"] for row in rows)
    duplicates = sorted(key for key, count in keys.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate download rows: {duplicates[:5]}")
    return rows


def build_records(
    rows: list[dict[str, Any]],
) -> tuple[list[SampleRecord], dict[str, Any]]:
    """Return manifest records plus the isolation and duplicate report."""
    groups = _groups(rows)
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_group[groups[row["member_key"]]].add(row["role"])
    leaking = {group for group, roles in splits_by_group.items() if len(roles) > 1}
    sample_ids = {row["member_key"]: _sample_id(row) for row in rows}
    real_ids = {
        Path(row["member"]).name: sample_ids[row["member_key"]]
        for row in rows
        if row["label"] == 0
    }
    records: list[SampleRecord] = []
    excluded: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: sample_ids[item["member_key"]]):
        group = groups[row["member_key"]]
        probe = row.get("ffprobe") or {}
        status, reason = "available", UNKNOWN
        if group in leaking:
            status, reason = "excluded", "group spans dev and holdout"
            excluded.append(
                {"sample_id": sample_ids[row["member_key"]], "group": group}
            )
        elif row["status"] != "ok":
            status, reason = "missing", str(row.get("error") or "download failed")
        paired = row.get("paired_real")
        records.append(
            SampleRecord(
                sample_id=sample_ids[row["member_key"]],
                path=_relative_path(row),
                modality="video",
                label=row["label"],
                source=SOURCE,
                content_type="general_generated",
                generator=UNKNOWN if row["label"] == 0 else row["generator"],
                group_id=group,
                sha256=row.get("sha256") or UNKNOWN,
                license_id=LICENSE_ID,
                original_id=row["member"],
                parent_id=real_ids.get(paired, UNKNOWN) if paired else UNKNOWN,
                derivation_group_id=row["declared_group"],
                container="mp4",
                codec=probe.get("vcodec") or UNKNOWN,
                duration_seconds=probe.get("duration") or None,
                frame_rate=probe.get("fps") or None,
                split=SPLITS[row["role"]],
                status=status,
                failure_reason=reason,
            )
        )
    report = {
        "groups": len(set(groups.values())),
        "leaking_groups": sorted(leaking),
        "excluded": excluded,
        "byte_duplicate_members": _duplicates(rows),
    }
    return records, report


def _sample_id(row: dict[str, Any]) -> str:
    if row["label"] == 0:
        return f"aigvdb:real:{Path(row['member']).stem}"
    return f"aigvdb:{row['generator']}:{Path(row['member']).stem}"


def _duplicates(rows: list[dict[str, Any]]) -> list[list[str]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["status"] == "ok" and row.get("sha256"):
            by_hash[row["sha256"]].append(row["member_key"])
    return sorted(sorted(keys) for keys in by_hash.values() if len(keys) > 1)


def format_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe container differences per label and split without changing media."""
    summary: dict[str, Any] = {}
    for label in (0, 1):
        for role in SPLITS:
            probes = [
                row["ffprobe"]
                for row in rows
                if row["label"] == label
                and row["role"] == role
                and row["status"] == "ok"
                and row.get("ffprobe")
            ]
            if not probes:
                continue
            durations = [p["duration"] for p in probes if p.get("duration")]
            summary[f"label{label}-{role}"] = {
                "count": len(probes),
                "resolutions": dict(
                    Counter(f"{p['width']}x{p['height']}" for p in probes).most_common(
                        5
                    )
                ),
                "fps": dict(Counter(round(p["fps"], 2) for p in probes).most_common(5)),
                "duration_min": min(durations) if durations else None,
                "duration_median": median(durations) if durations else None,
                "duration_max": max(durations) if durations else None,
                "audio_share": sum(bool(p.get("has_audio")) for p in probes)
                / len(probes),
            }
    return summary


def prepare(
    download: Path, selection: Path, output: Path, downloaded_at: str
) -> dict[str, Any]:
    rows = load_rows(download)
    chosen = json.loads(selection.read_text(encoding="utf-8"))
    revision = str(chosen.get("revision") or UNKNOWN)
    records, report = build_records(rows)
    source = SourceRecord(
        source=SOURCE,
        url=REPOSITORY,
        version=revision,
        license_id=LICENSE_ID,
        license_url="https://creativecommons.org/licenses/by/4.0/",
        downloaded_at=downloaded_at,
        allowed_uses=("research", "evaluation"),
        license_status="unresolved",
    )
    output.mkdir(parents=True, exist_ok=True)
    digest = write_manifest(records, output / "manifest.jsonl")
    evaluation_manifests = {}
    for split in SPLITS:
        eligible = [
            record
            for record in records
            if record.split == split and record.status != "excluded"
        ]
        name = f"{split}-manifest.jsonl"
        evaluation_manifests[split] = {
            "path": name,
            "sha256": write_manifest(eligible, output / name),
            "samples": len(eligible),
            "groups": len({record.group_id for record in eligible}),
        }
    (output / "sources.jsonl").write_text(
        json.dumps(
            asdict(source) | {"allowed_uses": list(source.allowed_uses)},
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    counts = Counter(
        (record.split, record.label, record.generator, record.status)
        for record in records
    )
    summary = {
        "manifest_sha256": digest,
        "revision": revision,
        "samples": len(records),
        "counts": [
            {"split": split, "label": label, "generator": gen, "status": state, "n": n}
            for (split, label, gen, state), n in sorted(counts.items())
        ],
        "isolation": report,
        "evaluation_manifests": evaluation_manifests,
        "format": format_summary(rows),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--downloaded-at", required=True)
    args = parser.parse_args()
    try:
        summary = prepare(
            args.download, args.selection, args.output_dir, args.downloaded_at
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"aigvdbench import failed: {exc}\n")
    print(json.dumps({k: summary[k] for k in ("manifest_sha256", "samples")}))


if __name__ == "__main__":
    main()
