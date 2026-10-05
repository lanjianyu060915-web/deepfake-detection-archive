#!/usr/bin/env python3
"""Evidence-first audit of the canonical deepfake-related media in the workspace.

This tool deliberately does not infer truth from file names, folder names, visual
appearance, or model predictions.  It inventories canonical source holdings rather
than the many task/report copies of the same files elsewhere in the workspace.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import zipfile
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path, PurePosixPath

from PIL import Image

from common import IMAGE_SUFFIXES, sha256_bytes, sha256_file, write_csv, write_json


INVENTORY_FIELDS = [
    "candidate_id",
    "absolute_path",
    "relative_path",
    "file_name",
    "extension",
    "width",
    "height",
    "file_size",
    "sha256",
    "candidate_label",
    "label_evidence",
    "source_description",
    "generation_method",
    "is_usable",
    "exclusion_reason",
    "storage_type",
    "archive_path",
    "archive_member",
    "data_split",
    "data_role",
    "previously_exposed",
    "exclude_from_first_blind_test",
    "known_training_overlap",
    "perceptual_hash_dhash64",
]

OVERLAP_FIELDS = [
    "overlap_type",
    "candidate_id",
    "other_candidate_id",
    "candidate_path",
    "other_path",
    "sha256",
    "file_name",
    "perceptual_distance",
    "evidence",
    "action",
]


def dhash64(data: bytes) -> str:
    with Image.open(BytesIO(data)) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(gray.get_flattened_data())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return f"{value:016x}"


def _image_facts(data: bytes) -> tuple[int, int, str]:
    with Image.open(BytesIO(data)) as image:
        image.load()
        width, height = image.size
    return width, height, dhash64(data)


def _direct_policy(path: Path, workspace: Path) -> dict[str, str]:
    rel = path.relative_to(workspace).as_posix()
    package = "first_week/source_search_package/sample_data/"
    main = "first_week/data/"
    if rel.startswith(package + "query/"):
        return {
            "candidate_label": "FAKE",
            "label_evidence": "sample_data/ATTRIBUTION.md explicitly identifies all 10 query files as synthetic fixtures created by face-region replacement; ground_truth.csv maps each derivative to its source image.",
            "source_description": "Source-search synthetic query fixture; not an independent deepfake acceptance sample.",
            "generation_method": "face-region replacement synthetic fixture (implementation details unavailable)",
            "is_usable": "false",
            "exclusion_reason": "Development fixture; ATTRIBUTION.md explicitly says it is not for AIGC/deepfake evaluation.",
            "data_role": "Development_or_Exposed",
            "previously_exposed": "false",
            "exclude_from_first_blind_test": "true",
            "known_training_overlap": "false",
        }
    if rel.startswith(package + "source_library/"):
        return {
            "candidate_label": "UNKNOWN",
            "label_evidence": "Open Images provenance exists, but no evidence establishes unmanipulated deepfake REAL ground truth.",
            "source_description": "Open Images V7 validation source-search library image.",
            "generation_method": "unknown",
            "is_usable": "false",
            "exclusion_reason": "No evidence-backed deepfake REAL/FAKE truth label.",
            "data_role": "Candidate_only",
            "previously_exposed": "false",
            "exclude_from_first_blind_test": "true",
            "known_training_overlap": "false",
        }
    is_main_query = rel.startswith(main + "换脸查询图/")
    exposed = is_main_query and path.stem in {"query_001", "query_002", "query_003"}
    if is_main_query:
        reason = "Identity/source ground truth does not establish deepfake truth; no qualifying generation record was found."
        if exposed:
            reason += " This exact file was already run through the integrated deepfake branch."
        return {
            "candidate_label": "UNKNOWN",
            "label_evidence": "A/B identity and source mappings exist, but they are not REAL/FAKE truth; folder/file naming is not accepted as label evidence.",
            "source_description": "Existing A/B query image.",
            "generation_method": "unknown",
            "is_usable": "false",
            "exclusion_reason": reason,
            "data_role": "Development_or_Exposed" if exposed else "Candidate_only",
            "previously_exposed": "true" if exposed else "false",
            "exclude_from_first_blind_test": "true",
            "known_training_overlap": "false",
        }
    return {
        "candidate_label": "UNKNOWN",
        "label_evidence": "No evidence-backed deepfake REAL/FAKE truth was found; source/identity membership is not a deepfake label.",
        "source_description": "Existing identity or source-search library image.",
        "generation_method": "unknown",
        "is_usable": "false",
        "exclusion_reason": "No evidence-backed deepfake REAL/FAKE truth label.",
        "data_role": "Candidate_only",
        "previously_exposed": "false",
        "exclude_from_first_blind_test": "true",
        "known_training_overlap": "false",
    }


def _manifest_frame_labels(archive: zipfile.ZipFile) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for collection in ("data_formal", "data_demo"):
        member = f"data_all/{collection}/manifest.csv"
        text = archive.read(member).decode("utf-8-sig")
        reader = csv.DictReader(StringIO(text))
        for row in reader:
            label = row["label"].upper()
            for relative_frame in row["frame_paths"].split("|"):
                normalized = relative_frame.replace("\\", "/")
                full_member = f"data_all/{collection}/{normalized}"
                split = row["split"]
                if collection == "data_formal" and split == "train":
                    role = "training"
                elif collection == "data_formal" and split == "val":
                    role = "threshold_or_fusion_tuning"
                elif collection == "data_formal":
                    role = "prior_formal_metrics"
                else:
                    role = "development_demo"
                mapping[full_member] = {
                    "label": label,
                    "collection": collection,
                    "split": split,
                    "role": role,
                    "source_repo": row["source_repo"],
                    "source_filename": row["source_filename"],
                }
    return mapping


def collect(workspace: Path) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    direct_roots = [
        workspace / "first_week" / "data",
        workspace / "first_week" / "source_search_package" / "sample_data",
    ]
    for root in direct_roots:
        if not root.is_dir():
            continue
        for path in sorted((item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES), key=lambda item: str(item).casefold()):
            data = path.read_bytes()
            width, height, phash = _image_facts(data)
            policy = _direct_policy(path, workspace)
            records.append(
                {
                    "absolute_path": str(path.resolve()),
                    "relative_path": path.relative_to(workspace).as_posix(),
                    "file_name": path.name,
                    "extension": path.suffix.lower(),
                    "width": width,
                    "height": height,
                    "file_size": len(data),
                    "sha256": sha256_bytes(data),
                    **policy,
                    "storage_type": "file",
                    "archive_path": "",
                    "archive_member": "",
                    "data_split": "",
                    "perceptual_hash_dhash64": phash,
                }
            )

    zip_path = workspace / "AIGC检测" / "源数据及数据来源" / "data_all.zip"
    if zip_path.is_file():
        with zipfile.ZipFile(zip_path) as archive:
            labels = _manifest_frame_labels(archive)
            members = sorted(
                [
                    info
                    for info in archive.infolist()
                    if not info.is_dir()
                    and PurePosixPath(info.filename).suffix.lower() in IMAGE_SUFFIXES
                    and (
                        info.filename.startswith("data_all/data_formal/")
                        or info.filename.startswith("data_all/data_demo/")
                    )
                ],
                key=lambda info: info.filename.casefold(),
            )
            for info in members:
                data = archive.read(info)
                width, height, phash = _image_facts(data)
                truth = labels.get(info.filename)
                if truth:
                    label = truth["label"]
                    evidence = (
                        f"{truth['collection']}/manifest.csv explicitly labels the frame {label}; "
                        "the archived README assigns this collection to training, threshold/fusion tuning, development, or prior metrics."
                    )
                    generation = "not_applicable" if label == "REAL" else "labeled deepfake from TonyStark03/deepfake; exact generation algorithm unknown"
                    split = truth["split"]
                    role = truth["role"]
                    known_training = "true" if role == "training" else "false"
                    description = f"Archived {truth['collection']} frame from {truth['source_repo']} / {truth['source_filename']}."
                    exclusion = f"Known baseline {role} data; not independent and must not enter first Core Blind."
                else:
                    label = "UNKNOWN"
                    evidence = "stress_manifest describes a stress condition but supplies no REAL/FAKE truth label."
                    generation = "unknown"
                    split = "stress"
                    role = "development_stress"
                    known_training = "false"
                    description = "Archived stress-test image."
                    exclusion = "No evidence-backed truth label and already part of development stress material."
                pseudo_path = f"zip://{zip_path.resolve().as_posix()}!/{info.filename}"
                records.append(
                    {
                        "absolute_path": pseudo_path,
                        "relative_path": f"{zip_path.relative_to(workspace).as_posix()}!/{info.filename}",
                        "file_name": PurePosixPath(info.filename).name,
                        "extension": PurePosixPath(info.filename).suffix.lower(),
                        "width": width,
                        "height": height,
                        "file_size": len(data),
                        "sha256": sha256_bytes(data),
                        "candidate_label": label,
                        "label_evidence": evidence,
                        "source_description": description,
                        "generation_method": generation,
                        "is_usable": "false",
                        "exclusion_reason": exclusion,
                        "storage_type": "zip_member",
                        "archive_path": str(zip_path.resolve()),
                        "archive_member": info.filename,
                        "data_split": split,
                        "data_role": role,
                        "previously_exposed": "true",
                        "exclude_from_first_blind_test": "true",
                        "known_training_overlap": known_training,
                        "perceptual_hash_dhash64": phash,
                    }
                )
    for index, record in enumerate(records, 1):
        record["candidate_id"] = f"C{index:06d}"
    return records


def overlap_rows(records: list[dict[str, str | int]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    by_sha: dict[str, list[dict[str, str | int]]] = collections.defaultdict(list)
    by_name: dict[str, list[dict[str, str | int]]] = collections.defaultdict(list)
    by_path: dict[str, list[dict[str, str | int]]] = collections.defaultdict(list)
    for record in records:
        by_sha[str(record["sha256"])].append(record)
        by_name[str(record["file_name"]).casefold()].append(record)
        by_path[str(record["absolute_path"]).casefold()].append(record)
        if str(record["data_role"]) in {"training", "threshold_or_fusion_tuning", "prior_formal_metrics", "development_demo"}:
            output.append(
                {
                    "overlap_type": "KNOWN_BASELINE_DATA_MEMBERSHIP",
                    "candidate_id": record["candidate_id"],
                    "other_candidate_id": "KNOWN_BASELINE_CORPUS",
                    "candidate_path": record["relative_path"],
                    "other_path": record["archive_member"],
                    "sha256": record["sha256"],
                    "file_name": record["file_name"],
                    "perceptual_distance": "0",
                    "evidence": f"Archived README declares role={record['data_role']}; manifest split={record['data_split']}.",
                    "action": "EXCLUDE_FROM_CORE_BLIND",
                }
            )
    for _, group in sorted(by_path.items()):
        if len(group) < 2:
            continue
        anchor = group[0]
        for other in group[1:]:
            output.append(
                {
                    "overlap_type": "PATH_DUPLICATE",
                    "candidate_id": anchor["candidate_id"],
                    "other_candidate_id": other["candidate_id"],
                    "candidate_path": anchor["relative_path"],
                    "other_path": other["relative_path"],
                    "sha256": anchor["sha256"],
                    "file_name": anchor["file_name"],
                    "perceptual_distance": "0" if anchor["sha256"] == other["sha256"] else "",
                    "evidence": "Canonical absolute/archive-member path appears more than once.",
                    "action": "COUNT_ONCE",
                }
            )
    for sha, group in sorted(by_sha.items()):
        if len(group) < 2:
            continue
        anchor = group[0]
        for other in group[1:]:
            output.append(
                {
                    "overlap_type": "SHA256_EXACT_DUPLICATE",
                    "candidate_id": anchor["candidate_id"],
                    "other_candidate_id": other["candidate_id"],
                    "candidate_path": anchor["relative_path"],
                    "other_path": other["relative_path"],
                    "sha256": sha,
                    "file_name": anchor["file_name"],
                    "perceptual_distance": "0",
                    "evidence": "Identical SHA-256.",
                    "action": "COUNT_ONCE_AND_EXCLUDE_KNOWN_DATA",
                }
            )
    exact_pairs = {
        tuple(sorted((str(row["candidate_id"]), str(row["other_candidate_id"]))))
        for row in output if row["overlap_type"] == "SHA256_EXACT_DUPLICATE"
    }
    for _, group in sorted(by_name.items()):
        if len(group) < 2:
            continue
        anchor = group[0]
        for other in group[1:]:
            pair = tuple(sorted((str(anchor["candidate_id"]), str(other["candidate_id"]))))
            if pair in exact_pairs:
                continue
            output.append(
                {
                    "overlap_type": "FILENAME_DUPLICATE_DIFFERENT_CONTENT",
                    "candidate_id": anchor["candidate_id"],
                    "other_candidate_id": other["candidate_id"],
                    "candidate_path": anchor["relative_path"],
                    "other_path": other["relative_path"],
                    "sha256": anchor["sha256"],
                    "file_name": anchor["file_name"],
                    "perceptual_distance": "",
                    "evidence": "Case-insensitive file name match; SHA-256 differs.",
                    "action": "MANUAL_REVIEW",
                }
            )

    # Perceptual comparison is scoped to archive-versus-direct candidates: this is
    # the leakage question, while within-video adjacent frames are correlated by design.
    direct = [row for row in records if row["storage_type"] == "file"]
    archived = [row for row in records if row["storage_type"] == "zip_member"]
    for left in direct:
        left_hash = int(str(left["perceptual_hash_dhash64"]), 16)
        for right in archived:
            if left["sha256"] == right["sha256"]:
                continue
            distance = (left_hash ^ int(str(right["perceptual_hash_dhash64"]), 16)).bit_count()
            if distance <= 4:
                output.append(
                    {
                        "overlap_type": "DHASH64_NEAR_DUPLICATE_CANDIDATE",
                        "candidate_id": left["candidate_id"],
                        "other_candidate_id": right["candidate_id"],
                        "candidate_path": left["relative_path"],
                        "other_path": right["relative_path"],
                        "sha256": left["sha256"],
                        "file_name": left["file_name"],
                        "perceptual_distance": distance,
                        "evidence": "64-bit dHash Hamming distance <= 4; this is a review lead, not proof of identity.",
                        "action": "MANUAL_REVIEW_AND_EXCLUDE_UNTIL_RESOLVED",
                    }
                )
    return output


def full_workspace_stats(workspace: Path) -> dict[str, int]:
    images = sorted(
        path for path in workspace.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and "deepfake_acceptance_v2" not in path.parts
    )
    hashes: collections.Counter[str] = collections.Counter()
    for path in images:
        try:
            hashes[sha256_file(path)] += 1
        except OSError:
            continue
    duplicate_groups = sum(count > 1 for count in hashes.values())
    paths_in_duplicate_groups = sum(count for count in hashes.values() if count > 1)
    return {
        "image_path_count": len(images),
        "unique_sha256_count": len(hashes),
        "duplicate_sha256_group_count": duplicate_groups,
        "paths_in_duplicate_groups": paths_in_duplicate_groups,
        "maximum_duplicate_multiplicity": max(hashes.values(), default=0),
    }


def archive_integrity(zip_path: Path) -> dict[str, object]:
    if not zip_path.is_file():
        return {"status": "NOT_FOUND", "archive": str(zip_path)}
    member_hashes: dict[str, str] = {}
    collections_summary: dict[str, object] = {}
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        artifact_members = [
            "data_all/data_formal/manifest.csv",
            "data_all/data_demo/manifest.csv",
            "data_all/deepfake_media_kit/README_数据获取与目录说明.md",
            "data_all/deepfake_media_kit/prepare_deepfake_course_data.py",
        ]
        for member in artifact_members:
            member_hashes[member] = sha256_bytes(archive.read(member))
        for collection in ("data_formal", "data_demo"):
            manifest_member = f"data_all/{collection}/manifest.csv"
            rows = list(csv.DictReader(StringIO(archive.read(manifest_member).decode("utf-8-sig"))))
            missing_videos: list[str] = []
            mismatched_videos: list[str] = []
            missing_frames: list[str] = []
            frame_count = 0
            for row in rows:
                video_member = f"data_all/{collection}/{row['local_video'].replace(chr(92), '/')}"
                if video_member not in names:
                    missing_videos.append(video_member)
                elif sha256_bytes(archive.read(video_member)) != row["video_sha256"]:
                    mismatched_videos.append(video_member)
                for frame in row["frame_paths"].split("|"):
                    frame_member = f"data_all/{collection}/{frame.replace(chr(92), '/')}"
                    frame_count += 1
                    if frame_member not in names:
                        missing_frames.append(frame_member)
            collections_summary[collection] = {
                "manifest_rows": len(rows),
                "video_paths_checked": len(rows),
                "frame_paths_checked": frame_count,
                "missing_video_count": len(missing_videos),
                "video_sha256_mismatch_count": len(mismatched_videos),
                "missing_frame_count": len(missing_frames),
                "status": "PASS" if not (missing_videos or mismatched_videos or missing_frames) else "FAIL",
            }
    return {
        "status": "PASS" if all(item["status"] == "PASS" for item in collections_summary.values()) else "FAIL",
        "archive": str(zip_path.resolve()),
        "archive_size_bytes": zip_path.stat().st_size,
        "archive_sha256": sha256_file(zip_path),
        "member_sha256": member_hashes,
        "collections": collections_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--overlap", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--skip-full-workspace-stats", action="store_true")
    args = parser.parse_args()
    try:
        records = collect(args.workspace.resolve())
        overlaps = overlap_rows(records)
        workspace_stats = {} if args.skip_full_workspace_stats else full_workspace_stats(args.workspace.resolve())
        write_csv(args.inventory, INVENTORY_FIELDS, records)
        write_csv(args.overlap, OVERLAP_FIELDS, overlaps)
        labels = collections.Counter(str(row["candidate_label"]) for row in records)
        eligible = collections.Counter(str(row["candidate_label"]) for row in records if row["is_usable"] == "true")
        archive_path = args.workspace.resolve() / "AIGC检测" / "源数据及数据来源" / "data_all.zip"
        summary = {
            "schema_version": "D-v2-candidate-audit-1.0",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "workspace": str(args.workspace.resolve()),
            "scope": [
                "first_week/data canonical A/B source holdings",
                "first_week/source_search_package/sample_data canonical source-search fixtures",
                "AIGC检测/源数据及数据来源/data_all.zip formal/demo image holdings",
            ],
            "candidate_path_count": len(records),
            "candidate_unique_sha256_count": len({str(row["sha256"]) for row in records}),
            "candidate_label_counts": {key: labels.get(key, 0) for key in ("REAL", "FAKE", "UNKNOWN")},
            "core_blind_eligible_counts": {key: eligible.get(key, 0) for key in ("REAL", "FAKE")},
            "previously_exposed_count": sum(row["previously_exposed"] == "true" for row in records),
            "known_training_membership_count": sum(row["known_training_overlap"] == "true" for row in records),
            "overlap_row_counts": dict(sorted(collections.Counter(str(row["overlap_type"]) for row in overlaps).items())),
            "training_data_overlap_status": "UNKNOWN",
            "training_data_overlap_reason": "The local fine-tune/development snapshot is exactly checkable and excluded, but complete sample-level inventories for all upstream/pretrained training sources are unavailable.",
            "full_workspace_duplicate_statistics": workspace_stats,
            "archive_source_integrity": archive_integrity(archive_path),
            "real_model_executed": False,
        }
        write_json(args.summary, summary)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"status": "success", **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
