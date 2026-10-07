"""Verify fixed ViF source media and import only Panda pairs for local research.

Kinetics member bodies are never opened. Evidence paths supplied to the import
command must already have been reviewed for the naming contract and local
noncommercial research permissions; file hashes preserve that review evidence.
This tool performs no media decoding, training, download, or competition approval.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
import zipfile
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, urlparse

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    SourceRecord,
    file_sha256,
    write_manifest,
)
from deepfake_detection_archive.tracking import atomic_json


REVISION = "c9e9b30e47e216f72c50664c43750674a79c0ded"
REPO = "JoeLeelyf/ViF-CoT-4K"
ARCHIVE_BYTES = 1294907616
ARCHIVE_SHA256 = "6f47fc9935c6545c7c237a6d15ef8c0e7e26a6826160bb54a6f2f5e83d2c6fae"
NORMALIZED_SHA256 = "33218338a98e23925beeed70ea0bbcd025eb625cf981b767ba6560509a4d7e11"
NORMALIZED_PATH = "source_videos/normalized.json"
EXPECTED_PAIRS = 3502
EXPECTED_PANDA_PAIRS = 2306
MAX_MEMBER_BYTES = 100 * 1024**2
MAX_EXTRACT_BYTES = 8 * 1024**3
CACHE_BUDGET = 20 * 1024**3
ASSET_BUDGET = 100 * 1024**3
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "vif-panda70m-local-noncommercial-research"
LICENSE_ID = "ViF-Panda70M-HD-VILA-local-noncommercial-research"
SOURCE_URL = f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}"
NAMING_CONTRACT = "panda70m-explicit-url-timestamp-map-v1"
SUSPECTED_RULE = "suspected_filename_component:v1"


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate metadata key: {key}")
        result[key] = value
    return result


def _safe_path(name: str) -> tuple[str, ...]:
    parts = tuple(name.removesuffix("/").split("/"))
    if (
        not name
        or PurePosixPath(name).is_absolute()
        or "\\" in name
        or any(not part or part in (".", "..") or ":" in part for part in parts)
        or any(ord(char) < 32 for char in name)
    ):
        raise ValueError(f"unsafe archive/metadata path: {name!r}")
    return parts


def _directory(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    result: dict[str, zipfile.ZipInfo] = {}
    names: dict[str, bool] = {}
    entries = archive.infolist()
    if len(entries) > 20000:
        raise ValueError("ZIP directory exceeds entry budget")
    for info in entries:
        parts = _safe_path(info.filename)
        key = "/".join(parts).casefold()
        if key in names or info.filename != info.orig_filename:
            raise ValueError("duplicate, case-colliding or truncated ZIP path")
        names[key] = info.is_dir()
        mode = stat.S_IFMT(info.external_attr >> 16)
        if mode not in (0, stat.S_IFREG, stat.S_IFDIR) or (
            mode == stat.S_IFDIR and not info.is_dir()
        ):
            raise ValueError("symlink or special archive entry")
        if info.flag_bits & 1:
            raise ValueError("encrypted archive entry")
        if not info.is_dir():
            result[info.filename] = info
    for name in names:
        parts = tuple(name.split("/"))
        if any(
            names.get("/".join(parts[:end])) is False for end in range(1, len(parts))
        ):
            raise ValueError("archive file/directory collision")
    return result


def _metadata(
    archive: zipfile.ZipFile, directory: dict[str, zipfile.ZipInfo]
) -> list[dict[str, str]]:
    if (
        NORMALIZED_PATH not in directory
        or directory[NORMALIZED_PATH].file_size > 2 * 1024**2
    ):
        raise ValueError("missing or oversized normalized metadata")
    raw = archive.read(directory[NORMALIZED_PATH])
    if hashlib.sha256(raw).hexdigest() != NORMALIZED_SHA256:
        raise ValueError("normalized metadata checksum mismatch")
    rows = json.loads(raw, object_pairs_hook=_json_pairs)
    if not isinstance(rows, list) or len(rows) != EXPECTED_PAIRS:
        raise ValueError("unexpected normalized pair count")
    required = {"video_id", "video_path", "aigc_model", "aigc_video_path"}
    media: set[str] = set()
    panda: list[dict[str, str]] = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != required
            or any(not isinstance(value, str) or not value for value in row.values())
        ):
            raise ValueError("invalid normalized pair schema")
        real, fake = _safe_path(row["video_path"]), _safe_path(row["aigc_video_path"])
        if (
            len(real) != 5
            or len(fake) != 5
            or real[0] != "source_videos"
            or real[1] not in ("Panda-70M", "kinetics")
            or real[2] != "real"
            or fake != (real[0], real[1], "fake", real[3], real[4])
            or real[3] != row["aigc_model"]
            or real[4] != row["video_id"] + ".mp4"
        ):
            raise ValueError("metadata pair paths/model/video ID disagree")
        for key in ("video_path", "aigc_video_path"):
            path = row[key]
            if path in media or path not in directory:
                raise ValueError("duplicate or absent paired media")
            media.add(path)
        if real[1] == "Panda-70M":
            panda.append(row)
    if set(directory) != media | {NORMALIZED_PATH}:
        raise ValueError("archive members differ from normalized media manifest")
    if len(panda) != EXPECTED_PANDA_PAIRS:
        raise ValueError("unexpected Panda pair count")
    return panda


def inspect_archive(path: Path) -> dict[str, object]:
    """Authenticate complete source ZIP and inspect metadata without opening media."""
    if path.stat().st_size != ARCHIVE_BYTES or file_sha256(path) != ARCHIVE_SHA256:
        raise ValueError("archive failed fixed revision size/LFS SHA-256 verification")
    with zipfile.ZipFile(path) as archive:
        directory = _directory(archive)
        panda = _metadata(archive, directory)
        selected = [
            directory[row[key]]
            for row in panda
            for key in ("video_path", "aigc_video_path")
        ]
        if any(
            info.file_size <= 0 or info.file_size > MAX_MEMBER_BYTES
            for info in selected
        ):
            raise ValueError("Panda member exceeds single-file budget or is empty")
        selected_bytes = sum(info.file_size for info in selected)
        if selected_bytes > MAX_EXTRACT_BYTES:
            raise ValueError("Panda media exceed extraction budget")
    return {
        "verified_at": datetime.now(UTC).isoformat(),
        "repository": REPO,
        "revision": REVISION,
        "archive_sha256": ARCHIVE_SHA256,
        "archive_bytes": ARCHIVE_BYTES,
        "normalized_sha256": NORMALIZED_SHA256,
        "all_pair_count": EXPECTED_PAIRS,
        "panda_pair_count": len(panda),
        "panda_media_count": len(selected),
        "panda_media_bytes": selected_bytes,
        "generators": dict(sorted(Counter(row["aigc_model"] for row in panda).items())),
        "kinetics_member_contents_read": False,
        "media_decoded": False,
    }


def _tree_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _budget(archive: Path, selected_bytes: int) -> None:
    cache = WORKSPACE_ROOT / "cache"
    cache_bytes = _tree_bytes(cache)
    external_archive = (
        0
        if archive.resolve().is_relative_to(cache.resolve())
        else archive.stat().st_size
    )
    if cache_bytes + external_archive + selected_bytes > CACHE_BUDGET:
        raise ValueError(
            "archive plus selected extraction exceed 20 GiB cache/acquisition budget"
        )
    total = sum(
        _tree_bytes(WORKSPACE_ROOT / name)
        for name in ("cache", "data", "models", "outputs")
    )
    if total + external_archive + selected_bytes > ASSET_BUDGET:
        raise ValueError("new asset total would exceed 100 GiB budget")


def _evidence(paths: list[Path]) -> dict[str, str]:
    if not paths or any(
        not path.is_file() or path.stat().st_size == 0 for path in paths
    ):
        raise ValueError(
            "nonempty reviewed naming and license evidence files are required"
        )
    return {str(path.resolve()): file_sha256(path) for path in paths}


def _publish(stage: Path, target: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    try:
        rename = library.renameat2
    except AttributeError as exc:
        raise RuntimeError(
            "atomic no-overwrite publication requires Linux renameat2"
        ) from exc
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage), -100, os.fsencode(target), 1):
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(target))


def _copy(archive: zipfile.ZipFile, info: zipfile.ZipInfo, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    copied = 0
    with archive.open(info) as source, destination.open("xb") as target:
        while chunk := source.read(1024**2):
            copied += len(chunk)
            if copied > min(info.file_size, MAX_MEMBER_BYTES):
                raise ValueError("extracted media exceed verified budget")
            digest.update(chunk)
            target.write(chunk)
        target.flush()
        os.fsync(target.fileno())
    if copied != info.file_size:
        raise ValueError("extracted media size mismatch")
    return digest.hexdigest()


def _source_map(
    path: Path, panda: list[dict[str, str]], evidence_sha256: str
) -> dict[str, str]:
    """Require explicit clip -> original URL/timestamps from primary metadata."""
    value = json.loads(path.read_text(), object_pairs_hook=_json_pairs)
    required = {"schema_version", "metadata_source_url", "metadata_sha256", "clips"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("invalid source mapping schema")
    if (
        value["schema_version"] != 1
        or not isinstance(value["metadata_source_url"], str)
        or not value["metadata_source_url"].startswith("https://")
    ):
        raise ValueError("source mapping requires primary metadata URL and version")
    if (
        not isinstance(value["metadata_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", value["metadata_sha256"]) is None
    ):
        raise ValueError("source mapping requires primary metadata SHA-256")
    if value["metadata_sha256"] != evidence_sha256:
        raise ValueError("source mapping metadata hash differs from supplied evidence")
    clips = value["clips"]
    if not isinstance(clips, dict) or set(clips) != {row["video_id"] for row in panda}:
        raise ValueError("source mapping must cover exactly all Panda clip IDs")
    return {clip: _mapped_group(item) for clip, item in clips.items()}


def _mapped_group(item: object) -> str:
    if (
        not isinstance(item, dict)
        or set(item) != {"url", "timestamp"}
        or not isinstance(item["url"], str)
    ):
        raise ValueError("source mapping clip requires URL and timestamp")
    url = urlparse(item["url"])
    if url.scheme != "https" or url.hostname not in (
        "www.youtube.com",
        "youtube.com",
        "youtu.be",
    ):
        raise ValueError("source mapping requires original YouTube URLs")
    video_id = (
        url.path.removeprefix("/")
        if url.hostname == "youtu.be"
        else parse_qs(url.query).get("v", [""])[0]
    )
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id) is None:
        raise ValueError("invalid original YouTube ID in source mapping")
    timestamps = item["timestamp"]
    if (
        not isinstance(timestamps, list)
        or len(timestamps) != 2
        or any(
            isinstance(t, bool)
            or not isinstance(t, (int, float))
            or not math.isfinite(t)
            or t < 0
            for t in timestamps
        )
        or timestamps[1] <= timestamps[0]
    ):
        raise ValueError("source mapping requires increasing numeric start/end seconds")
    return "panda70m:youtube:" + video_id


def _records(
    row: dict[str, str],
    hashes: dict[str, str],
    group: str,
    *,
    exploratory: bool = False,
) -> list[SampleRecord]:
    real_id = "vif:" + row["video_path"]
    return [
        SampleRecord(
            sample_id="vif:" + row[key],
            path=row[key],
            modality="video",
            label=label,
            source=SOURCE,
            content_type="general_generated",
            generator=row["aigc_model"] if label else "unknown",
            group_id="unknown" if exploratory else group,
            identity_id="unknown",
            sha256=hashes[row[key]],
            license_id=LICENSE_ID,
            original_id="unknown" if exploratory else "panda70m:" + row["video_id"],
            parent_id=real_id if label else "unknown",
            derivation_group_id=group,
            split="unassigned",
            decode_status="unknown",
        )
        for label, key in ((0, "video_path"), (1, "aigc_video_path"))
    ]


def _suspected_group(video_id: str) -> str:
    # This is only a conservative leakage edge, never original-recording evidence.
    match = re.match(r"^([A-Za-z0-9_-]{11})-", video_id)
    component = match[1] if match else "clip:" + video_id
    return SUSPECTED_RULE + ":" + component


def prepare_panda(  # noqa: PLR0913 - explicit CLI evidence and mode inputs
    archive_path: Path,
    output: Path,
    *,
    naming_evidence: Path,
    license_evidence: list[Path],
    source_mapping: Path | None = None,
    exploratory: bool = False,
) -> dict[str, object]:
    """Import all Panda pairs only after the caller's documented evidence review."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("output must not exist")
    if exploratory == (source_mapping is not None):
        raise ValueError(
            "choose exploratory mode or explicit source mapping, exclusively"
        )
    naming = _evidence([naming_evidence])
    licenses = _evidence(license_evidence)
    mapping_evidence = _evidence([source_mapping]) if source_mapping else {}
    before = archive_path.stat()
    audit = inspect_archive(archive_path)
    _budget(archive_path, int(str(audit["panda_media_bytes"])))
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        directory = _directory(archive)
        panda = _metadata(archive, directory)
        source_groups = (
            _source_map(source_mapping, panda, next(iter(naming.values())))
            if source_mapping
            else {row["video_id"]: _suspected_group(row["video_id"]) for row in panda}
        )
        with tempfile.TemporaryDirectory(
            prefix=f".{output.name}.staging-", dir=output.parent
        ) as temp:
            stage = Path(temp)
            records: list[SampleRecord] = []
            for row in panda:
                hashes = {
                    row[key]: _copy(
                        archive, directory[row[key]], stage / "media" / row[key]
                    )
                    for key in ("video_path", "aigc_video_path")
                }
                records.extend(
                    _records(
                        row,
                        hashes,
                        source_groups[row["video_id"]],
                        exploratory=exploratory,
                    )
                )
            manifest_sha = write_manifest(records, stage / "manifest.jsonl")
            grouping = {
                "rule": SUSPECTED_RULE if exploratory else NAMING_CONTRACT,
                "proves_common_original_recording": not exploratory,
                "explanation": (
                    "First 11 filename characters before '-' conservatively link clips; "
                    "this is an unverified heuristic. Author real/fake pairs are retained "
                    "as parent edges; matching SHA-256 adds closure during splitting."
                    if exploratory
                    else "Original URLs and timestamps mapped by evidence."
                ),
                "components": {
                    group: {
                        "author_video_ids": sorted(
                            {
                                row["video_id"]
                                for row in panda
                                if source_groups[row["video_id"]] == group
                            }
                        ),
                        "sample_ids": sorted(
                            record.sample_id
                            for record in records
                            if record.derivation_group_id == group
                        ),
                    }
                    for group in sorted(set(source_groups.values()))
                },
            }
            atomic_json(stage / "grouping-evidence.json", grouping)
            date = datetime.now(UTC).isoformat()
            source = SourceRecord(
                SOURCE,
                SOURCE_URL,
                REVISION,
                LICENSE_ID,
                "https://github.com/snap-research/Panda-70M/blob/main/LICENSE",
                date,
                (
                    "training",
                    "evaluation",
                    "local_noncommercial_computational_research",
                ),
                "approved",
                ARCHIVE_SHA256,
            )
            atomic_json(stage / "sources.jsonl", asdict(source))
            summary = {
                **audit,
                "created_at": date,
                "source": SOURCE,
                "revision": REVISION,
                "manifest_sha256": manifest_sha,
                "sample_count": len(records),
                "source_video_group_count": len(
                    {
                        record.group_id
                        for record in records
                        if record.group_id != "unknown"
                    }
                ),
                "suspected_component_count": len(set(source_groups.values()))
                if exploratory
                else 0,
                "naming_contract": SUSPECTED_RULE if exploratory else NAMING_CONTRACT,
                "exploratory": exploratory,
                "local_research_only": True,
                "independent_group_ci_allowed": not exploratory,
                "formal_promotion_allowed": False,
                "naming_evidence": naming,
                "source_mapping": mapping_evidence,
                "license_evidence": licenses,
                "permission_scope": "local noncommercial computational research only",
                "redistribute_media_allowed": False,
                "commercial_product_use_allowed": False,
                "competition_approved": False,
                "identity_known": False,
                "split": "unassigned",
                "data_root": "media",
            }
            atomic_json(stage / "import-summary.json", summary)
            after = archive_path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError("archive changed during preparation")
            _publish(stage, output)
    return summary


def _import_args(args: argparse.Namespace) -> dict[str, object]:
    if (
        args.naming_evidence is None
        or not args.license_evidence
        or (args.source_mapping is None and not args.exploratory)
    ):
        raise ValueError(
            "reviewed naming/license evidence and explicit source mapping or exploratory mode are required before import"
        )
    return prepare_panda(
        args.archive,
        args.output,
        naming_evidence=args.naming_evidence,
        license_evidence=args.license_evidence,
        source_mapping=args.source_mapping,
        exploratory=args.exploratory,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--naming-evidence", type=Path)
    parser.add_argument("--license-evidence", type=Path, nargs="+")
    parser.add_argument("--source-mapping", type=Path)
    parser.add_argument("--exploratory", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.audit_only:
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError("audit output must not exist")
            summary = inspect_archive(args.archive)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".vif-audit-staging-", dir=args.output.parent
            ) as temp:
                stage = Path(temp) / "audit.json"
                atomic_json(stage, summary)
                _publish(stage, args.output)
        else:
            summary = _import_args(args)
    except (ValueError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        print(
            json.dumps({"error": type(exc).__name__, "message": str(exc)}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
