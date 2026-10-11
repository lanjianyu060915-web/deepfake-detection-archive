"""Import pinned ViF-Bench source videos for cross-generator local diagnosis.

The fixed archive keeps generated clips under ``source_videos/Fake/<generator>/``
beside ``source_videos/Real/``. A generated file is treated as the author pair of a
real file only when both file names are byte-identical; the first eleven characters
of a name are kept as a conservative leakage edge, never as evidence of a shared
original recording. Rows whose component already appears in a supplied exclusion
manifest are dropped before any frame is decoded. This tool performs no download,
training, identity inference, or competition approval.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import tempfile
import zipfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from prepare_vif_panda import _copy, _directory, _publish

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    SourceRecord,
    file_sha256,
    load_manifest,
    write_manifest,
)
from deepfake_detection_archive.tracking import atomic_json


REPO = "JoeLeelyf/ViF-Bench"
REVISION = "67e11331e26dbf732e0b8c7d4dc52f4442d174dd"
ARCHIVE_BYTES = 607734233
ARCHIVE_SHA256 = "41e79dff9f8ff16f7bcddd99eb18e4019b52f5ed9fe26a6a0fa0739b085907a8"
INDEX_PATH = "source_videos/ladm_paths.json"
INDEX_PREFIX = "path_to_source_videos_dir/"
INDEX_ROOT = "source_videos/"
REAL_DIR = "source_videos/Real"
FAKE_DIR = "source_videos/Fake"
SOURCE = "vif-bench-cross-generator-evaluation"
SOURCE_URL = f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}"
LICENSE_ID = "ViF-Bench-CC-BY-4.0-with-unresolved-upstream-media-boundaries"
LICENSE_URL = f"https://huggingface.co/datasets/{REPO}"
ALLOWED_USES = ("evaluation", "local_noncommercial_computational_research")
PAIRING_RULE = "vifbench_identical_filename_pair_v1"
COMPONENT_RULE = "suspected_filename_component:v1"
NOTES = (
    "Generated files are paired to a real file only on identical names; the rule "
    "describes the author's real/fake alignment, not an independent recording. The "
    "eleven-character component is a conservative leakage edge only. The pinned "
    "index lists paths with a placeholder prefix and carries no source field, so "
    "upstream media boundaries stay unresolved."
)
MAX_MEMBER_BYTES = 100 * 1024**2
MAX_EXTRACT_BYTES = 4 * 1024**3
MAX_MEMBERS = 20000


def component(stem: str) -> str:
    """Return the conservative component name for one media stem."""
    match = re.match(r"^([A-Za-z0-9_-]{11})-", stem)
    return match.group(1) if match else "clip:" + stem


def stem_of(path: str) -> str:
    return path.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def excluded_components(manifests: list[Path]) -> dict[str, str]:
    """Map every component already registered in the given manifests to its source."""
    result: dict[str, str] = {}
    for manifest in manifests:
        for record in load_manifest(manifest):
            name = component(stem_of(record.path))
            result.setdefault(name, str(manifest.resolve()))
    return result


def layout(
    directory: dict[str, zipfile.ZipInfo],
) -> tuple[list[str], dict[str, list[str]], str]:
    """Split the archive into real media, per-generator fake media and the index."""
    reals: list[str] = []
    fakes: dict[str, list[str]] = {}
    index = ""
    for name in directory:
        parts = name.split("/")
        if (
            name.endswith(".mp4")
            and len(parts) == 3
            and "/".join(parts[:2]) == REAL_DIR
        ):
            reals.append(name)
        elif (
            name.endswith(".mp4")
            and len(parts) == 4
            and "/".join(parts[:2]) == FAKE_DIR
        ):
            fakes.setdefault(parts[2], []).append(name)
        elif name == INDEX_PATH:
            index = name
        else:
            raise ValueError(f"unexpected archive member: {name}")
    if not reals or not fakes or not index:
        raise ValueError("archive lacks Real, Fake or pinned index members")
    real_names = [name.rsplit("/", 1)[-1] for name in reals]
    if len(set(real_names)) != len(real_names):
        raise ValueError("duplicate real file names cannot be paired unambiguously")
    for generator, members in fakes.items():
        names = [name.rsplit("/", 1)[-1] for name in members]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate file names inside generator: {generator}")
    return (
        sorted(reals),
        {generator: sorted(members) for generator, members in fakes.items()},
        index,
    )


def index_paths(raw: bytes) -> list[str]:
    """Read the pinned index and return its declared paths under the archive root."""
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError("pinned index must be a nonempty object")
    declared: list[str] = []
    for key, members in value.items():
        if not isinstance(key, str) or not isinstance(members, list) or not members:
            raise ValueError("pinned index entries must map names to nonempty lists")
        for member in members:
            if not isinstance(member, str) or not member.startswith(INDEX_PREFIX):
                raise ValueError("pinned index members must carry the fixed prefix")
            declared.append(INDEX_ROOT + member[len(INDEX_PREFIX) :])
    if len(set(declared)) != len(declared):
        raise ValueError("pinned index repeats a media path")
    return declared


def audit(path: Path) -> dict[str, object]:
    """Authenticate the fixed archive and describe its layout without copying media."""
    if path.stat().st_size != ARCHIVE_BYTES or file_sha256(path) != ARCHIVE_SHA256:
        raise ValueError("archive failed fixed revision size/LFS SHA-256 verification")
    with zipfile.ZipFile(path) as archive:
        directory = _directory(archive)
        if len(directory) > MAX_MEMBERS:
            raise ValueError("archive exceeds member budget")
        reals, fakes, index = layout(directory)
        index_raw = archive.read(directory[index])
        declared = index_paths(index_raw)
        members = set(reals) | {name for group in fakes.values() for name in group}
        dangling = sorted(set(declared) - members)
        if dangling:
            raise ValueError(f"pinned index references absent media: {dangling[:3]}")
        unlisted = sorted(members - set(declared))
        if unlisted:
            raise ValueError(f"pinned index omits present media: {unlisted[:3]}")
        selected = [directory[name] for name in sorted(members)]
        if any(
            info.file_size <= 0 or info.file_size > MAX_MEMBER_BYTES
            for info in selected
        ):
            raise ValueError("selected media exceed single-file budget or are empty")
        selected_bytes = sum(info.file_size for info in selected)
        if selected_bytes > MAX_EXTRACT_BYTES:
            raise ValueError("selected media exceed extraction budget")
        real_names = {name.rsplit("/", 1)[-1] for name in reals}
        paired = {
            generator: sum(1 for name in group if name.rsplit("/", 1)[-1] in real_names)
            for generator, group in fakes.items()
        }
    return {
        "verified_at": datetime.now(UTC).isoformat(),
        "repository": REPO,
        "revision": REVISION,
        "archive_sha256": ARCHIVE_SHA256,
        "archive_bytes": ARCHIVE_BYTES,
        "index_sha256": hashlib.sha256(index_raw).hexdigest(),
        "real_count": len(reals),
        "fake_count": sum(len(group) for group in fakes.values()),
        "generators": dict(sorted((name, len(group)) for name, group in fakes.items())),
        "paired_fake_count": dict(sorted(paired.items())),
        "index_declared_count": len(declared),
        "index_matches_media": set(declared) == members,
        "media_bytes": selected_bytes,
        "media_decoded": False,
    }


def selection(
    reals: list[str],
    fakes: dict[str, list[str]],
    excluded: dict[str, str],
) -> tuple[list[str], dict[str, list[str]], dict[str, object]]:
    """Drop excluded components entirely and keep the author pairs that remain."""
    kept_reals = [name for name in reals if component(stem_of(name)) not in excluded]
    real_names = {name.rsplit("/", 1)[-1] for name in kept_reals}
    kept_fakes: dict[str, list[str]] = {}
    dropped: collections.Counter[str] = collections.Counter()
    for generator, group in fakes.items():
        keep = [name for name in group if component(stem_of(name)) not in excluded]
        kept_fakes[generator] = [
            name for name in keep if name.rsplit("/", 1)[-1] in real_names
        ]
        dropped[generator] = len(group) - len(keep)
    matched_components = {component(stem_of(name)) for name in kept_reals}
    return (
        kept_reals,
        kept_fakes,
        {
            "excluded_real_count": len(reals) - len(kept_reals),
            "excluded_fake_count": sum(dropped.values()),
            "excluded_fake_by_generator": dict(sorted(dropped.items())),
            "unpaired_fake_count": sum(
                len(group) - len(kept_fakes[generator])
                for generator, group in fakes.items()
            ),
            "kept_component_count": len(matched_components),
            "exclusion_sources": dict(sorted(excluded.items())),
        },
    )


def records(
    kept_reals: list[str],
    kept_fakes: dict[str, list[str]],
    hashes: dict[str, str],
) -> list[SampleRecord]:
    """Build one record per kept member; pairs share the real parent identifier."""
    real_by_name = {name.rsplit("/", 1)[-1]: name for name in kept_reals}
    output: list[SampleRecord] = []
    for path in kept_reals:
        group = f"{COMPONENT_RULE}:{component(stem_of(path))}"
        output.append(
            SampleRecord(
                sample_id="vifbench:" + path,
                path=path,
                modality="video",
                label=0,
                source=SOURCE,
                content_type="general_generated",
                generator="unknown",
                group_id="unknown",
                sha256=hashes[path],
                license_id=LICENSE_ID,
                original_id="unknown",
                parent_id="unknown",
                derivation_group_id=group,
                split="dev",
            )
        )
    for generator in sorted(kept_fakes):
        for path in kept_fakes[generator]:
            parent = real_by_name.get(path.rsplit("/", 1)[-1])
            group = f"{COMPONENT_RULE}:{component(stem_of(path))}"
            output.append(
                SampleRecord(
                    sample_id="vifbench:" + path,
                    path=path,
                    modality="video",
                    label=1,
                    source=SOURCE,
                    content_type="general_generated",
                    generator=generator,
                    group_id="unknown",
                    sha256=hashes[path],
                    license_id=LICENSE_ID,
                    original_id="unknown",
                    parent_id="vifbench:" + parent if parent else "unknown",
                    derivation_group_id=group,
                    split="dev",
                )
            )
    return output


def prepare(
    archive_path: Path, output: Path, excluded_paths: list[Path]
) -> dict[str, object]:
    """Copy every kept member byte-for-byte, then publish the import atomically."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("output must not exist")
    for path in excluded_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("exclusion manifests must be nonempty files")
    excluded = excluded_components(excluded_paths)
    before = archive_path.stat()
    summary = audit(archive_path)
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        directory = _directory(archive)
        reals, fakes, _index = layout(directory)
        kept_reals, kept_fakes, selection_summary = selection(reals, fakes, excluded)
        if not kept_reals or not any(kept_fakes.values()):
            raise ValueError("exclusion leaves no paired real/fake diagnostic rows")
        hashes: dict[str, str] = {}
        with tempfile.TemporaryDirectory(
            prefix=f".{output.name}.staging-", dir=output.parent
        ) as temp:
            stage = Path(temp)
            members = [
                *kept_reals,
                *sorted(name for group in kept_fakes.values() for name in group),
            ]
            for member in members:
                hashes[member] = _copy(
                    archive, directory[member], stage / "media" / member
                )
            rows = records(kept_reals, kept_fakes, hashes)
            digest = write_manifest(rows, stage / "manifest.jsonl")
            diagnostics: dict[str, object] = {}
            for generator in sorted(kept_fakes):
                group = kept_fakes[generator]
                names = {name.rsplit("/", 1)[-1] for name in group}
                subset = [
                    row
                    for row in rows
                    if row.generator == generator
                    or (row.label == 0 and row.path.rsplit("/", 1)[-1] in names)
                ]
                subset_path = stage / "diagnostics" / f"{generator}.jsonl"
                subset_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics[generator] = {
                    "path": f"diagnostics/{generator}.jsonl",
                    "sha256": write_manifest(subset, subset_path),
                    "real_count": sum(1 for row in subset if row.label == 0),
                    "fake_count": sum(1 for row in subset if row.label == 1),
                }
            atomic_json(stage / "diagnostics" / "index.json", diagnostics)
            created = datetime.now(UTC).isoformat()
            atomic_json(
                stage / "sources.jsonl",
                asdict(
                    SourceRecord(
                        SOURCE,
                        SOURCE_URL,
                        REVISION,
                        LICENSE_ID,
                        LICENSE_URL,
                        created,
                        ALLOWED_USES,
                        "approved",
                        ARCHIVE_SHA256,
                    )
                ),
            )
            atomic_json(
                stage / "grouping-evidence.json",
                {
                    "pairing_rule": PAIRING_RULE,
                    "component_rule": COMPONENT_RULE,
                    "proves_common_original_recording": False,
                    "notes": NOTES,
                    "components": {
                        name: sorted(
                            row.sample_id
                            for row in rows
                            if row.derivation_group_id == f"{COMPONENT_RULE}:{name}"
                        )
                        for name in sorted(
                            {component(stem_of(row.path)) for row in rows}
                        )
                    },
                },
            )
            atomic_json(
                stage / "import-summary.json",
                {
                    **summary,
                    **selection_summary,
                    "created_at": created,
                    "source": SOURCE,
                    "license_id": LICENSE_ID,
                    "allowed_uses": list(ALLOWED_USES),
                    "pairing_rule": PAIRING_RULE,
                    "component_rule": COMPONENT_RULE,
                    "manifest_sha256": digest,
                    "sample_count": len(rows),
                    "kept_real_count": len(kept_reals),
                    "diagnostics": diagnostics,
                    "exclusion_manifest_sha256": {
                        str(path.resolve()): file_sha256(path)
                        for path in excluded_paths
                    },
                    "local_research_only": True,
                    "competition_approved": False,
                    "redistribute_media_allowed": False,
                    "independent_group_ci_allowed": False,
                    "identity_known": False,
                    "data_root": "media",
                },
            )
            after = archive_path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError("archive changed during preparation")
            _publish(stage, output)
    return {
        **summary,
        **selection_summary,
        "manifest_sha256": digest,
        "sample_count": len(rows),
        "kept_real_count": len(kept_reals),
        "diagnostics": diagnostics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.audit_only:
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError("audit output must not exist")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".vif-bench-audit-staging-", dir=args.output.parent
            ) as temp:
                stage = Path(temp) / "audit.json"
                atomic_json(stage, audit(args.archive))
                _publish(stage, args.output)
            summary: dict[str, object] = {"output": str(args.output)}
        else:
            summary = prepare(args.archive, args.output, args.exclude_manifest)
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
