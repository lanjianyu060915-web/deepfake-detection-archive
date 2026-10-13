"""File/provenance audit. No decoding or inferred metadata is silently substituted."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .manifest import UNKNOWN, SampleRecord, SourceRecord, file_sha256, manifest_hash
from .split import connected_groups, statistical_groups, validate_splits


@dataclass(frozen=True)
class AuditIssue:
    sample_id: str
    code: str
    detail: str
    severity: str = "error"


@dataclass(frozen=True)
class AuditReport:
    manifest_hash: str
    total_samples: int
    total_bytes: int
    group_count: int
    known_statistical_groups: int
    unknown_statistical_samples: int
    trainable_ids: tuple[str, ...]
    issues: tuple[AuditIssue, ...]
    counts: dict[str, dict[str, int]]
    coverage: dict[str, dict[str, object]]

    @property
    def passed(self) -> bool:
        return self.total_samples > 0 and not any(
            issue.severity == "error" for issue in self.issues
        )

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "passed": self.passed}


def _audit_record(
    record: SampleRecord,
    root: Path,
    sources: dict[str, SourceRecord],
    require_decoded: bool,
) -> tuple[list[AuditIssue], str | None, int]:
    issues: list[AuditIssue] = []

    def issue(code: str, detail: str, severity: str = "error") -> None:
        issues.append(AuditIssue(record.sample_id, code, detail, severity))

    if record.status != "available":
        issue("unavailable", record.failure_reason)
    if record.label is None:
        issue("unknown_label", "label is not available; cannot train/evaluate")
    if record.sha256 == UNKNOWN:
        issue("unknown_hash", "register expected bytes before using sample")
    actual: str | None = None
    size = 0
    try:
        path = (root / record.path).resolve()
        if not path.is_relative_to(root):
            issue("path_escape", "resolved path escapes the media root")
        elif not path.is_file():
            issue("missing_file", record.path)
        else:
            actual = file_sha256(path)
            size = path.stat().st_size
            if record.sha256 not in (UNKNOWN, actual):
                issue("hash_mismatch", f"actual sha256={actual}")
    except OSError as exc:
        issue("file_read_error", str(exc))
    issues.extend(_audit_source(record, sources.get(record.source)))
    if require_decoded and record.decode_status != "ok":
        issue("decode_unverified", f"decode_status={record.decode_status}")
    for field in ("content_type", "identity_id", "generator", "original_id"):
        if getattr(record, field) == UNKNOWN:
            issue("unknown_metadata", f"{field}=unknown", "warning")
    return issues, actual, size


def _audit_source(
    record: SampleRecord, source: SourceRecord | None
) -> list[AuditIssue]:
    if source is None or source.source != record.source:
        return [AuditIssue(record.sample_id, "unregistered_source", record.source)]
    issues: list[AuditIssue] = []
    if source.license_status != "approved" or "training" not in source.allowed_uses:
        issues.append(
            AuditIssue(
                record.sample_id,
                "license_not_approved",
                "training permission not approved",
            )
        )
    if record.license_id == UNKNOWN or record.license_id != source.license_id:
        issues.append(
            AuditIssue(
                record.sample_id,
                "license_mismatch",
                "sample and source license IDs must agree",
            )
        )
    issues.extend(
        AuditIssue(record.sample_id, "incomplete_source", f"source.{field} is unknown")
        for field in ("url", "version", "license_id", "license_url", "downloaded_at")
        if getattr(source, field) == UNKNOWN
    )
    return issues


def _hash_conflicts(actual_hashes: dict[str, list[SampleRecord]]) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for checksum, items in actual_hashes.items():
        if len({record.label for record in items if record.label is not None}) > 1:
            issues.extend(
                AuditIssue(r.sample_id, "label_conflict", checksum) for r in items
            )
        splits = {record.split for record in items} - {"unassigned"}
        if len(splits) > 1:
            issues.extend(
                AuditIssue(r.sample_id, "actual_hash_leakage", checksum) for r in items
            )
    return issues


def _coverage(
    records: list[SampleRecord], groups: dict[str, str], statistical: dict[str, str]
) -> dict[str, dict[str, object]]:
    coverage: dict[str, dict[str, object]] = {}
    for modality in ("audio", "video"):
        types = (
            ("speech", "music", "environment", "sound_effect", "mixed", UNKNOWN)
            if modality == "audio"
            else ("face_manipulation", "general_generated", "other", UNKNOWN)
        )
        observed = {r.content_type for r in records if r.modality == modality}
        for content_type in sorted(set(types) | observed):
            bucket = [
                r
                for r in records
                if r.modality == modality and r.content_type == content_type
            ]
            bucket_groups = {groups[r.sample_id] for r in bucket}
            known = [r for r in bucket if statistical[r.sample_id] != UNKNOWN]
            label_counts = Counter(r.label for r in bucket)
            per_label_groups = {
                label: len(
                    {statistical[r.sample_id] for r in known if r.label == label}
                )
                for label in (0, 1)
            }
            status = "unverified" if content_type == UNKNOWN or not bucket else "sparse"
            if (
                content_type != UNKNOWN
                and len(known) == len(bucket)
                and all(
                    label_counts[label] >= 50 and per_label_groups[label] >= 10
                    for label in (0, 1)
                )
            ):
                status = "counts_sufficient"
            coverage[f"{modality}/{content_type}"] = {
                "samples": len(bucket),
                "groups": len(bucket_groups),
                "known_statistical_groups": len(
                    {statistical[r.sample_id] for r in known}
                ),
                "unknown_statistical_samples": len(bucket) - len(known),
                "real_statistical_groups": per_label_groups[0],
                "fake_statistical_groups": per_label_groups[1],
                "real": label_counts[0],
                "fake": label_counts[1],
                "unknown_label": label_counts[None],
                "status": status,
                "fraction_of_modality": len(bucket)
                / max(sum(r.modality == modality for r in records), 1),
            }
    return coverage


def audit_manifest(
    records: list[SampleRecord],
    root: Path,
    sources: dict[str, SourceRecord],
    duplicate_pairs: tuple[tuple[str, str], ...] = (),
    *,
    require_decoded: bool = True,
) -> AuditReport:
    """Verify bytes and registry. Full eligibility also needs decode_status='ok'.

    Missing/failed entries remain in all denominators. The caller must obtain
    decode_status from an actual decoding pass; this function cannot certify it.
    A caller that decodes during preparation may request a pre-decode audit;
    such a passing report establishes registry/byte eligibility, not decodability.
    Coverage counts alone do not certify generalization or statistical sufficiency.
    """
    groups = connected_groups(records, duplicate_pairs)
    statistical = statistical_groups(records, duplicate_pairs)
    root = root.resolve()
    issues: list[AuditIssue] = []
    total_bytes = 0
    actual_hashes: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        record_issues, actual, size = _audit_record(
            record, root, sources, require_decoded
        )
        issues.extend(record_issues)
        total_bytes += size
        if actual is not None:
            actual_hashes[actual].append(record)
    issues.extend(_hash_conflicts(actual_hashes))
    split_errors = validate_splits(
        records,
        duplicate_pairs,
        require_class_coverage=any(r.split != "unassigned" for r in records),
    )
    issues.extend(
        AuditIssue("manifest", "split_invalid", error) for error in split_errors
    )
    invalid = {issue.sample_id for issue in issues if issue.severity == "error"}
    trainable = (
        []
        if split_errors
        else [r.sample_id for r in records if r.sample_id not in invalid]
    )
    counts = {
        key: dict(sorted(Counter(str(getattr(r, key)) for r in records).items()))
        for key in (
            "modality",
            "label",
            "source",
            "content_type",
            "generator",
            "split",
            "status",
            "codec",
            "container",
            "decode_status",
        )
    }
    return AuditReport(
        manifest_hash(records),
        len(records),
        total_bytes,
        len(set(groups.values())),
        len(set(statistical.values()) - {UNKNOWN}),
        sum(group == UNKNOWN for group in statistical.values()),
        tuple(sorted(item for item in trainable if item not in invalid)),
        tuple(issues),
        counts,
        _coverage(records, groups, statistical),
    )


def write_audit_report(report: AuditReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
