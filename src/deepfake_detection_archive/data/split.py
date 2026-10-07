"""Connected-component partitions that preserve all declared leakage relations."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import replace

from .manifest import SPLITS, UNKNOWN, SampleRecord


def connected_groups(
    records: list[SampleRecord],
    duplicate_pairs: tuple[tuple[str, str], ...] = (),
) -> dict[str, str]:
    """Return sample -> minimum sample ID in its transitive leakage component.

    Identity/group/derivation IDs are globally namespaced by the data curator.
    Original source item IDs are scoped to source. Unknown never links samples.
    External parent IDs are retained as nodes, linking siblings without a parent row.
    """
    ids = [record.sample_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample_id")
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != key:
            next_key = parent[key]
            parent[key] = root
            key = next_key
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for record in records:
        node = f"sample:{record.sample_id}"
        find(node)
        for name in ("group_id", "sha256", "identity_id", "derivation_group_id"):
            value = getattr(record, name)
            if value != UNKNOWN:
                union(node, f"{name}:{value}")
        if record.parent_id != UNKNOWN:
            union(node, f"sample:{record.parent_id}")
        if record.original_id != UNKNOWN:
            union(node, f"original:{record.source}:{record.original_id}")
    for left, right in duplicate_pairs:
        if left not in ids or right not in ids:
            raise ValueError("near-duplicate edge refers to an absent sample")
        union(f"sample:{left}", f"sample:{right}")
    components: dict[str, list[str]] = defaultdict(list)
    for sample_id in ids:
        components[find(f"sample:{sample_id}")].append(sample_id)
    return {item: min(group) for group in components.values() for item in group}


def statistical_groups(
    records: list[SampleRecord],
    duplicate_pairs: tuple[tuple[str, str], ...] = (),
) -> dict[str, str]:
    """Return components only where registered group/identity evidence exists.

    Hashes, source item IDs and singleton components prevent known leakage but
    cannot establish independent people or independent acquisition units. A
    component without explicit group_id/identity_id evidence remains unknown.
    """
    components = connected_groups(records, duplicate_pairs)
    evidenced = {
        components[record.sample_id]
        for record in records
        if record.group_id.strip().casefold() != UNKNOWN
        or record.identity_id.strip().casefold() != UNKNOWN
    }
    return {
        sample_id: group if group in evidenced else UNKNOWN
        for sample_id, group in components.items()
    }


def validate_splits(
    records: list[SampleRecord],
    duplicate_pairs: tuple[tuple[str, str], ...] = (),
    require_class_coverage: bool = True,
) -> list[str]:
    groups = connected_groups(records, duplicate_pairs)
    assignments: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.split != "unassigned":
            assignments[groups[record.sample_id]].add(record.split)
    errors = [
        f"leakage: group {group} spans {sorted(splits)}"
        for group, splits in sorted(assignments.items())
        if len(splits) > 1
    ]
    if require_class_coverage:
        for modality in sorted({record.modality for record in records}):
            for split in SPLITS:
                labels = {
                    record.label
                    for record in records
                    if record.modality == modality
                    and record.split == split
                    and record.status == "available"
                }
                if not {0, 1}.issubset(labels):
                    errors.append(f"missing_class: {modality}/{split}")
    return errors


def _validate_split_inputs(
    records: list[SampleRecord], ratios: tuple[float, float, float]
) -> None:
    if (
        len(ratios) != 3
        or any(not math.isfinite(x) or x <= 0 for x in ratios)
        or not math.isclose(sum(ratios), 1.0)
    ):
        raise ValueError("ratios must be three positive finite values summing to 1")
    if not records:
        raise ValueError("cannot split empty manifest")
    if any(record.label is None or record.split == "test" for record in records):
        raise ValueError("unlabelled/test samples cannot enter research splits")


def assign_splits(
    records: list[SampleRecord],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    duplicate_pairs: tuple[tuple[str, str], ...] = (),
) -> list[SampleRecord]:
    """Deterministic component-level stratification with immutable existing splits.

    Reserve groups covering both labels for holdout first, then dev, then train.
    Greedily balance remaining label/modality counts against target fractions.
    Impossible coverage is a hard error, rather than splitting a connected group.
    """
    _validate_split_inputs(records, ratios)
    group_map = connected_groups(records, duplicate_pairs)
    grouped: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        grouped[group_map[record.sample_id]].append(record)
    counts: dict[str, Counter[tuple[str, int | None]]] = {
        group: Counter((r.modality, r.label) for r in items if r.status == "available")
        for group, items in grouped.items()
    }
    totals: Counter[tuple[str, int | None]] = Counter()
    for values in counts.values():
        totals.update(values)
    keys = sorted(totals)
    modalities = {record.modality for record in records}
    if any(
        (modality, label) not in totals for modality in modalities for label in (0, 1)
    ):
        raise ValueError("both classes are required for every modality")
    assigned: dict[str, str] = {}
    running: dict[str, Counter[tuple[str, int | None]]] = {
        split: Counter() for split in SPLITS
    }
    for group, items in grouped.items():
        fixed = {r.split for r in items if r.split != "unassigned"}
        if len(fixed) > 1:
            raise ValueError(f"existing split leakage in group {group}")
        if fixed:
            split = fixed.pop()
            assigned[group] = split
            running[split].update(counts[group])

    def rank(group: str) -> str:
        return hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()

    remaining = set(grouped) - assigned.keys()
    # Most constrained class first; prefer components covering several missing keys.
    for split in ("holdout", "dev", "train"):
        while missing := {key for key in keys if running[split][key] == 0}:
            key = min(
                missing,
                key=lambda item: (sum(counts[g][item] > 0 for g in remaining), item),
            )
            candidates = [g for g in remaining if counts[g][key] > 0]
            if not candidates:
                raise ValueError(
                    f"insufficient independent groups for {split}/{key}; "
                    "collect more groups or revise predeclared partitions"
                )
            group = min(
                candidates,
                key=lambda g: (
                    -sum(counts[g][item] > 0 for item in missing),
                    sum(counts[g].values()),
                    rank(g),
                ),
            )
            assigned[group] = split
            running[split].update(counts[group])
            remaining.remove(group)
    targets = dict(zip(SPLITS, ratios, strict=True))
    for group in sorted(remaining, key=lambda g: (-len(grouped[g]), rank(g))):

        def cost(split: str, current_group: str = group) -> float:
            return sum(
                (
                    (running[split][key] + counts[current_group][key]) ** 2
                    - running[split][key] ** 2
                )
                / max(targets[split] * totals[key], 1.0)
                for key in keys
            )

        split = min(SPLITS, key=cost)
        assigned[group] = split
        running[split].update(counts[group])
    result = [
        replace(record, split=assigned[group_map[record.sample_id]])
        for record in records
    ]
    errors = validate_splits(result, duplicate_pairs)
    if errors:
        raise ValueError("; ".join(errors))
    return result
