"""Behavior checks for the research-only pilot preparation script."""

import importlib.util
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    load_manifest,
    write_manifest,
)
from deepfake_detection_archive.data.split import connected_groups, validate_splits
from deepfake_detection_archive.tracking import atomic_json


SPEC = importlib.util.spec_from_file_location(
    "prepare_pilot", Path(__file__).with_name("prepare_pilot.py")
)
assert SPEC is not None
assert SPEC.loader is not None
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


def pairs(count: int) -> list[SampleRecord]:
    result: list[SampleRecord] = []
    for index in range(count):
        generator = f"generator{index % 7}"
        real = SampleRecord(
            sample_id=f"real{index}",
            path=f"real{index}.mp4",
            modality="video",
            label=0,
            source="local-test",
            derivation_group_id=f"prefix{index // 2}",
        )
        fake = replace(
            real,
            sample_id=f"fake{index}",
            path=f"fake{index}.mp4",
            label=1,
            parent_id=real.sample_id,
            generator=generator,
        )
        result.extend((real, fake))
    return result


class TestPilot:
    def test_partition_preserves_transitive_relations_and_unknown_statistics(
        self,
    ) -> None:
        rows = pairs(70)
        edges = (("fake0", "real16"), ("fake16", "real24"))
        result = PILOT.partition(rows, edges)
        assert validate_splits(result, edges) == []
        assert set(PILOT.statistical_groups(result).values()) == {"unknown"}
        assignments = {r.sample_id: r.split for r in result}
        assert assignments["real0"] == assignments["fake24"]
        assert assignments == {
            r.sample_id: r.split for r in PILOT.partition(list(reversed(rows)), edges)
        }
        for split in PILOT.SPLITS:
            present = {
                PILOT.strata(rows)[r.sample_id] for r in result if r.split == split
            }
            assert len(present) == 14

    def test_unsplittable_stratum_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="insufficient components"):
            PILOT.partition(pairs(7), ())

    def test_selection_retains_full_source_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("work", "doc", "data"):
                (root / name).mkdir()
            rows = pairs(560)
            checksum = write_manifest(rows, root / "data/manifest.jsonl")
            atomic_json(
                root / "data/import-summary.json", {"manifest_sha256": checksum}
            )
            atomic_json(root / "data/grouping-evidence.json", {"test": True})
            atomic_json(root / "work/preregistration.json", {"test": True})
            with patch.multiple(
                PILOT, WORK=root / "work", DOC=root / "doc", DATA=root / "data"
            ):
                PILOT.select()
                selected = load_manifest(root / "work/selected.jsonl")
                assert len(selected) == 1024
                groups = connected_groups(rows)
                ids = {r.sample_id for r in selected}
                selected_groups = {groups[i] for i in ids}
                assert ids == {i for i, g in groups.items() if g in selected_groups}
                assert {r.generator for r in selected if r.label == 1} == {
                    f"generator{i}" for i in range(7)
                }
                with pytest.raises(ValueError, match="already exists"):
                    PILOT.select()

    def test_missing_media_is_explicit_failure(self) -> None:
        result = PILOT.decode_one(pairs(1)[0])
        assert result["status"] == "failed"
        assert "error_type" in result
        assert result["sample_id"] == "real0"

    def test_padding_does_not_make_flat_content_valid(self) -> None:
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame[56:168] = 128
        # The padded frame itself varies, but its source content is flat.
        assert len(PILOT.image_phash(frame)) == 63
        with pytest.raises(PILOT.MediaError, match="nearly flat"):
            PILOT.frame_hash(frame, 448, 224)

    def test_metadata_mutation_invalidates_selected_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "work").mkdir()
            (root / "data").mkdir()
            rows = pairs(7)
            source_sha = write_manifest(rows, root / "data/manifest.jsonl")
            selected_sha = write_manifest(rows, root / "work/selected.jsonl")
            atomic_json(
                root / "work/selection.json",
                {
                    "source_manifest_sha256": source_sha,
                    "selected_manifest_sha256": selected_sha,
                },
            )
            with patch.multiple(PILOT, WORK=root / "work", DATA=root / "data"):
                PILOT.check_selected()
                changed = [replace(rows[0], label=1), *rows[1:]]
                write_manifest(changed, root / "work/selected.jsonl")
                with pytest.raises(ValueError, match="changed after selection"):
                    PILOT.check_selected()

    def test_unsplittable_similarity_component_preserves_candidate_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "work").mkdir()
            (root / "doc").mkdir()
            rows = pairs(7)
            features = np.ones((len(rows), 4, 512), dtype=np.float32)
            np.save(root / "features.npy", features, allow_pickle=False)
            samples = [
                {
                    "sample_id": r.sample_id,
                    "duration_seconds": 1.0,
                    "phash": [None] * 4,
                    "phash_failures": ["low_information"],
                }
                for r in rows
            ]
            with (
                patch.multiple(PILOT, ROOT=root, WORK=root / "work", DOC=root / "doc"),
                patch.object(
                    PILOT,
                    "audited_inputs",
                    return_value=(rows, {"samples": samples}, {"path": "features.npy"}),
                ),
            ):
                with pytest.raises(ValueError, match="insufficient components"):
                    PILOT.lock()
                audit = PILOT.read(root / "work/duplicate-audit.json")
                assert audit["candidate_count"] == 91
                assert all(c["embedding_candidate"] for c in audit["candidates"])
                assert (
                    PILOT.read(root / "work/component-audit.json")["max_fraction"]
                    == 1.0
                )
                assert not (root / "work/split-lock.json").exists()
