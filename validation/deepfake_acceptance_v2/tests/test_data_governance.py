from __future__ import annotations

import csv
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_manifest import build  # noqa: E402
from common import GROUND_TRUTH_FIELDS, sha256_file, write_csv  # noqa: E402
from generate_perturbations import generate  # noqa: E402
from validate_dataset import validate  # noqa: E402


def make_pattern(path: Path, offset: int = 0) -> None:
    image = Image.new("RGB", (19, 17))
    image.putdata([((x * 17 + offset) % 256, (y * 23 + offset) % 256, ((x + y) * 11 + offset) % 256) for y in range(17) for x in range(19)])
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def original_row(file_path: str, label: str = "REAL", sample_id: str = "R0001") -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "file_path": file_path,
        "label": label,
        "source": "unit-test provenance",
        "generation_method": "not_applicable" if label == "REAL" else "unit_test_faceswap",
        "source_face_id": "",
        "target_face_id": "",
        "is_original": "true",
        "parent_sample_id": "",
        "perturbation_type": "",
        "perturbation_level": "",
        "width": "",
        "height": "",
        "sha256": "",
        "notes": "fixture",
    }


class GovernanceTests(unittest.TestCase):
    def test_manifest_freeze_sha_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_pattern(root / "originals" / "r.png")
            curation = root / "curation.csv"
            write_csv(curation, GROUND_TRUTH_FIELDS, [original_row("originals/r.png")])
            frozen = root / "manifest.csv"
            summary = build(curation, root, frozen)
            self.assertEqual(summary["sample_count"], 1)
            with frozen.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["sha256"], sha256_file(root / "originals" / "r.png"))
            self.assertEqual(validate(frozen, root)["status"], "PASS")

    def test_unknown_never_enters_formal_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_pattern(root / "x.png")
            row = original_row("x.png", label="UNKNOWN", sample_id="U1")
            curation = root / "curation.csv"
            write_csv(curation, GROUND_TRUTH_FIELDS, [row])
            with self.assertRaisesRegex(ValueError, "REAL or FAKE"):
                build(curation, root, root / "formal.csv")

    def test_validator_detects_duplicate_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_pattern(root / "a.png")
            (root / "b.png").write_bytes((root / "a.png").read_bytes())
            sha = sha256_file(root / "a.png")
            rows = []
            for sample_id, name, label in (("R1", "a.png", "REAL"), ("F1", "b.png", "FAKE")):
                row = original_row(name, label, sample_id)
                row.update(width="19", height="17", sha256=sha)
                rows.append(row)
            manifest = root / "manifest.csv"
            write_csv(manifest, GROUND_TRUTH_FIELDS, rows)
            report = validate(manifest, root)
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("DUPLICATE_IMAGE", {item["code"] for item in report["errors"]})

    def test_validator_detects_duplicate_sample_id_and_bad_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_pattern(root / "a.png", 1)
            make_pattern(root / "b.png", 2)
            rows = []
            for name, label in (("a.png", "REAL"), ("b.png", "FAKE")):
                row = original_row(name, label, "DUPLICATE_ID")
                row.update(width="19", height="17", sha256="0" * 64)
                rows.append(row)
            manifest = root / "manifest.csv"
            write_csv(manifest, GROUND_TRUTH_FIELDS, rows)
            codes = {item["code"] for item in validate(manifest, root)["errors"]}
            self.assertIn("SAMPLE_ID_DUPLICATE", codes)
            self.assertIn("SHA256_MISMATCH", codes)

    def test_validator_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            root.mkdir()
            outside = root.parent / "outside.png"
            make_pattern(outside)
            row = original_row("../outside.png")
            row.update(width="19", height="17", sha256=sha256_file(outside))
            manifest = root / "manifest.csv"
            write_csv(manifest, GROUND_TRUTH_FIELDS, [row])
            report = validate(manifest, root)
            self.assertIn("PATH_INVALID", {item["code"] for item in report["errors"]})

    def test_perturbations_are_deterministic_named_and_inherit_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "originals" / "fake.png"
            make_pattern(source, 7)
            row = original_row("originals/fake.png", "FAKE", "F0001")
            row.update(width="19", height="17", sha256=sha256_file(source))
            manifest = root / "original.csv"
            write_csv(manifest, GROUND_TRUTH_FIELDS, [row])
            first_manifest = root / "first.csv"
            second_manifest = root / "second.csv"
            first = generate(manifest, root, root / "pert_a", first_manifest)
            second = generate(manifest, root, root / "pert_b", second_manifest)
            self.assertEqual(first["generated_count"], 9)
            self.assertEqual(second["generated_count"], 9)
            with first_manifest.open(encoding="utf-8", newline="") as handle:
                first_rows = list(csv.DictReader(handle))[1:]
            with second_manifest.open(encoding="utf-8", newline="") as handle:
                second_rows = list(csv.DictReader(handle))[1:]
            self.assertTrue(all(row["label"] == "FAKE" and row["parent_sample_id"] == "F0001" for row in first_rows))
            self.assertEqual(len({row["sample_id"] for row in first_rows}), 9)
            self.assertTrue(all(row["sample_id"] in Path(row["file_path"]).stem for row in first_rows))
            self.assertEqual([row["sha256"] for row in first_rows], [row["sha256"] for row in second_rows])
            self.assertEqual(validate(first_manifest, root)["status"], "PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
