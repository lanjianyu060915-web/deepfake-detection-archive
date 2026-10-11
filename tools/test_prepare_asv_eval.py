"""Behavior checks for the ASV eval import; run with unittest discovery."""

from __future__ import annotations

# ruff: noqa: PT009, PT027 - standalone unittest checks alongside the CLI tool
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import prepare_asv_eval
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from deepfake_detection_archive.data.manifest import load_manifest

Row = tuple[str, str, str, int]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _flac(sample_id: str) -> bytes:
    return b"fLaC" + sample_id.encode("utf-8") + b"\x00" * 4


def _rows() -> list[Row]:
    return [
        ("LA_0100", "LA_E_0001", "-", 0),
        ("LA_0100", "LA_E_0002", "A07", 1),
        ("LA_0101", "LA_E_0003", "-", 0),
        ("LA_0101", "LA_E_0004", "A07", 1),
        ("LA_0101", "LA_E_0005", "A08", 1),
    ]


def _protocol_text(rows: list[tuple[Any, ...]]) -> str:
    keys = {0: "bonafide", 1: "spoof"}
    return "".join(
        f"{row[0]} {row[1]} - {row[2]} {keys[row[3]]}\n" + " ".join(row[4:]) + "\n"
        if len(row) > 4
        else f"{row[0]} {row[1]} - {row[2]} {keys[row[3]]}\n"
        for row in rows
    )


def _write_inputs(root: Path, rows: list[Row]) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "speaker_id": pa.array([row[0] for row in rows], pa.string()),
            "audio_file_name": pa.array([row[1] for row in rows], pa.string()),
            "audio": pa.array(
                [
                    {"bytes": _flac(row[1]), "path": f"{row[1]}.flac"}
                    for row in rows
                ],
                prepare_asv_eval.SCHEMA.field("audio").type,
            ),
            "system_id": pa.array([row[2] for row in rows], pa.string()),
            "key": pa.array([row[3] for row in rows], pa.int64()),
        },
        schema=prepare_asv_eval.SCHEMA,
    )
    parquet = root / prepare_asv_eval.EVAL_SHARD
    pq.write_table(table, parquet)
    protocol = root / "ASVspoof2019.LA.cm.eval.trl.txt"
    protocol.write_text(_protocol_text(rows), encoding="utf-8")
    attribution = root / "attribution"
    attribution.mkdir(exist_ok=True)
    (attribution / "README.LA.txt").write_text("fixture\n", encoding="utf-8")
    return parquet, protocol, attribution


class PrepareAsvEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.parquet, self.protocol, self.attribution = _write_inputs(
            self.root, _rows()
        )
        self.pin(self.parquet, self.protocol)

    def pin(self, parquet: Path, protocol: Path) -> None:
        for name, path in (
            ("EVAL_SHARD_SHA256", parquet),
            ("EVAL_PROTOCOL_SHA256", protocol),
        ):
            patch = mock.patch.object(prepare_asv_eval, name, _sha(path.read_bytes()))
            patch.start()
            self.addCleanup(patch.stop)

    def materialize(self, **overrides: Any) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "parquet_path": self.parquet,
            "protocol_path": self.protocol,
            "output_root": self.root / "eval-out",
            "attribution_root": self.attribution,
        }
        arguments.update(overrides)
        return prepare_asv_eval.materialize(**arguments)

    def test_publishes_protocol_labels_with_eval_groups(self) -> None:
        output = self.root / "eval-out"
        summary = self.materialize(output_root=output)
        records = {r.sample_id: r for r in load_manifest(output / "manifest.jsonl")}
        self.assertEqual(summary["rows"], 5)
        self.assertEqual(summary["label_counts"], {"fake": 3, "real": 2})
        self.assertEqual(
            summary["generator_counts"], {"A07": 2, "A08": 1, "human": 2}
        )
        self.assertEqual(summary["speakers"], 2)
        self.assertEqual(summary["split"], "test")
        self.assertFalse(summary["official_evaluation"])
        self.assertEqual(records["LA_E_0001"].label, 0)
        self.assertEqual(records["LA_E_0001"].generator, "human")
        self.assertEqual(records["LA_E_0002"].label, 1)
        self.assertEqual(records["LA_E_0002"].generator, "A07")
        self.assertEqual(
            records["LA_E_0002"].group_id,
            "asvspoof2019-la-eval-official:speaker:LA_0100",
        )
        self.assertEqual(
            records["LA_E_0002"].path, "ASVspoof2019_LA_eval/flac/LA_E_0002.flac"
        )
        self.assertEqual(records["LA_E_0002"].sha256, _sha(_flac("LA_E_0002")))
        media = output / "ASVspoof2019_LA_eval" / "flac"
        self.assertEqual((media / "LA_E_0002.flac").read_bytes(), _flac("LA_E_0002"))
        self.assertEqual(len(list(media.iterdir())), 5)
        self.assertEqual(
            (output / prepare_asv_eval.EVAL_PROTOCOL).read_bytes(),
            self.protocol.read_bytes(),
        )
        source = json.loads((output / "sources.jsonl").read_text("utf-8"))
        self.assertEqual(source["source"], "asvspoof2019-la-eval-official")
        self.assertEqual(source["allowed_uses"], ["research", "evaluation"])
        self.assertEqual(
            sorted(path.name for path in output.iterdir()),
            [
                "ASVspoof2019_LA_cm_protocols",
                "ASVspoof2019_LA_eval",
                "README.LA.txt",
                "import-summary.json",
                "manifest.jsonl",
                "sources.jsonl",
            ],
        )

    def test_rejects_media_metadata_that_contradicts_the_protocol(self) -> None:
        rows = _rows()
        rows[1] = ("LA_0100", "LA_E_0002", "A09", 1)
        parquet, protocol, _ = _write_inputs(self.root / "contradiction", rows)
        # Keep the official protocol unchanged so only the mirror disagrees.
        protocol.write_text(_protocol_text(_rows()), encoding="utf-8")
        self.pin(parquet, protocol)
        output = self.root / "out-contradiction"
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            self.materialize(
                parquet_path=parquet, protocol_path=protocol, output_root=output
            )
        self.assertFalse(output.exists())

    def test_rejects_incomplete_mirror_coverage(self) -> None:
        parquet, _, _ = _write_inputs(self.root / "partial", _rows())
        extra: Row = ("LA_0101", "LA_E_0006", "A09", 1)
        protocol = self.root / "partial" / "extra-protocol.txt"
        protocol.write_text(_protocol_text([*_rows(), extra]), encoding="utf-8")
        self.pin(parquet, protocol)
        with self.assertRaisesRegex(ValueError, "do not cover"):
            self.materialize(
                parquet_path=parquet,
                protocol_path=protocol,
                output_root=self.root / "out-partial",
            )

    def test_rejects_duplicate_mirror_samples(self) -> None:
        parquet, protocol, _ = _write_inputs(
            self.root / "duplicate", [*_rows(), _rows()[0]]
        )
        protocol.write_text(_protocol_text(_rows()), encoding="utf-8")
        self.pin(parquet, protocol)
        with self.assertRaisesRegex(ValueError, "duplicate mirror sample"):
            self.materialize(
                parquet_path=parquet,
                protocol_path=protocol,
                output_root=self.root / "out-duplicate",
            )

    def test_rejects_unpinned_shard_or_protocol(self) -> None:
        output = self.root / "out-unpinned"
        with (
            mock.patch.object(prepare_asv_eval, "EVAL_SHARD_SHA256", "0" * 64),
            self.assertRaisesRegex(ValueError, "eval shard SHA256 mismatch"),
        ):
            self.materialize(output_root=output)
        with (
            mock.patch.object(prepare_asv_eval, "EVAL_PROTOCOL_SHA256", "0" * 64),
            self.assertRaisesRegex(ValueError, "eval protocol SHA256 mismatch"),
        ):
            self.materialize(output_root=output)
        self.assertFalse(output.exists())

    def test_rejects_existing_output_root(self) -> None:
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaisesRegex(ValueError, "output root already exists"):
            self.materialize(output_root=output)

    def test_rejects_train_era_or_malformed_protocol_rows(self) -> None:
        cases: list[list[tuple[Any, ...]]] = [
            [("LA_0100", "LA_E_0001", "A01", 1)],
            [("LA_0100", "LA_T_0001", "A07", 1)],
            [("speaker", "LA_E_0001", "A07", 1)],
            [("LA_0100", "LA_E_0001", "A07", 0)],
            [("LA_0100", "LA_E_0001", "-", 1)],
            [("LA_0100", "LA_E_0001", "A07", 1, "extra")],
        ]
        for index, rows in enumerate(cases):
            with self.subTest(rows=rows):
                protocol = self.root / f"bad-protocol-{index}.txt"
                protocol.write_text(_protocol_text(rows), encoding="utf-8")
                self.pin(self.parquet, protocol)
                with self.assertRaisesRegex(
                    ValueError, "invalid eval protocol row|expected five protocol"
                ):
                    self.materialize(
                        protocol_path=protocol,
                        output_root=self.root / f"out-bad-{index}",
                    )

    def test_rejects_media_bytes_that_are_not_registered_flac(self) -> None:
        table = pq.read_table(self.parquet)
        audio = table.column("audio").to_pylist()
        audio[0] = {"bytes": _flac("LA_E_0001"), "path": "wrong.flac"}
        table = table.set_column(
            2, "audio", pa.array(audio, table.schema.field("audio").type)
        )
        pq.write_table(table, self.parquet)
        self.pin(self.parquet, self.protocol)
        output = self.root / "out-bad-media"
        with self.assertRaisesRegex(ValueError, "invalid embedded FLAC/path"):
            self.materialize(output_root=output)
        self.assertFalse((output / "manifest.jsonl").exists())

    def test_requires_attribution_text(self) -> None:
        empty = self.root / "empty-attribution"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "README or license"):
            self.materialize(
                attribution_root=empty, output_root=self.root / "out-no-attrib"
            )


if __name__ == "__main__":
    unittest.main()
