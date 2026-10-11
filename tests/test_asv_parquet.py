"""Pinned miniature mirror shards exercise raw-byte import and protocol checks."""

import hashlib
import json
import runpy
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from deepfake_detection_archive.data.manifest import load_manifest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "prepare_asv_parquet.py"
MODULE: dict[str, Any] = runpy.run_path(str(SCRIPT))
materialize = MODULE["materialize"]
STATE = materialize.__globals__


def _trust_shard(path: Path, split: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        STATE["SHARDS"],
        split,
        (path.name, hashlib.sha256(path.read_bytes()).hexdigest()),
    )


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    shards = tmp_path / "shards"
    protocols = tmp_path / "protocols"
    shards.mkdir()
    protocols.mkdir()
    for split, prefix, speaker in (("train", "T", "LA_0001"), ("dev", "D", "LA_0002")):
        rows = []
        lines = []
        for label in (0, 1):
            sample_id = f"LA_{prefix}_{label:07d}"
            system = "A01" if label else "-"
            rows.append(
                {
                    "speaker_id": speaker,
                    "audio_file_name": sample_id,
                    "audio": {
                        "bytes": b"fLaC" + sample_id.encode(),
                        "path": f"{sample_id}.flac",
                    },
                    "system_id": system,
                    "key": label,
                }
            )
            key = "spoof" if label else "bonafide"
            lines.append(f"{speaker} {sample_id} - {system} {key}")
        # Actual mirror schema, including its int64 ClassLabel and nested audio.
        metadata = {
            b"huggingface": json.dumps(
                {"info": {"features": {"key": {"names": ["bonafide", "spoof"]}}}}
            ).encode()
        }
        table = pa.Table.from_pylist(
            rows, schema=STATE["SCHEMA"].with_metadata(metadata)
        )
        path = shards / STATE["SHARDS"][split][0]
        pq.write_table(table, path, row_group_size=1)
        _trust_shard(path, split, monkeypatch)
        protocol = protocols / STATE["PROTOCOLS"][split]
        protocol.parent.mkdir(exist_ok=True)
        protocol.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (protocols / "README.LA.txt").write_bytes(b"Synthetic official attribution fixture")
    (protocols / "ASVspoof2019_LA_eval").mkdir()
    return shards, protocols


def test_materialization_preserves_bytes_and_label_mapping(
    inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shards, protocols = inputs
    output = tmp_path / "LA"
    summary = materialize(shards, protocols, output, batch_size=1)
    assert summary["split_label_counts"] == {
        "train": {"real": 1, "fake": 1},
        "dev": {"real": 1, "fake": 1},
    }
    for split, (name, checksum) in STATE["SHARDS"].items():
        for row in pq.read_table(shards / name).to_pylist():
            audio = output / f"ASVspoof2019_LA_{split}" / "flac" / row["audio"]["path"]
            assert audio.read_bytes() == row["audio"]["bytes"]
        assert summary["parquet_sha256"][name] == checksum
        relative = STATE["PROTOCOLS"][split]
        assert (output / relative).read_bytes() == (protocols / relative).read_bytes()
    assert (output / "README.LA.txt").read_bytes() == (
        protocols / "README.LA.txt"
    ).read_bytes()
    assert not (output / "ASVspoof2019_LA_eval").exists()
    assert json.loads((output / "materialization-summary.json").read_text()) == summary
    # Feed the materialized layout into the existing official manifest importer.
    # Only corpus-size expectations change for these miniature fixtures.
    importer = runpy.run_path(str(SCRIPT.with_name("prepare_asvspoof2019.py")))[
        "prepare_asvspoof2019"
    ]
    monkeypatch.setitem(
        importer.__globals__,
        "EXPECTED_COUNTS",
        {
            "train": {"speakers": 1, "real": 1, "fake": 1},
            "dev": {"speakers": 1, "real": 1, "fake": 1},
        },
    )
    imported = tmp_path / "imported"
    importer(output, imported)
    records = load_manifest(imported / "manifest.jsonl")
    assert len(records) == 4
    assert {record.label for record in records} == {0, 1}
    assert all(
        record.generator == ("A01" if record.label else "human") for record in records
    )
    assert all(
        record.sha256 == hashlib.sha256((output / record.path).read_bytes()).hexdigest()
        for record in records
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("speaker_id", "LA_9999"),
        ("audio_file_name", "LA_T_9999999"),
        ("system_id", "A02"),
        ("key", 1),
    ],
)
def test_mirror_metadata_must_match_official_protocol(
    inputs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    shards, protocols = inputs
    path = shards / STATE["SHARDS"]["train"][0]
    rows = pq.read_table(path).to_pylist()
    rows[0][field] = value
    pq.write_table(pa.Table.from_pylist(rows, schema=STATE["SCHEMA"]), path)
    _trust_shard(path, "train", monkeypatch)
    with pytest.raises(ValueError, match="metadata mismatch with protocol"):
        materialize(shards, protocols, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_fixed_shard_hash_required_before_parquet_access(
    inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shards, protocols = inputs
    path = shards / STATE["SHARDS"]["train"][0]
    with path.open("ab") as stream:
        stream.write(b"corrupted")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("unverified Parquet content must not be inspected")

    monkeypatch.setattr(pq, "ParquetFile", forbidden)
    with pytest.raises(ValueError, match="pinned Parquet SHA256 mismatch"):
        materialize(shards, protocols, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_existing_output_refused(
    inputs: tuple[Path, Path], tmp_path: Path, kind: str
) -> None:
    shards, protocols = inputs
    output = tmp_path / "existing"
    if kind == "directory":
        output.mkdir()
    elif kind == "file":
        output.write_bytes(b"preserved")
    else:
        output.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="already exists"):
        materialize(shards, protocols, output)
    if kind == "file":
        assert output.read_bytes() == b"preserved"


@pytest.mark.parametrize("violation", ["duplicate", "missing", "schema"])
def test_complete_unique_schema_matching_rows_required(
    inputs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    violation: str,
) -> None:
    shards, protocols = inputs
    path = shards / STATE["SHARDS"]["train"][0]
    table = pq.read_table(path)
    rows = table.to_pylist()
    if violation == "duplicate":
        rows[1] = rows[0]
        table = pa.Table.from_pylist(rows, schema=STATE["SCHEMA"])
        expected = "duplicate mirror"
    elif violation == "missing":
        table = table.slice(0, 1)
        expected = "complete official protocol"
    else:
        table = table.drop(["audio"])
        expected = "schema differs"
    pq.write_table(table, path)
    _trust_shard(path, "train", monkeypatch)
    with pytest.raises(ValueError, match=expected):
        materialize(shards, protocols, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("violation", ["path", "bytes"])
def test_invalid_embedded_audio_cannot_produce_completion_summary(
    inputs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    violation: str,
) -> None:
    shards, protocols = inputs
    path = shards / STATE["SHARDS"]["train"][0]
    rows = pq.read_table(path).to_pylist()
    rows[0]["audio"][violation] = "../escape.flac" if violation == "path" else b"bad"
    pq.write_table(pa.Table.from_pylist(rows, schema=STATE["SCHEMA"]), path)
    _trust_shard(path, "train", monkeypatch)
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="invalid embedded FLAC/path"):
        materialize(shards, protocols, output)
    assert not (output / "materialization-summary.json").exists()
    assert not (output / "escape.flac").exists()
