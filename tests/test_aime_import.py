"""Music prompt isolation, symmetric audio processing and failed-media accounting."""

import io
import json
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import soundfile as sf  # type: ignore[import-untyped]

from deepfake_detection_archive.data.manifest import (
    file_sha256,
    load_manifest,
    load_sources,
)
from deepfake_detection_archive.data.split import validate_splits


MODULE: dict[str, Any] = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/prepare_aime.py")
)


def _audio(frequency: float, rate: int = 16000, seconds: int = 4) -> bytes:
    waveform = np.sin(np.arange(rate * seconds) * 2 * np.pi * frequency / rate)
    stream = io.BytesIO()
    sf.write(stream, waveform, rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "parquet"
    (root / "data").mkdir(parents=True)
    rows = []
    for group in range(12):
        for ordinal, model in enumerate(
            ("MTG-Jamendo", "MusicGen Small", "Suno v3", "Udio")
        ):
            rows.append(
                {
                    "id": f"{group}-{ordinal}",
                    "model": model,
                    "description": f"genre{group}, piano, mellow"
                    if ordinal % 2
                    else f"mellow, genre{group}, piano",
                    "audio": {
                        "bytes": b"broken"
                        if model == "Udio"
                        else _audio(100 + 13 * group + ordinal),
                        "path": "source.wav",
                    },
                }
            )
    rows.append(
        {
            "id": "unmatched",
            "model": "Suno v3.5",
            "description": "absent, tags, example",
            "audio": {"bytes": b"must not decode", "path": "excluded.wav"},
        }
    )
    parquet = root / "data" / "input.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    resources = tmp_path / "resources.json"
    resources.write_text(
        json.dumps(
            {
                "repo": MODULE["REPO"],
                "revision": MODULE["REVISION"],
                "files": [
                    {
                        "path": "data/input.parquet",
                        "size": parquet.stat().st_size,
                        "sha256": file_sha256(parquet),
                        "url": "fixture",
                        "index": 0,
                    }
                ],
            }
        )
    )
    return {"parquet_dir": root, "resources": resources, "output": tmp_path / "out"}


def test_prompt_isolation_and_reserved_generators(inputs: dict[str, Path]) -> None:
    summary = MODULE["prepare"](**inputs)
    manifest = load_manifest(inputs["output"] / "manifest.jsonl")
    unseen = load_manifest(inputs["output"] / "unseen.jsonl")
    inventory = load_manifest(inputs["output"] / "inventory.jsonl")
    assert validate_splits(inventory) == []
    assert {r.generator for r in manifest} == {"unknown", "MusicGen Small"}
    assert all(r.split != "train" for r in unseen)
    assert {r.generator for r in unseen} == {"unknown", "Suno v3", "Udio"}
    assert all(r.parent_id == "unknown" for r in inventory)
    assert all(
        r.original_id == f"aime-row:{r.sample_id.removeprefix('aime:')}"
        for r in inventory
    )
    excluded = next(r for r in inventory if r.sample_id == "aime:unmatched")
    assert excluded.failure_reason == "no_matching_real_description"
    reserved = [r for r in inventory if r.generator == "Suno v3" and r.split == "train"]
    assert reserved
    assert all(r.status == "excluded" for r in reserved)
    sources = load_sources(inputs["output"] / "sources.jsonl")
    assert sources[MODULE["SOURCE"]].allowed_uses == ("training", "research")
    assert sources[MODULE["SOURCE"]].license_status == "approved"
    assert not (inputs["output"] / "sources.json").exists()
    assert summary["competition_approved"] is False
    assert summary["connected_groups"] == 12
    assert not (inputs["output"] / "raw").exists()


def test_decode_failures_stay_in_unseen_denominator(inputs: dict[str, Path]) -> None:
    summary = MODULE["prepare"](**inputs)
    unseen = load_manifest(inputs["output"] / "unseen.jsonl")
    failed = [r for r in unseen if r.generator == "Udio"]
    assert failed
    assert all(
        r.status == "decode_failed" and r.decode_status == "failed" for r in failed
    )
    assert all(not (inputs["output"] / r.path).exists() for r in failed)
    assert summary["unseen"]["status"]["decode_failed"] == len(failed)
    assert len(unseen) == 3 * len(failed)
    manifest = load_manifest(inputs["output"] / "manifest.jsonl")
    assert all(r.status == "available" for r in manifest if r.split == "train")


@pytest.mark.parametrize(("rate", "seconds", "channels"), [(8000, 2, 1), (44100, 6, 2)])
def test_normalization_is_symmetric(
    tmp_path: Path, rate: int, seconds: int, channels: int
) -> None:
    waveform = np.sin(np.arange(rate * seconds) * 2 * np.pi * 330 / rate)
    if channels == 2:
        waveform = np.column_stack((waveform * 0.4, waveform * 0.8))
    stream = io.BytesIO()
    sf.write(stream, waveform, rate, format="WAV", subtype="PCM_16")
    a = tmp_path / "real.wav"
    b = tmp_path / "fake.wav"
    metadata = MODULE["_normalize"](stream.getvalue(), a)
    MODULE["_normalize"](stream.getvalue(), b)
    assert a.read_bytes() == b.read_bytes()
    signal, actual_rate = sf.read(a)
    assert actual_rate == 16000
    assert signal.shape == (64000,)
    assert sf.info(a).subtype == "PCM_16"
    assert np.max(np.abs(signal)) == pytest.approx(0.95, abs=1 / 32768)
    assert metadata["original"]["channels"] == channels
    assert metadata["original"]["duration_seconds"] == seconds
    if seconds == 2:
        assert np.count_nonzero(signal[:16000]) == 0
        assert np.count_nonzero(signal[-16000:]) == 0


def test_wrong_official_hash_rejects_before_publication(
    inputs: dict[str, Path],
) -> None:
    resources = json.loads(inputs["resources"].read_text())
    resources["files"][0]["sha256"] = "0" * 64
    inputs["resources"].write_text(json.dumps(resources))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        MODULE["prepare"](**inputs)
    assert not (inputs["output"] / "manifest.jsonl").exists()


def test_reserved_generator_duplicate_links_prompts(inputs: dict[str, Path]) -> None:
    parquet = inputs["parquet_dir"] / "data/input.parquet"
    rows = pq.read_table(parquet).to_pylist()
    # A held-out generator duplicates a real in a different prompt group.
    rows[2]["audio"] = rows[4]["audio"]
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    resources = json.loads(inputs["resources"].read_text())
    resources["files"][0].update(
        size=parquet.stat().st_size, sha256=file_sha256(parquet)
    )
    inputs["resources"].write_text(json.dumps(resources))
    summary = MODULE["prepare"](**inputs)
    inventory = load_manifest(inputs["output"] / "inventory.jsonl")
    assert summary["connected_groups"] == 11
    assert (
        len({r.split for r in inventory if r.sample_id in {"aime:0-0", "aime:1-0"}})
        == 1
    )
    assert validate_splits(inventory) == []
