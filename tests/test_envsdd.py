"""EnvSDD family import, attribution isolation and failed-media accounting."""

import csv
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
    SampleRecord,
    load_manifest,
    write_manifest,
)
from deepfake_detection_archive.data.split import validate_splits


MODULE: dict[str, Any] = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/prepare_envsdd.py")
)


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    parquet = tmp_path / "parquet"
    parquet.mkdir()
    metadata = []
    rows = []
    for index in range(8):
        filename = f"{100 + index}-0-0-0.wav"
        metadata.append(
            {
                "slice_file_name": filename,
                "fsID": str(100 + index),
                "start": "0",
                "end": "4",
                "fold": "1",
                "class": "dog_bark",
            }
        )
        for ordinal, (model, attack) in enumerate(sorted(MODULE["MEMBERS"])):
            stream = io.BytesIO()
            # Different signals avoid accidental exact-hash linkage in split tests.
            rate = 8000 if ordinal % 2 else 16000
            signal = np.sin(
                np.arange(4 * rate) * (0.01 + index * 0.003 + ordinal * 0.0003)
            )
            sf.write(stream, signal, rate, format="WAV", subtype="PCM_16")
            rows.append(
                {
                    "dataset": "UrbanSound8K",
                    "filename": filename,
                    "generative_model": model,
                    "attack_type": attack,
                    "label": "real" if model == "real" else "fake",
                    "audio": {"bytes": stream.getvalue(), "path": filename},
                    "event": "dog_bark",
                    "scene": "urban",
                    "caption": "dog",
                }
            )
    rows.append({**rows[0], "dataset": "TUT", "filename": "not-urban.wav"})
    pq.write_table(pa.Table.from_pylist(rows), parquet / "train.parquet")
    urban = tmp_path / "urban.csv"
    with urban.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    credits_path = tmp_path / "credits.txt"
    credits_path.write_text(
        "Header\n" + "\n".join(f"{100 + i} by User{i}" for i in range(8))
    )
    exclude = tmp_path / "old.jsonl"
    write_manifest(
        [
            SampleRecord(
                sample_id="unrelated",
                path="old.wav",
                modality="audio",
                label=0,
                source="old",
                original_id="freesound:999",
            )
        ],
        exclude,
    )
    return {
        "parquet_dir": parquet,
        "credits_path": credits_path,
        "urban_metadata": urban,
        "exclude_manifest": exclude,
        "output": tmp_path / "out",
    }


def test_import_preserves_families_and_symmetric_audio(inputs: dict[str, Path]) -> None:
    summary = MODULE["prepare"](**inputs)
    output = inputs["output"]
    records = load_manifest(output / "manifest.jsonl")
    assert len(records) == 40
    assert validate_splits(records) == []
    assert summary["input_dataset_counts"] == {"UrbanSound8K": 40, "TUT": 1}
    assert summary["decode_failed"] == 0
    assert summary["competition_approved"] is False
    for original in {record.original_id for record in records}:
        family = [record for record in records if record.original_id == original]
        assert len(family) == 5
        assert sum(record.label == 0 for record in family) == 1
        assert len({record.split for record in family}) == 1
    for record in records:
        waveform, rate = sf.read(output / record.path)
        assert rate == 16000
        assert waveform.shape == (64000,)
        assert np.max(np.abs(waveform)) == pytest.approx(0.95, abs=1 / 32768)
    with (output / "provenance.jsonl").open() as stream:
        provenance = [json.loads(line) for line in stream]
    assert len(provenance) == 40
    assert all((output / row["raw_path"]).is_file() for row in provenance)
    assert all(len(row["raw_sha256"]) == 64 for row in provenance)


def test_short_and_previous_relations_exclude_whole_families(
    inputs: dict[str, Path],
) -> None:
    metadata = (
        inputs["urban_metadata"]
        .read_text()
        .replace("100-0-0-0.wav,100,0,4", "100-0-0-0.wav,100,0,3.5")
    )
    inputs["urban_metadata"].write_text(metadata)
    write_manifest(
        [
            SampleRecord(
                sample_id="old1",
                path="old1.wav",
                modality="audio",
                label=0,
                source="old",
                original_id="freesound:101",
            ),
            SampleRecord(
                sample_id="old2",
                path="old2.wav",
                modality="audio",
                label=0,
                source="old",
                group_id="freesound-uploader:USER2",
            ),
        ],
        inputs["exclude_manifest"],
    )
    summary = MODULE["prepare"](**inputs)
    assert summary["eligible_samples"] == 25
    assert summary["failure_reasons"] == {
        "original_slice_shorter_than_3.99s": 5,
        "overlap_with_previous_dataset": 10,
    }
    inventory = load_manifest(inputs["output"] / "inventory.jsonl")
    assert len(inventory) == 40
    assert all(
        record.status == "excluded"
        for record in inventory
        if record.original_id in {"freesound:100", "freesound:101", "freesound:102"}
    )


def test_decode_failure_and_missing_member_stay_in_inventory(
    inputs: dict[str, Path],
) -> None:
    path = inputs["parquet_dir"] / "train.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["audio"]["bytes"] = b"broken wave"
    del rows[5]
    pq.write_table(pa.Table.from_pylist(rows), path)
    summary = MODULE["prepare"](**inputs)
    assert summary["urban_samples"] == 39
    assert summary["eligible_samples"] == 30
    assert summary["decode_attempted"] == 39
    assert summary["decode_failed"] == 1
    assert summary["incomplete_families"] == 1
    inventory = load_manifest(inputs["output"] / "inventory.jsonl")
    failed = [record for record in inventory if record.status == "decode_failed"]
    assert len(failed) == 1
    assert failed[0].decode_status == "failed"
    assert all(
        record.status != "available"
        for record in inventory
        if record.original_id in {"freesound:100", "freesound:101"}
    )
    assert summary["normalized_coverage"] == 30 / 39
