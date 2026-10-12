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


@pytest.fixture
def eval_inputs(inputs: dict[str, Path]) -> dict[str, Any]:
    MODULE["prepare"](**inputs)
    root = inputs["output"]
    records = load_manifest(root / "manifest.jsonl")
    # Only dev real media remain accessible: evaluating must not read known
    # training/holdout audio or even the already-seen dev generators.
    for record in records:
        if record.split != "dev" or record.label != 0:
            (root / record.path).unlink()
    rows = pq.read_table(inputs["parquet_dir"] / "train.parquet").to_pylist()
    unseen = []
    for row in rows:
        if row["generative_model"] == "real":
            unseen.extend(
                {
                    **row,
                    "generative_model": model,
                    "attack_type": "tta",
                    "label": "fake",
                }
                for model in ("audiolcm", "tangoflux")
            )
    dev_names = {
        f"{r.sample_id.removeprefix('envsdd:').removesuffix('-real-real')}.wav"
        for r in records
        if r.split == "dev" and r.label == 0
    }
    # Broken unselected audio also proves it never enters the normalizer.
    for row in unseen:
        if row["filename"] not in dev_names:
            row["audio"] = {"bytes": b"must not decode", "path": row["filename"]}
    parquet = root.parent / "remain"
    parquet.mkdir()
    pq.write_table(pa.Table.from_pylist(unseen), parquet / "remain.parquet")
    return {
        "parquet_dir": parquet,
        "manifest": root / "manifest.jsonl",
        "data_root": root,
        "split": "dev",
        "output": root.parent / "eval",
    }


def test_eval_preserves_selected_split_and_real_media(
    eval_inputs: dict[str, Any],
) -> None:
    summary = MODULE["prepare_eval"](**eval_inputs)
    records = load_manifest(eval_inputs["output"] / "manifest.jsonl")
    known = load_manifest(eval_inputs["manifest"])
    reals = [r for r in known if r.split == "dev" and r.label == 0]
    assert len(records) == 3 * len(reals)
    assert [r for r in records if r.label == 0] == reals
    assert {r.split for r in records} == {"dev"}
    assert {r.generator for r in records if r.label == 1} == {"audiolcm", "tangoflux"}
    assert validate_splits(records + [r for r in known if r.split != "dev"]) == []
    for real in reals:
        assert (eval_inputs["output"] / real.path).read_bytes() == (
            eval_inputs["data_root"] / real.path
        ).read_bytes()
        assert not (eval_inputs["output"] / real.path).is_symlink()
        children = [r for r in records if r.parent_id == real.sample_id]
        assert len(children) == 2
        assert all(r.group_id == real.group_id for r in children)
        assert all(r.original_id == real.original_id for r in children)
    for record in records:
        signal, rate = sf.read(eval_inputs["output"] / record.path)
        assert rate == 16000
        assert signal.shape == (64000,)
    assert summary["normalized_coverage"] == 1
    assert summary["decode_failed"] == 0


def test_eval_decode_failure_keeps_denominator(eval_inputs: dict[str, Any]) -> None:
    selected = next(
        r
        for r in load_manifest(eval_inputs["manifest"])
        if r.split == "dev" and r.label == 0
    )
    filename = (
        selected.sample_id.removeprefix("envsdd:").removesuffix("-real-real") + ".wav"
    )
    path = eval_inputs["parquet_dir"] / "remain.parquet"
    rows = pq.read_table(path).to_pylist()
    broken = next(row for row in rows if row["filename"] == filename)
    broken["audio"] = {"bytes": b"broken wave", "path": filename}
    pq.write_table(pa.Table.from_pylist(rows), path)
    summary = MODULE["prepare_eval"](**eval_inputs)
    records = load_manifest(eval_inputs["output"] / "manifest.jsonl")
    failed = [r for r in records if r.status == "decode_failed"]
    assert len(failed) == 1
    assert failed[0].parent_id == selected.sample_id
    assert failed[0].split == "dev"
    assert failed[0].failure_reason != "unknown"
    assert len(records) == summary["real_samples"] * 3
    assert summary["normalized_coverage"] == (len(records) - 1) / len(records)
    assert summary["decode_failed"] == 1
    assert load_manifest(eval_inputs["output"] / "inventory.jsonl") == records


def test_eval_missing_member_prevents_manifest_publication(
    eval_inputs: dict[str, Any],
) -> None:
    path = eval_inputs["parquet_dir"] / "remain.parquet"
    rows = [
        row
        for row in pq.read_table(path).to_pylist()
        if row["generative_model"] != "tangoflux"
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="missing unseen-generator members"):
        MODULE["prepare_eval"](**eval_inputs)
    assert not (eval_inputs["output"] / "manifest.jsonl").exists()
