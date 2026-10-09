"""Real decoder and waveform-diagnostic checks for the scaled ASV preparation."""

import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from deepfake_detection_archive.data.manifest import (
    SampleRecord,
    load_manifest,
    write_manifest,
)


MODULE: dict[str, Any] = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools" / "prepare_asv_main.py")
)


def test_preparation_preserves_rows_and_reports_failed_decode(tmp_path: Path) -> None:
    sf.write(tmp_path / "ok.flac", np.linspace(-0.5, 0.5, 1600), 16000)
    (tmp_path / "broken.flac").write_bytes(b"not audio")
    records = [
        SampleRecord(
            sample_id=name,
            path=f"{name}.flac",
            modality="audio",
            label=index,
            source="fixture",
            split="train" if index == 0 else "dev",
            generator="human" if index == 0 else "A01",
            group_id=f"speaker-{index}",
        )
        for index, name in enumerate(("ok", "broken"))
    ]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(records, manifest)
    sources = tmp_path / "sources.jsonl"
    sources.write_text('{"existing":"registered source"}\n', encoding="utf-8")
    output = tmp_path / "out"
    summary = MODULE["prepare"](manifest, tmp_path, sources, output, 2)
    decoded = {r.sample_id: r for r in load_manifest(output / "manifest-decoded.jsonl")}
    assert summary["total"] == 2
    assert len(summary["decode_failures"]) == 1
    assert decoded["ok"].duration_seconds == pytest.approx(0.1)
    assert decoded["ok"].sample_rate == 16000
    assert decoded["ok"].decode_status == "ok"
    assert decoded["broken"].decode_status == "failed"
    assert decoded["broken"].failure_reason != "unknown"
    for record in records:
        updated = decoded[record.sample_id]
        assert (updated.split, updated.label, updated.group_id, updated.generator) == (
            record.split,
            record.label,
            record.group_id,
            record.generator,
        )
    assert (output / "sources.jsonl").read_bytes() == sources.read_bytes()


def test_pcm_diagnostic_recognizes_identity_and_time_shift() -> None:
    waveform = np.random.default_rng(42).normal(size=1024).astype(np.float32)
    waveform -= waveform.mean()
    same = MODULE["compare_pcm"](waveform, waveform)
    shifted = MODULE["compare_pcm"](waveform, np.pad(waveform, (32, 0)))
    assert same["identical_pcm"] is True
    assert same["pcm_float32_sha256"][0] == same["pcm_float32_sha256"][1]
    assert same["max_absolute_normalized_cross_correlation"] == pytest.approx(1.0)
    assert shifted["identical_pcm"] is False
    assert shifted["max_absolute_normalized_cross_correlation"] == pytest.approx(1.0)
    assert abs(shifted["peak_lag_samples"]) == 32
    assert shifted["peak_overlap_samples"] == 1024
