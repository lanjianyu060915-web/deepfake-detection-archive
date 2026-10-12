"""Paired external clips retain music labels and symmetric time/processing."""

import json
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

from deepfake_detection_archive.data.manifest import load_manifest, load_sources


MODULE: dict[str, Any] = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/prepare_ai_openbmat.py")
)


def _metadata(segments: list[tuple[float, list[str]]]) -> dict[str, Any]:
    return {
        "config": {"crossfade_ms": 0},
        "segments": [
            {
                "duration": duration,
                "snr_db": -6,
                "sources": [
                    {
                        "type": kind,
                        "file": f"/original/{kind}.wav",
                        "start": 2.0,
                        "duration": duration,
                    }
                    for kind in kinds
                ],
            }
            for duration, kinds in segments
        ],
        "reference_structure": {"file": "broadcast", "segments": [{"duration": 99}]},
    }


def _pair(raw: Path, real: dict[str, Any], fake: dict[str, Any]) -> None:
    rate = 22050
    waveform = np.sin(np.arange(rate * 30) * 2 * np.pi * 333 / rate)
    stereo = np.column_stack((waveform, waveform * 0.5))
    for side, metadata in (("human", real), ("ai", fake)):
        directory = raw / side
        directory.mkdir(parents=True)
        stem = "gen_00001" if side == "human" else "gen_00001_ai"
        (directory / f"{stem}.json").write_text(json.dumps(metadata))
        sf.write(directory / f"{stem}.wav", stereo, rate, subtype="PCM_16")


def test_buckets_no_music_exclusion_and_symmetric_processing(tmp_path: Path) -> None:
    metadata = _metadata([(5, ["speech"]), (6, ["music"]), (10, ["music", "speech"])])
    raw, output = tmp_path / "raw", tmp_path / "output"
    _pair(raw, metadata, metadata)
    summary = MODULE["prepare"](raw, output)
    records = load_manifest(output / "manifest.jsonl")
    assert len(records) == 4
    assert summary["bucket_counts"] == {"mixed": 2, "music": 2}
    assert summary["competition_approved"] is False
    assert len({r.group_id for r in records}) == 1
    assert all(r.split == "dev" and r.decode_status == "ok" for r in records)
    for bucket in ("mixed", "music"):
        rows = load_manifest(output / f"{bucket}-dev-manifest.jsonl")
        assert {r.label for r in rows} == {0, 1}
        waveforms = [sf.read(output / r.path)[0] for r in rows]
        np.testing.assert_array_equal(*waveforms)
        assert waveforms[0].shape == (64000,)
        assert np.max(np.abs(waveforms[0])) == pytest.approx(0.95, abs=1 / 32768)
    provenance = [
        json.loads(line)
        for line in (output / "provenance.jsonl").read_text().splitlines()
    ]
    assert {p["start_seconds"] for p in provenance if p["bucket"] == "music"} == {6.0}
    assert {p["start_seconds"] for p in provenance if p["bucket"] == "mixed"} == {14.0}
    source = load_sources(output / "sources.jsonl")[MODULE["SOURCE"]]
    assert source.license_status == "unresolved"
    assert source.archive_sha256 == "unknown"
    assert (
        provenance[0]["pair_metadata"]["ai"]["segments"][0]["sources"][0]["file"]
        == "speech.wav"
    )


def test_intersection_uses_actual_durations_and_skips_short_segments(
    tmp_path: Path,
) -> None:
    real = _metadata([(2, ["speech"]), (7, ["music"]), (4.1, ["music", "speech"])])
    fake = _metadata([(3, ["speech"]), (6, ["music"]), (8, ["music", "speech"])])
    raw, output = tmp_path / "raw", tmp_path / "output"
    _pair(raw, real, fake)
    summary = MODULE["prepare"](raw, output)
    assert summary["bucket_counts"] == {"music": 2}
    assert summary["exclusions"] == {"mixed:no_eligible_common_segment": 1}
    provenance = [
        json.loads(line)
        for line in (output / "provenance.jsonl").read_text().splitlines()
    ]
    assert {p["start_seconds"] for p in provenance} == {4.0}


@pytest.mark.parametrize("failure", ["missing", "broken", "short"])
def test_bad_audio_is_recorded_per_sample(tmp_path: Path, failure: str) -> None:
    metadata = _metadata([(8, ["music"]), (8, ["music", "speech"])])
    raw, output = tmp_path / "raw", tmp_path / "output"
    _pair(raw, metadata, metadata)
    audio = raw / "ai/gen_00001_ai.wav"
    if failure == "missing":
        audio.unlink()
    elif failure == "broken":
        audio.write_bytes(b"broken")
    else:
        sf.write(audio, np.ones(16000), 16000)
    summary = MODULE["prepare"](raw, output)
    assert summary["status"] == {"available": 2, "decode_failed": 2}
    records = load_manifest(output / "manifest.jsonl")
    assert len(records) == 4
    assert all(
        r.status == "decode_failed" and r.failure_reason != "unknown"
        for r in records
        if r.label == 1
    )


def test_mismatched_reference_is_rejected(tmp_path: Path) -> None:
    real = _metadata([(8, ["music"])])
    fake = _metadata([(8, ["music"])])
    fake["reference_structure"]["file"] = "different-broadcast"
    raw = tmp_path / "raw"
    _pair(raw, real, fake)
    with pytest.raises(ValueError, match=r"reference_structure\.file differs"):
        MODULE["prepare"](raw, tmp_path / "output")


def test_longest_common_segment_and_music_required_on_both_sides() -> None:
    real = _metadata([(6, ["music"]), (8, ["music"]), (10, ["music", "speech"])])
    fake = _metadata([(6, ["music"]), (8, ["music"]), (10, ["speech"])])
    assert MODULE["_select"](real, fake) == {"music": (1, 8.0)}


def test_crossfade_offsets_accumulate_and_clip_stays_inside_actual_segment() -> None:
    metadata = _metadata([(3.5, ["speech"])] * 15 + [(5.0, ["music", "speech"])])
    metadata["config"]["crossfade_ms"] = 50
    intervals = MODULE["_intervals"](metadata)
    start, end, _ = intervals[15]
    assert start == pytest.approx(15 * (3.5 - 0.05))
    assert end == pytest.approx(57.5 - 15 * 0.05)
    index, clip_start = MODULE["_select"](metadata, metadata)["mixed"]
    assert index == 15
    assert clip_start == pytest.approx(52.25)
    assert clip_start >= start + 0.1
    assert clip_start + 4 <= end - 0.1
