"""Real local stream discovery and unified entry orchestration regressions."""

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from deepfake_detection_archive import media
from deepfake_detection_archive.cli import main
from deepfake_detection_archive.config import load_inference_config
from deepfake_detection_archive.media import VideoConfig, decode_video
from deepfake_detection_archive.models import LogMelCNN, LogMelConfig, save_checkpoint
from deepfake_detection_archive.scoring import OfflineScorer
from deepfake_detection_archive.unified_predict import (
    _validate_ast_frontend,
    discover,
    predict_media,
    probe_media,
)


def ffmpeg(*args):
    subprocess.run(
        ["ffmpeg", "-v", "error", *map(str, args)],
        check=True,
        capture_output=True,
        timeout=20,
    )


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "语音 space.WAV"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=0.2", path)
    return path


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    folder = tmp_path / "bundle"
    (folder / "configs").mkdir(parents=True)
    config_root = Path(__file__).resolve().parents[1] / "configs/release"
    for name in ("models.json", "audio-environment.json", "video.json"):
        shutil.copyfile(config_root / name, folder / "configs" / name)
    metadata = save_checkpoint(
        folder / "weights/audio.pt", LogMelCNN(LogMelConfig(channels=4, hidden_dim=4))
    )
    config_path = folder / "configs/audio-environment.json"
    config = json.loads(config_path.read_text())
    config.update(
        checkpoint="../weights/audio.pt",
        checkpoint_sha256=metadata["sha256"],
        segment_samples=4096,
    )
    config_path.write_text(json.dumps(config))
    registry_path = folder / "configs/models.json"
    registry = json.loads(registry_path.read_text())
    registry["models"]["audio"]["architecture"] = "logmel_cnn_v1"
    registry_path.write_text(json.dumps(registry))
    monkeypatch.setenv("DEEPFAKE_MODEL_PACKAGE", str(folder))
    return folder


def test_discovery_aliases_distinct_copies_and_errors(audio, tmp_path):
    copied = tmp_path / "other" / audio.name
    copied.parent.mkdir()
    shutil.copyfile(audio, copied)
    no_suffix = tmp_path / "no-extension"
    shutil.copyfile(audio, no_suffix)
    (tmp_path / "ordinary.txt").write_text("hello")
    (tmp_path / "broken.MP4").write_bytes(b"broken")
    (tmp_path / "link.wav").symlink_to(audio)
    output = tmp_path / "results"
    output.mkdir()
    shutil.copyfile(audio, output / "must-not-scan.wav")
    records, entries = discover([audio, tmp_path, tmp_path / "missing.wav"], output)
    assert len(records) == 3
    assert len({row.sample_id for row in records}) == 3
    assert len({row.sha256 for row in records}) == 1
    assert sum(entry["status"] == "duplicate" for entry in entries) == 1
    assert sum(entry["status"] == "error" for entry in entries) == 3
    assert {
        entry.get("reason") for entry in entries if entry["status"] == "excluded"
    } == {"non_media", "output_directory"}
    assert discover([audio, tmp_path, tmp_path / "missing.wav"], output) == (
        records,
        entries,
    )


def test_stream_routes_audio_container_cover_and_first_true_video(audio, tmp_path):
    pure_audio_mp4 = tmp_path / "pure.mp4"
    ffmpeg("-i", audio, "-c:a", "aac", pure_audio_mp4)
    assert probe_media(pure_audio_mp4) == ("audio", 0)
    cover = tmp_path / "cover.png"
    ffmpeg("-f", "lavfi", "-i", "color=red:s=32x32", "-frames:v", 1, cover)
    covered = tmp_path / "covered.mp3"
    ffmpeg(
        "-i",
        audio,
        "-i",
        cover,
        "-map",
        "0:a",
        "-map",
        "1:v",
        "-c:a",
        "libmp3lame",
        "-c:v",
        "copy",
        "-disposition:v",
        "attached_pic",
        covered,
    )
    assert probe_media(covered) == ("audio", 0)
    video = tmp_path / "multi.mkv"
    ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=red:s=32x32:d=0.2",
        "-i",
        audio,
        "-map",
        "1:a",
        "-map",
        "0:v",
        "-c:v",
        "ffv1",
        video,
    )
    assert probe_media(video) == ("video", 1)
    decoded = decode_video(
        video,
        VideoConfig(
            n_frames=1,
            height=32,
            width=32,
            stream_index=1,
            resize_mode="native_center_crop",
        ),
    )
    assert decoded.frames[0, 0, 0, 0] > 200
    assert decoded.frames[0, 0, 0, 1] < 10


def test_single_multiple_directory_scores_resume_and_identity(audio, bundle, tmp_path):
    output = tmp_path / "results"
    report = predict_media([audio], output, "cpu", resume=False)
    assert report["failures"] == 0
    row = json.loads((output / "results.jsonl").read_text())
    records, _ = discover([audio], output)
    config = replace(
        load_inference_config(bundle / "configs/audio-environment.json"),
        data_root="/",
        device="cpu",
    )
    old = OfflineScorer(config, bundle / "configs")
    assert row["score"] == old(records[0])
    before = (output / "audio/scores.jsonl").read_bytes()
    assert predict_media([audio], output, "cpu", resume=True)["failures"] == 0
    assert (output / "audio/scores.jsonl").read_bytes() == before
    copied = tmp_path / "copy.wav"
    shutil.copyfile(audio, copied)
    multi = tmp_path / "multiple"
    predict_media([audio, copied], multi, "cpu", resume=False)
    assert all(
        item["score"] == row["score"]
        for item in map(json.loads, (multi / "results.jsonl").read_text().splitlines())
    )
    with pytest.raises(ValueError, match="resume rejected"):
        predict_media([copied], output, "cpu", resume=True)
    with pytest.raises(FileExistsError):
        predict_media([audio], output, "cpu", resume=False)


def test_missing_weight_records_global_abort(audio, bundle, tmp_path):
    (bundle / "weights/audio.pt").unlink()
    output = tmp_path / "abort"
    report = predict_media([audio], output, "cpu", resume=False)
    assert report["failures"] == 1
    assert report["global_error"]
    assert json.loads((output / "results.jsonl").read_text())["status"] == "not_run"


def test_nonmedia_only_and_explicit_failure_nonzero(bundle, tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    (folder / "note.txt").write_text("hello")
    assert (
        main(
            [
                "predict",
                str(folder),
                "--output",
                str(tmp_path / "out"),
                "--device",
                "cpu",
            ]
        )
        == 2
    )
    assert main(["predict", str(folder), "--config", "anything.json"]) == 2
    assert main(["predict", "--show-models"]) == 0


def test_default_bundle_can_be_relocated(audio, bundle, tmp_path, monkeypatch):
    moved = tmp_path / "moved bundle"
    shutil.move(bundle, moved)
    monkeypatch.setenv("DEEPFAKE_MODEL_PACKAGE", str(moved))
    assert (
        predict_media([audio], tmp_path / "relocated-output", "cpu", resume=False)[
            "failures"
        ]
        == 0
    )


def test_output_is_excluded_through_parent_symlink(audio, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    shutil.copyfile(audio, output / "generated.wav")
    parent_alias = tmp_path / "parent-alias"
    parent_alias.symlink_to(tmp_path, target_is_directory=True)
    inputs = [parent_alias / "output/generated.wav", parent_alias / "output"]
    records, entries = discover(inputs, output)
    assert records == []
    assert len(entries) == 2
    assert all(entry["reason"] == "output_directory" for entry in entries)
    assert entries[0]["input"].endswith("parent-alias/output/generated.wav")


def test_parent_symlink_input_resume_excludes_generated_output(audio, bundle, tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    shutil.copyfile(audio, inputs / "source.wav")
    parent_alias = tmp_path / "parent-alias"
    parent_alias.symlink_to(tmp_path, target_is_directory=True)
    output = inputs / "results"
    paths = [parent_alias / "inputs"]
    assert predict_media(paths, output, "cpu", resume=False)["failures"] == 0
    first = json.loads((output / "discovery.json").read_text())
    shutil.copyfile(audio, output / "generated.wav")
    assert predict_media(paths, output, "cpu", resume=True)["failures"] == 0
    assert json.loads((output / "discovery.json").read_text()) == first


def test_missing_input_and_empty_directory_have_single_failure(bundle, tmp_path):
    missing = predict_media(
        [tmp_path / "missing.wav"], tmp_path / "missing-run", "cpu", resume=False
    )
    assert missing["failures"] == 1
    assert missing["global_error"] is None
    empty = tmp_path / "empty"
    empty.mkdir()
    report = predict_media([empty], tmp_path / "empty-run", "cpu", resume=False)
    assert report["failures"] == 1
    assert str(report["global_error"]).startswith("no_media:")


def test_ast_release_frontend_contract_is_narrow(bundle):
    expected = {"transformers": "5.19.0", "numpy": "2.5.3", "unrelated": "old"}
    (bundle / "release.json").write_text(json.dumps({"runtime_dependencies": expected}))
    registry = {"runtime_contract": "release.json"}
    assert (
        _validate_ast_frontend(
            bundle / "configs", registry, {**expected, "torchaudio": "installed"}
        )["frontend"]
        == "numpy_spectrogram_v1"
    )
    with pytest.raises(ValueError, match="numpy"):
        _validate_ast_frontend(
            bundle / "configs", registry, {**expected, "numpy": "different"}
        )
    kaldi = {**expected, "torch": "2", "torchaudio": "2"}
    (bundle / "release.json").write_text(json.dumps({"runtime_dependencies": kaldi}))
    assert (
        _validate_ast_frontend(bundle / "configs", registry, kaldi)["frontend"]
        == "torchaudio_kaldi_v1"
    )
    with pytest.raises(ValueError, match="torchaudio"):
        _validate_ast_frontend(
            bundle / "configs", registry, {**kaldi, "torchaudio": None}
        )
    (bundle / "release.json").write_text(
        json.dumps(
            {"runtime_dependencies": kaldi, "ast_frontend": "numpy_spectrogram_v1"}
        )
    )
    assert (
        _validate_ast_frontend(bundle / "configs", registry, kaldi)["frontend"]
        == "numpy_spectrogram_v1"
    )


def test_resume_records_media_tool_build_identity(audio, bundle, tmp_path, monkeypatch):
    output = tmp_path / "tools-run"
    predict_media([audio], output, "cpu", resume=False)
    identity = json.loads((output / "input.json").read_text())
    assert set(identity["media_tools"]) == {"ffmpeg", "ffprobe"}
    original = media._run

    def changed_version(command, *args):
        result = original(command, *args)
        return (
            result + b"changed build\n" if command == ["ffmpeg", "-version"] else result
        )

    monkeypatch.setattr(media, "_run", changed_version)
    with pytest.raises(ValueError, match="resume rejected"):
        predict_media([audio], output, "cpu", resume=True)


def test_loop_symlink_does_not_abort_other_media(audio, tmp_path):
    loop = tmp_path / "loop.wav"
    loop.symlink_to(loop.name)
    records, entries = discover([tmp_path], tmp_path / "output")
    assert len(records) == 1
    assert records[0].path.endswith(audio.name)
    failed = [entry for entry in entries if entry["status"] == "error"]
    assert len(failed) == 1
    assert failed[0]["reason"] == "restricted_symlink"


def test_legacy_numpy_ast_release_runs_with_torchaudio_installed(
    audio, bundle, tmp_path
):
    pytest.importorskip("torchaudio")
    from importlib.metadata import version  # noqa: PLC0415

    from deepfake_detection_archive.models.ast import (  # noqa: PLC0415
        ASTClassifier,
        ASTConfig,
    )

    model = ASTClassifier(
        ASTConfig(
            backbone_config={
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "intermediate_size": 24,
                "max_length": 32,
                "num_mel_bins": 32,
            },
            trainable_layers=1,
        )
    )
    metadata = save_checkpoint(bundle / "weights/ast.pt", model)
    config_path = bundle / "configs/audio-environment.json"
    config = json.loads(config_path.read_text())
    config.update(checkpoint="../weights/ast.pt", checkpoint_sha256=metadata["sha256"])
    config_path.write_text(json.dumps(config))
    registry_path = bundle / "configs/models.json"
    registry = json.loads(registry_path.read_text())
    registry["models"]["audio"]["architecture"] = "audio_ast_v1"
    registry_path.write_text(json.dumps(registry))
    dependencies = {name: version(name) for name in ("transformers", "numpy")}
    (bundle / "release.json").write_text(
        json.dumps({"runtime_dependencies": dependencies})
    )
    assert (
        predict_media([audio], tmp_path / "numpy-run", "cpu", resume=False)["failures"]
        == 0
    )
    dependencies.update({name: version(name) for name in ("torch", "torchaudio")})
    (bundle / "release.json").write_text(
        json.dumps(
            {
                "runtime_dependencies": dependencies,
                "ast_frontend": "torchaudio_kaldi_v1",
            }
        )
    )
    report = predict_media([audio], tmp_path / "wrong-contract", "cpu", resume=False)
    assert "checkpoint AST frontend differs" in report["global_error"]
    assert report["failures"] > 0
