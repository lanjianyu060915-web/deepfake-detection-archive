"""Default model packages resolve only their own local resources."""

import json

import pytest

from deepfake_detection_archive.cli import main
from deepfake_detection_archive.unified_predict import (
    _resolved_config,
    _validate_ast_frontend,
    default_models,
)


@pytest.fixture
def package(tmp_path, monkeypatch):
    root = tmp_path / "package"
    (root / "configs").mkdir(parents=True)
    (root / "weights").mkdir()
    registry = {
        "schema_version": 1,
        "runtime_contract": "release.json",
        "models": {"audio": {"config": "audio.json"}, "video": {}},
    }
    (root / "configs/models.json").write_text(json.dumps(registry))
    config = {
        "manifest": "unused",
        "data_root": "unused",
        "output_dir": "unused",
        "modality": "audio",
        "checkpoint": "../weights/audio.pt",
        "checkpoint_sha256": "a" * 64,
    }
    (root / "configs/audio.json").write_text(json.dumps(config))
    (root / "release.json").write_text(
        json.dumps({"runtime_dependencies": {"numpy": "1", "transformers": "2"}})
    )
    (root / "weights/audio.pt").write_bytes(b"checkpoint")
    monkeypatch.setenv("DEEPFAKE_MODEL_PACKAGE", str(root))
    return root, config


def resolve(root, entry=None):
    folder, registry = default_models()
    return _resolved_config(
        folder, entry or registry["models"]["audio"], root / "out", "audio", "cpu"
    )


def test_nested_config_resources_use_actual_config_directory(package):
    root, config = package
    nested = root / "configs/nested"
    nested.mkdir()
    config["checkpoint"] = "../../weights/audio.pt"
    config["encoder"] = {
        "checkpoint": "../../weights/encoder.pt",
    }
    (root / "weights/encoder.pt").write_bytes(b"encoder")
    (nested / "audio.json").write_text(json.dumps(config))
    result = resolve(root, {"config": "nested/audio.json"})
    assert result.checkpoint == str(root / "weights/audio.pt")
    assert result.encoder["checkpoint"] == str(root / "weights/encoder.pt")


def test_internal_config_and_checkpoint_symlinks_are_accepted(package):
    root, _ = package
    (root / "configs/alias.json").symlink_to("audio.json")
    (root / "weights/audio.pt").unlink()
    (root / "weights/real.pt").write_bytes(b"checkpoint")
    (root / "weights/audio.pt").symlink_to("real.pt")
    assert resolve(root, {"config": "alias.json"}).checkpoint == str(
        root / "weights/real.pt"
    )


def test_config_symlink_resources_use_target_directory(package):
    root, config = package
    nested = root / "configs/nested"
    nested.mkdir()
    config["checkpoint"] = "../../weights/audio.pt"
    (nested / "actual.json").write_text(json.dumps(config))
    (root / "configs/alias.json").symlink_to("nested/actual.json")
    assert resolve(root, {"config": "alias.json"}).checkpoint == str(
        root / "weights/audio.pt"
    )


def test_internal_audio_bundle_and_declared_file_symlinks_are_accepted(package):
    root, config = package
    model_dir = root / "weights/encoder"
    model_dir.mkdir()
    (model_dir / "model.safetensors").symlink_to("../audio.pt")
    (root / "weights/encoder-link").symlink_to("encoder", target_is_directory=True)
    config["encoder"] = {
        "model_dir": "../weights/encoder-link",
        "expected_hashes": {"model.safetensors": "a" * 64},
    }
    (root / "configs/audio.json").write_text(json.dumps(config))
    assert resolve(root).encoder["model_dir"] == str(model_dir)


def test_source_layout_allows_relative_repository_models(package, monkeypatch):
    root, config = package
    monkeypatch.delenv("DEEPFAKE_MODEL_PACKAGE")
    monkeypatch.setattr(
        "deepfake_detection_archive.unified_predict.__file__",
        str(root / "src/deepfake_detection_archive/unified_predict.py"),
    )
    release = root / "configs/release"
    release.mkdir()
    (root / "configs/models.json").replace(release / "models.json")
    config["checkpoint"] = "../../models/audio.pt"
    (release / "audio.json").write_text(json.dumps(config))
    assert resolve(root).checkpoint == str(root / "models/audio.pt")


@pytest.mark.parametrize("reference", ["absolute", "relative", "symlink"])
@pytest.mark.parametrize("resource", ["config", "checkpoint", "encoder", "model_dir"])
def test_external_package_resources_are_rejected(
    package, tmp_path, reference, resource
):
    root, config = package
    external = tmp_path / "external"
    if resource == "model_dir":
        external.mkdir()
    else:
        external.write_text(json.dumps(config))
    if reference == "absolute":
        path = str(external)
    elif reference == "relative":
        path = "../../external"
    else:
        (root / "configs/link").symlink_to(external)
        path = "link"
    entry = {"config": "audio.json"}
    if resource == "config":
        entry["config"] = path
    elif resource == "checkpoint":
        config["checkpoint"] = path
    else:
        config["encoder"] = {
            "checkpoint" if resource == "encoder" else "model_dir": path
        }
    (root / "configs/audio.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="escapes model package"):
        resolve(root, entry)


def test_registry_symlink_cannot_select_another_package(package, tmp_path):
    root, _ = package
    registry = root / "configs/models.json"
    external = tmp_path / "models.json"
    registry.replace(external)
    registry.symlink_to(external)
    with pytest.raises(ValueError, match="escapes model package"):
        default_models()


@pytest.mark.parametrize("reference", ["relative", "symlink"])
def test_show_models_obeys_package_boundary(package, tmp_path, reference, capsys):
    root, config = package
    external = tmp_path / "external.json"
    external.write_text(json.dumps(config))
    path = "../../external.json"
    if reference == "symlink":
        (root / "configs/link.json").symlink_to(external)
        path = "link.json"
    registry = root / "configs/models.json"
    value = json.loads(registry.read_text())
    value["models"]["audio"]["config"] = path
    registry.write_text(json.dumps(value))
    assert main(["predict", "--show-models"]) == 2
    assert "escapes model package" in capsys.readouterr().err


@pytest.mark.parametrize("reference", ["absolute", "relative", "symlink"])
def test_runtime_contract_cannot_escape_package(package, tmp_path, reference):
    root, _ = package
    external = tmp_path / "release.json"
    external.write_text((root / "release.json").read_text())
    path = str(external) if reference == "absolute" else "../release.json"
    if reference == "symlink":
        (root / "contract.json").symlink_to(external)
        path = "contract.json"
    folder, registry = default_models()
    registry["runtime_contract"] = path
    with pytest.raises(ValueError, match="escapes model package"):
        _validate_ast_frontend(folder, registry, {"numpy": "1", "transformers": "2"})


@pytest.mark.parametrize("declared", [False, True])
def test_audio_bundle_files_cannot_escape_package(package, tmp_path, declared):
    root, config = package
    model_dir = root / "weights/encoder"
    model_dir.mkdir()
    external = tmp_path / "external.safetensors"
    external.write_bytes(b"weights")
    filename = "declared.bin" if declared else "model.safetensors"
    (model_dir / filename).symlink_to(external)
    config["encoder"] = {
        "model_dir": "../weights/encoder",
        "expected_hashes": {filename: "b" * 64} if declared else {},
    }
    (root / "configs/audio.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="escapes model package"):
        resolve(root)
