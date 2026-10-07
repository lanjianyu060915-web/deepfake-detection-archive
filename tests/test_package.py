"""Offline packaging integrity, deterministic output and extraction boundaries."""

import hashlib
import json
import os
import stat
import struct
import zipfile
from pathlib import Path

import pytest

from deepfake_detection_archive.submission.package import (
    BundleError,
    BundleLimits,
    JSONValue,
    package_bundle,
    verify_bundle,
)


META: dict[str, JSONValue] = {
    "official_adapter_status": "pending",
    "offline_rehearsal_status": "not_run",
}


@pytest.fixture
def asset(tmp_path: Path) -> Path:
    path = tmp_path / "model.bin"
    path.write_bytes(b"local model checkpoint fixture")
    return path


def _bundle(tmp_path: Path, asset: Path) -> Path:
    path = tmp_path / "bundle.zip"
    package_bundle(path, {"weights/model.bin": asset}, metadata=META)
    return path


def _rewrite(path: Path, operation: str) -> Path:
    with zipfile.ZipFile(path) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(contents["MANIFEST.json"])
    if operation == "payload":
        contents["weights/model.bin"] = b"x" * len(contents["weights/model.bin"])
    elif operation == "missing":
        del contents["weights/model.bin"]
    elif operation == "extra":
        contents["extra.txt"] = b"extra"
    elif operation == "hash":
        manifest["files"][0]["sha256"] = "0" * 64
    elif operation == "size":
        manifest["files"][0]["size"] += 1
    elif operation == "type":
        manifest["files"][0]["type"] = "symlink"
    elif operation == "duplicate-entry":
        manifest["files"].append(manifest["files"][0])
    elif operation == "traversal-entry":
        manifest["files"][0]["path"] = "../escape"
    elif operation == "metadata":
        del manifest["metadata"]["official_adapter_status"]
    elif operation == "empty":
        manifest["files"] = []
    contents["MANIFEST.json"] = json.dumps(manifest).encode()
    result = path.with_name("tampered.zip")
    with zipfile.ZipFile(result, "w") as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    return result


def test_reproducible_bytes_and_verified_extract(tmp_path: Path, asset: Path) -> None:
    config = tmp_path / "settings.json"
    config.write_text('{"device":"cpu"}')
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    files = {"weights/model.bin": asset, "config/settings.json": config}
    result = package_bundle(first, files, metadata=META)
    os.utime(asset, (1, 1))
    package_bundle(
        second,
        dict(reversed(list(files.items()))),
        metadata=dict(reversed(list(META.items()))),
    )
    assert first.read_bytes() == second.read_bytes()
    assert result["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    output = tmp_path / "extracted"
    report = verify_bundle(
        first, expected_sha256=str(result["sha256"]), extract_to=output
    )
    assert report["file_count"] == 2
    assert report["integrity_verified"] is True
    assert (output / "weights/model.bin").read_bytes() == asset.read_bytes()
    assert (output / "MANIFEST.json").is_file()
    manifest = json.loads((output / "MANIFEST.json").read_text())
    assert manifest["metadata"] == META
    with zipfile.ZipFile(first) as archive:
        assert all(
            info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist()
        )


@pytest.mark.parametrize(
    "name",
    [
        "/absolute",
        "../escape",
        "a/../escape",
        "a//b",
        "./a",
        "a\\b",
        "C:/a",
        "data/a",
        "archive/a",
        "cache/a",
        "MANIFEST.json",
        "manifest.JSON/a",
        "NUL.txt",
        "a.",
        "a ",
        "a\x00b",
    ],
)
def test_unsafe_asset_names_rejected(tmp_path: Path, asset: Path, name: str) -> None:
    with pytest.raises(BundleError):
        package_bundle(tmp_path / "bad.zip", {name: asset}, metadata=META)
    assert not (tmp_path / "bad.zip").exists()


@pytest.mark.parametrize(
    "operation",
    [
        "payload",
        "missing",
        "extra",
        "hash",
        "size",
        "type",
        "duplicate-entry",
        "traversal-entry",
        "metadata",
        "empty",
    ],
)
def test_tampering_rejected_before_extraction(
    tmp_path: Path, asset: Path, operation: str
) -> None:
    path = _rewrite(_bundle(tmp_path, asset), operation)
    destination = tmp_path / "must-not-exist"
    with pytest.raises(BundleError):
        verify_bundle(path, extract_to=destination)
    assert not destination.exists()


def test_case_and_prefix_collisions_rejected(tmp_path: Path, asset: Path) -> None:
    for paths in ({"A": asset, "a": asset}, {"a": asset, "a/b": asset}):
        with pytest.raises(BundleError, match=r"collision|colliding"):
            package_bundle(tmp_path / "bad.zip", paths, metadata=META)


def test_metadata_statuses_never_default_to_pass(tmp_path: Path, asset: Path) -> None:
    with pytest.raises(BundleError, match="official_adapter_status"):
        package_bundle(tmp_path / "bad.zip", {"model": asset}, metadata={})
    with pytest.raises(BundleError, match="finite JSON"):
        package_bundle(
            tmp_path / "bad.zip",
            {"model": asset},
            metadata={**META, "nan": float("nan")},
        )


def test_symlink_and_excluded_sources_rejected(tmp_path: Path, asset: Path) -> None:
    linked = tmp_path / "link.bin"
    linked.symlink_to(asset)
    with pytest.raises(BundleError, match="symlinks"):
        package_bundle(tmp_path / "bad.zip", {"model": linked}, metadata=META)
    data = tmp_path / "data"
    data.mkdir()
    private = data / "private.bin"
    private.write_bytes(b"data")
    with pytest.raises(BundleError, match="excluded source"):
        package_bundle(tmp_path / "bad.zip", {"renamed": private}, metadata=META)


def test_output_and_extraction_never_overwrite(tmp_path: Path, asset: Path) -> None:
    path = _bundle(tmp_path, asset)
    original = path.read_bytes()
    with pytest.raises(BundleError, match="already exists"):
        package_bundle(path, {"model": asset}, metadata=META)
    assert path.read_bytes() == original
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(BundleError, match="must not exist"):
        verify_bundle(path, extract_to=destination)


def test_size_and_count_limits_apply_to_create_and_verify(
    tmp_path: Path, asset: Path
) -> None:
    with pytest.raises(BundleError, match="size limit"):
        package_bundle(
            tmp_path / "bad.zip",
            {"model": asset},
            metadata=META,
            limits=BundleLimits(max_file_bytes=1),
        )
    assert not (tmp_path / "bad.zip").exists()
    path = _bundle(tmp_path, asset)
    with pytest.raises(BundleError, match="size limit"):
        verify_bundle(path, limits=BundleLimits(max_file_bytes=1))
    with pytest.raises(BundleError, match="max_files"):
        package_bundle(
            tmp_path / "bad.zip",
            {"a": asset, "b": asset},
            metadata=META,
            limits=BundleLimits(max_files=1),
        )


@pytest.mark.parametrize(
    "mode", ["traversal", "symlink", "duplicate", "compressed", "duplicate-json"]
)
def test_hostile_zip_structures_rejected(tmp_path: Path, mode: str) -> None:
    path = tmp_path / "hostile.zip"
    with zipfile.ZipFile(path, "w") as archive:
        if mode == "traversal":
            archive.writestr("../escape", b"bad")
        elif mode == "symlink":
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, b"/etc/passwd")
        elif mode == "duplicate":
            archive.writestr("same", b"a")
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("same", b"b")
        elif mode == "compressed":
            archive.writestr("bomb", b"0" * 1000000, compress_type=zipfile.ZIP_DEFLATED)
        else:
            archive.writestr("MANIFEST.json", b'{"files":[],"files":[]}')
    with pytest.raises(BundleError):
        verify_bundle(path, extract_to=tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "escape").exists()


def test_trusted_hash_detects_self_consistent_replacement(
    tmp_path: Path, asset: Path
) -> None:
    path = _bundle(tmp_path, asset)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    asset.write_bytes(b"different checkpoint")
    replacement = tmp_path / "replacement.zip"
    package_bundle(replacement, {"weights/model.bin": asset}, metadata=META)
    assert verify_bundle(replacement)["integrity_verified"] is True
    with pytest.raises(BundleError, match="trusted expected hash"):
        verify_bundle(replacement, expected_sha256=expected)


def test_directory_allocation_is_bounded_before_parsing(
    tmp_path: Path, asset: Path
) -> None:
    path = _bundle(tmp_path, asset)
    payload = bytearray(path.read_bytes())
    end = payload.rfind(b"PK\x05\x06")
    struct.pack_into("<L", payload, end + 12, 2**31)
    path.write_bytes(payload)
    with pytest.raises(BundleError, match="central directory exceeds"):
        verify_bundle(path)


def test_symlink_parent_blocks_extraction(tmp_path: Path, asset: Path) -> None:
    path = _bundle(tmp_path, asset)
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink = tmp_path / "shortcut"
    symlink.symlink_to(outside, target_is_directory=True)
    with pytest.raises(BundleError, match="symlinks"):
        verify_bundle(path, extract_to=symlink / "extracted")
    assert not (outside / "extracted").exists()
