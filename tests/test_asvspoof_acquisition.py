"""Miniature archives exercise extraction safety; no official data are fetched."""

import hashlib
import json
import runpy
import stat
import zipfile
from pathlib import Path
from typing import Any

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "finalize_asvspoof2019.py"
MODULE: dict[str, Any] = runpy.run_path(str(SCRIPT))
finalize = MODULE["finalize_archive"]
STATE = finalize.__globals__


def make_archive(
    path: Path,
    extras: list[tuple[str | zipfile.ZipInfo, bytes]] | None = None,
    prefix: str = "LA/",
) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            prefix + "ASVspoof2019_LA_train/flac/LA_T_0001.flac", b"train audio"
        )
        archive.writestr(
            prefix + "ASVspoof2019_LA_dev/flac/LA_D_0001.flac", b"dev audio"
        )
        for protocol in STATE["PROTOCOLS"]:
            archive.writestr(prefix + protocol, b"protocol fixture\n")
        archive.writestr(prefix + "README.txt", b"readme")
        archive.writestr(
            prefix + "ASVspoof2019_LA_eval/flac/LA_E_0001.flac", b"never read"
        )
        archive.writestr(
            prefix + "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
            b"never read protocol",
        )
        for name, content in extras or []:
            archive.writestr(name, content)
    return path


def trust_fixture(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Test-only replacement; the real CLI offers no checksum/count bypass.
    monkeypatch.setitem(STATE, "OFFICIAL_SIZE", path.stat().st_size)
    monkeypatch.setitem(
        STATE,
        "OFFICIAL_MD5",
        hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest(),
    )
    monkeypatch.setitem(STATE, "EXPECTED_MEDIA_COUNTS", {"train": 1, "dev": 1})


def test_success_publishes_only_selected_entries_without_opening_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_archive(tmp_path / "LA.zip.part")
    trust_fixture(source, monkeypatch)
    actual_open = zipfile.ZipFile.open
    opened: list[str] = []

    def tracked_open(
        self: zipfile.ZipFile, name: Any, *args: Any, **kwargs: Any
    ) -> Any:
        filename = name.filename if isinstance(name, zipfile.ZipInfo) else name
        assert "eval" not in filename, "eval entry content must never be read"
        opened.append(filename)
        return actual_open(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", tracked_open)
    output = tmp_path / "extracted"
    report = finalize(source, output)
    assert report["archive_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["archive_name"] == "LA.zip.part"
    assert report["dataset_root"] == "LA"
    assert report["media_counts"] == {"train": 1, "dev": 1}
    assert report["protocol_count"] == 2
    assert report["selected_file_count"] == 5
    assert report["eval_media_extracted"] is False
    assert report["eval_member_contents_read"] is False
    assert report["decode_verified"] is False
    assert report["protocol_contents_validated"] is False
    assert len(opened) == 5
    assert source.exists()
    assert not (output / "LA/ASVspoof2019_LA_eval").exists()
    assert json.loads((output / "acquisition.json").read_text()) == report
    assert not list(tmp_path.glob(".extracted.staging-*"))


@pytest.mark.parametrize("corruption", ["size", "md5"])
def test_official_checksum_is_enforced_before_zip_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    source = make_archive(tmp_path / "LA.zip")
    trust_fixture(source, monkeypatch)
    if corruption == "size":
        monkeypatch.setitem(STATE, "OFFICIAL_SIZE", source.stat().st_size + 1)
    else:
        monkeypatch.setitem(STATE, "OFFICIAL_MD5", "0" * 32)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("unverified archive must not be inspected as ZIP")

    monkeypatch.setitem(STATE, "_selected_members", forbidden)
    with pytest.raises(ValueError, match="official"):
        finalize(source, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()
    assert not list(tmp_path.glob(".*.staging-*"))


@pytest.mark.parametrize(
    "name",
    [
        "../outside",
        "/absolute",
        "LA/../outside",
        "LA\\outside",
        "C:/outside",
        "LA//outside",
    ],
)
def test_unsafe_zip_paths_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    source = make_archive(tmp_path / "LA.zip", [(name, b"unsafe")])
    trust_fixture(source, monkeypatch)
    with pytest.raises(ValueError, match="unsafe ZIP path"):
        finalize(source, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()


def test_symlink_and_encrypted_entries_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = zipfile.ZipInfo("LA/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    source = make_archive(tmp_path / "symlink.zip", [(link, b"../outside")])
    trust_fixture(source, monkeypatch)
    with pytest.raises(ValueError, match="symlink"):
        finalize(source, tmp_path / "symlink-out")
    encrypted = make_archive(tmp_path / "encrypted.zip")
    data = bytearray(encrypted.read_bytes())
    # Mark the first member encrypted in both its local and central headers.
    data[data.index(b"PK\x03\x04") + 6] |= 1
    data[data.index(b"PK\x01\x02") + 8] |= 1
    encrypted.write_bytes(data)
    trust_fixture(encrypted, monkeypatch)
    with pytest.raises(ValueError, match="encrypted"):
        finalize(encrypted, tmp_path / "encrypted-out")


def test_duplicate_and_file_directory_collision_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.warns(UserWarning, match="Duplicate name"):
        source = make_archive(
            tmp_path / "duplicate.zip", [("LA/README.txt", b"second")]
        )
    trust_fixture(source, monkeypatch)
    with pytest.raises(ValueError, match="duplicate"):
        finalize(source, tmp_path / "duplicate-out")
    source = make_archive(
        tmp_path / "collision.zip", [("LA", b"file blocks directory")]
    )
    trust_fixture(source, monkeypatch)
    with pytest.raises(ValueError, match="prefix collision"):
        finalize(source, tmp_path / "collision-out")


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_existing_target_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    source = make_archive(tmp_path / "LA.zip")
    trust_fixture(source, monkeypatch)
    output = tmp_path / "existing"
    if kind == "directory":
        output.mkdir()
    elif kind == "file":
        output.write_bytes(b"original")
    else:
        output.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError, match="must not exist"):
        finalize(source, output)
    assert output.exists() or output.is_symlink()


@pytest.mark.parametrize(
    ("constant", "value", "reason"),
    [
        ("MAX_MEMBER_BYTES", 10, "single-file"),
        ("MAX_EXTRACTED_BYTES", 20, "total extraction"),
        ("MAX_ZIP_ENTRIES", 3, "entry budget"),
    ],
)
def test_budgets_checked_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    value: int,
    reason: str,
) -> None:
    source = make_archive(tmp_path / "LA.zip")
    trust_fixture(source, monkeypatch)
    monkeypatch.setitem(STATE, constant, value)
    with pytest.raises(ValueError, match=reason):
        finalize(source, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()


def test_unexpected_counts_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_archive(tmp_path / "LA.zip")
    trust_fixture(source, monkeypatch)
    monkeypatch.setitem(STATE, "EXPECTED_MEDIA_COUNTS", {"train": 2, "dev": 1})
    with pytest.raises(ValueError, match="media counts"):
        finalize(source, tmp_path / "extracted")


def test_crc_error_cleans_staging_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_archive(tmp_path / "LA.zip")
    source.write_bytes(source.read_bytes().replace(b"train audio", b"wrong audio"))
    trust_fixture(source, monkeypatch)
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        finalize(source, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()
    assert not list(tmp_path.glob(".extracted.staging-*"))


def test_publication_race_does_not_replace_newly_created_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_archive(tmp_path / "LA.zip")
    trust_fixture(source, monkeypatch)
    original_publish = STATE["_publish"]

    def racing_publish(stage: Path, target: Path) -> None:
        target.mkdir()
        original_publish(stage, target)

    monkeypatch.setitem(STATE, "_publish", racing_publish)
    with pytest.raises(FileExistsError):
        finalize(source, tmp_path / "extracted")
    assert list((tmp_path / "extracted").iterdir()) == []
    assert not list(tmp_path.glob(".extracted.staging-*"))


def test_original_prefix_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_archive(tmp_path / "LA.zip", prefix="bundle/LA/")
    trust_fixture(source, monkeypatch)
    report = finalize(source, tmp_path / "extracted")
    assert report["dataset_root"] == "bundle/LA"
    assert (
        tmp_path / "extracted/bundle/LA/ASVspoof2019_LA_train/flac/LA_T_0001.flac"
    ).is_file()


def test_cli_has_no_checksum_bypass(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        MODULE["main"](
            [
                "--archive",
                str(tmp_path / "x"),
                "--output",
                str(tmp_path / "y"),
                "--skip-checksum",
            ]
        )


def test_cli_reports_validation_failure_as_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = make_archive(tmp_path / "LA.zip")
    assert (
        MODULE["main"](["--archive", str(source), "--output", str(tmp_path / "out")])
        == 2
    )
    diagnostic = json.loads(capsys.readouterr().err)
    assert diagnostic["error"] == "ValueError"
    assert "official" in diagnostic["message"]


def test_archive_mutation_during_extraction_aborts_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_archive(tmp_path / "LA.zip")
    trust_fixture(source, monkeypatch)
    original_extract = STATE["_extract_member"]

    def mutate_after_extract(*args: Any, **kwargs: Any) -> int:
        size: int = original_extract(*args, **kwargs)
        source.touch()
        return size

    monkeypatch.setitem(STATE, "_extract_member", mutate_after_extract)
    with pytest.raises(ValueError, match="archive changed"):
        finalize(source, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()
    assert not list(tmp_path.glob(".extracted.staging-*"))
