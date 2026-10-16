"""Active dependency metadata must select matching official backend wheels."""

import subprocess
import tomllib
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("backend", ["cpu", "cu126"])
def test_backend_dependencies_and_lock_sources(backend: str) -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    extras = project["project"]["optional-dependencies"]
    assert set(extras) == {"cpu", "cu126"}
    assert "numpy==2.5.3" in project["project"]["dependencies"]
    assert "transformers==5.19.0" in extras[backend]
    expected = {"torch": "2.14.1", "torchvision": "0.29.1", "torchaudio": "2.11.0"}
    for name, version in expected.items():
        assert f"{name}=={version}+{backend}" in extras[backend]
        assert project["tool"]["uv"]["sources"][name] == [
            {"index": "pytorch-cpu", "extra": "cpu"},
            {"index": "pytorch-cu126", "extra": "cu126"},
        ]
        packages = [p for p in lock["package"] if p["name"] == name]
        package = next(p for p in packages if p["version"] == f"{version}+{backend}")
        assert package["source"] == {
            "registry": f"https://download.pytorch.org/whl/{backend}"
        }
    assert "cu132" not in (ROOT / "uv.lock").read_text()


@pytest.fixture(scope="module")
def wheel_metadata(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    output = tmp_path_factory.mktemp("dependency-wheel")
    subprocess.run(
        ["uv", "--no-config", "build", "--no-sources", "--out-dir", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    wheel = next(output.glob("*.whl"))
    with ZipFile(wheel) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(name))
    assert set(metadata.get_all("Provides-Extra", [])) == {"cpu", "cu126"}
    return metadata.get_all("Requires-Dist", [])


@pytest.mark.parametrize("backend", ["cpu", "cu126"])
def test_wheel_metadata_pins_matching_backend(
    wheel_metadata: list[str], backend: str
) -> None:
    for name, version in (
        ("torch", "2.14.1"),
        ("torchvision", "0.29.1"),
        ("torchaudio", "2.11.0"),
    ):
        requirement = f"{name}=={version}+{backend} ; extra == '{backend}'"
        assert requirement in wheel_metadata
    assert all("cu132" not in requirement for requirement in wheel_metadata)
