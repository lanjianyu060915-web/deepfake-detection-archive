"""Exercise the existing offline package in a weight-free saved/loaded image."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


TOLERANCE = 1e-4


def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, **kwargs)  # noqa: S603


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scores(output: Path, media: Path) -> dict[tuple[str, str], float]:
    """Align by logical relative path AND input bytes, never ordering/basename."""
    discovery = json.loads((output / "discovery.json").read_text())["entries"]
    identities = {}
    for entry in discovery:
        if entry["status"] != "media":
            raise ValueError(f"unexpected discovery result: {entry}")
        relative = Path(entry["path"]).relative_to(media).as_posix()
        identities[entry["sample_id"]] = (relative, entry["sha256"])
    result = {}
    for line in (output / "results.jsonl").read_text().splitlines():
        row = json.loads(line)
        score = row["score"]
        if row["status"] != "ok" or not math.isfinite(score):
            raise ValueError(f"inference failure: {row}")
        identity = identities[row["sample_id"]]
        if identity in result:
            raise ValueError(f"duplicate logical input: {identity}")
        result[identity] = float(score)
    if len(result) != 4 or len(identities) != 4:
        raise ValueError("rehearsal requires all four self-generated examples")
    return result


def compare(
    reference: dict[tuple[str, str], float], actual: dict[tuple[str, str], float]
) -> float:
    if reference.keys() != actual.keys():
        raise ValueError("media logical paths or SHA256 identities differ")
    difference = max(abs(reference[key] - actual[key]) for key in reference)
    if difference > TOLERANCE:
        raise ValueError(f"score difference {difference} exceeds {TOLERANCE}")
    return difference


def rehearse(  # noqa: PLR0915 -- explicit sequential build/run/save/load rehearsal
    package: Path, dockerfile: Path, output: Path, backend: str
) -> None:
    package, output = package.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    work = output.parent / f"container-work-{backend}"
    work.mkdir(exist_ok=False)
    models = work / "models"
    models.mkdir()
    for name in ("configs", "weights"):
        shutil.copytree(package / name, models / name)
    shutil.copyfile(package / "release.json", models / "release.json")
    cache = work / "cache"
    cache.mkdir()
    reference_output = output / "reference"
    run(
        [
            str(package / "run.sh"),
            "predict",
            str(package / "media"),
            "--device",
            "cpu",
            "--output",
            str(reference_output),
        ],
        env={**os.environ, "DEEPFAKE_MODEL_PACKAGE": str(models)},
    )
    reference = scores(reference_output, package / "media")
    image = f"deepfake-rehearsal:{backend}"
    context = work / "context"
    context.mkdir()
    shutil.copyfile(dockerfile, context / "Dockerfile")
    # Move, rather than duplicate, the multi-GB runtime for the build context.
    shutil.move(str(package), context / "package")
    run(["docker", "build", "--target", "runtime", "-t", image, str(context)])
    image_info = json.loads(
        run(["docker", "image", "inspect", image], capture_output=True).stdout
    )[0]
    image_id = image_info["Id"]
    # Runtime-stage payload explicitly excludes all model assets and inputs.
    run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            "test ! -e /app/weights && test ! -e /app/configs "
            "&& test ! -e /app/media && test ! -e /models",
        ]
    )
    media = context / "package/media"

    def infer(name: str) -> dict[tuple[str, str], float]:
        destination = output / name
        destination.mkdir()
        run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=512m",  # noqa: S108 -- container tmpfs
                "--mount",
                f"type=bind,src={models},dst=/models,readonly",
                "--mount",
                f"type=bind,src={media},dst=/inputs,readonly",
                "--mount",
                f"type=bind,src={cache},dst=/cache",
                "--mount",
                f"type=bind,src={destination},dst=/outputs",
                image,
                "predict",
                "/inputs",
                "--device",
                "cpu",
                "--output",
                "/outputs/run",
            ]
        )
        return scores(destination / "run", Path("/inputs"))

    first = infer("container")
    initial_delta = compare(reference, first)
    archive = output / f"runtime-image-{backend}.tar.gz"
    with (
        archive.open("wb") as stream,
        gzip.GzipFile(fileobj=stream, mode="wb", compresslevel=1, mtime=0) as zipped,
    ):
        process = subprocess.Popen(["docker", "save", image], stdout=subprocess.PIPE)  # noqa: S603,S607
        assert process.stdout is not None  # noqa: S101 -- PIPE guarantees a stream
        shutil.copyfileobj(process.stdout, zipped)
        process.stdout.close()
        if process.wait():
            raise RuntimeError("docker save failed")
    archive_hash = sha256(archive)
    run(["docker", "image", "rm", image])
    run(["docker", "load", "--input", str(archive)])
    loaded_id = run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
    ).stdout.strip()
    if loaded_id != image_id:
        raise ValueError("image config digest changed after save/load")
    shutil.rmtree(cache)
    cache.mkdir()
    repeated = infer("loaded-container")
    loaded_delta = compare(reference, repeated)
    repeat_delta = compare(first, repeated)
    metadata = json.loads((context / "package/release.json").read_text())
    shutil.copyfile(
        context / "package/runtime/media-tools/build.json",
        output / "media-tools-build.json",
    )
    summary = {
        "backend": backend,
        "execution_device": "cpu",
        "count": 4,
        "tolerance": TOLERANCE,
        "host_container_max_abs_delta": initial_delta,
        "host_loaded_max_abs_delta": loaded_delta,
        "container_repeat_max_abs_delta": repeat_delta,
        "image_id": image_id,
        "image_config_digest": image_id,
        "archive_sha256": archive_hash,
        "archive_bytes": archive.stat().st_size,
        "runtime_dependencies": metadata["runtime_dependencies"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "source": metadata["source"],
        "alignment": [
            {
                "path": key[0],
                "sha256": key[1],
                "reference": reference[key],
                "container": first[key],
                "loaded": repeated[key],
            }
            for key in reference
        ],
        "conditions": {
            "network": "none",
            "root_filesystem": "read-only",
            "user": f"{os.getuid()}:{os.getgid()}",
            "models_inputs": "read-only",
            "cache": "empty before each run",
            "host_source_or_venv_mount": False,
        },
        "limitations": [
            "GPU container execution not tested",
            "10k-scale container capacity not tested",
            "Self-generated unlabelled fixtures; no EER conclusion",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--dockerfile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cu126"), required=True)
    args = parser.parse_args()
    rehearse(args.package, args.dockerfile, args.output, args.backend)


if __name__ == "__main__":
    main()
