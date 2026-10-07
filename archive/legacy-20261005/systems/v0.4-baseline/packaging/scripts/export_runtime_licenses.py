"""Export a deterministic runtime package inventory and bundled license texts."""

from __future__ import annotations

import argparse
import csv
from importlib import metadata
from pathlib import Path
import re
import shutil
import sys


BUILD_ONLY = {
    "altgraph",
    "pefile",
    "pip",
    "pyinstaller",
    "pyinstaller-hooks-contrib",
    "pywin32-ctypes",
    "setuptools",
    "wheel",
}
LICENSE_PREFIXES = ("license", "licence", "copying", "notice", "copyright")


def canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "package"


def metadata_value(dist: metadata.Distribution, key: str) -> str:
    value = dist.metadata.get(key, "")
    return " ".join(str(value).split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--component", required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    component = safe_name(args.component)
    component_dir = output_dir / component
    package_license_root = component_dir / "package_licenses"
    component_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    copied: set[tuple[str, str]] = set()
    distributions = sorted(
        metadata.distributions(),
        key=lambda item: canonical_name(metadata_value(item, "Name")),
    )
    for dist in distributions:
        name = metadata_value(dist, "Name") or "unknown"
        if canonical_name(name) in BUILD_ONLY:
            continue
        classifiers = [
            value.removeprefix("License :: ")
            for value in dist.metadata.get_all("Classifier", [])
            if value.startswith("License :: ")
        ]
        rows.append(
            {
                "component": component,
                "name": name,
                "version": dist.version,
                "license_expression": metadata_value(dist, "License-Expression"),
                "license": metadata_value(dist, "License"),
                "license_classifiers": "; ".join(classifiers),
                "project_url": metadata_value(dist, "Home-page"),
            }
        )

        for listed_file in dist.files or ():
            basename = Path(str(listed_file)).name.lower()
            if not basename.startswith(LICENSE_PREFIXES):
                continue
            source = Path(dist.locate_file(listed_file))
            if not source.is_file():
                continue
            try:
                size = source.stat().st_size
            except OSError:
                continue
            if size > 2 * 1024 * 1024:
                continue
            package_name = safe_name(f"{name}-{dist.version}")
            destination_name = safe_name(Path(str(listed_file)).name)
            key = (package_name.casefold(), destination_name.casefold())
            if key in copied:
                continue
            copied.add(key)
            destination = package_license_root / package_name / destination_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    inventory_path = component_dir / "runtime_packages.csv"
    with inventory_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "component",
                "name",
                "version",
                "license_expression",
                "license",
                "license_classifiers",
                "project_url",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    runtime_path = component_dir / "python_runtime.txt"
    runtime_path.write_text(
        "\n".join(
            [
                f"component={component}",
                f"python={sys.version.replace(chr(10), ' ')}",
                f"executable={sys.executable}",
                f"base_prefix={sys.base_prefix}",
                "python_license=https://docs.python.org/3/license.html",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    for candidate in (
        Path(sys.base_prefix) / "LICENSE.txt",
        Path(sys.base_prefix) / "LICENSE",
        Path(sys.prefix) / "LICENSE.txt",
        Path(sys.prefix) / "LICENSE",
    ):
        if candidate.is_file():
            shutil.copy2(candidate, component_dir / "PYTHON_LICENSE.txt")
            break
    print(f"{component}: {len(rows)} runtime distributions, {len(copied)} license files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

