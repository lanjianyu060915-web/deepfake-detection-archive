"""Build a deterministic, balanced UrbanSound8K EnvSDD exploration manifest.

The input Viewer responses are fixed small metadata pages saved beside this file.
This script does not call the network or decode/download media.  It derives the
parent recording identifier only through UrbanSound8K's documented filename
contract, then selects one real and one TTA/Audioldm2 fake row per parent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parent
PAGES = ROOT / "viewer-pages"
FILENAME = re.compile(r"^(?P<fsid>\d+)-\d+-\d+-\d+\.wav$")
EXPECTED = {
    "train-offset-0.json": "218c72e1161a80156322416e34f030602ecc8c08d6253af948d9a4e0c1b68268",
    "train-offset-100.json": "e6fb4cd15dd0f46a8b89450052e76521832d3425cdc709e4e72090967c905710",
    "train-offset-1000.json": "691b54735019c0deb5bb33e7247d984b86fa158d80076d58a5e234ae4a7b715e",
    "train-offset-5000.json": "01766b3e0e52434a46edc97767fb09885649e2f2b35613eb4641957a96fce74e",
    "train-offset-10000.json": "06125eccca59a06e054dfbd818f962b321145e4fe7758e3bb45642024ff2a058",
    "train-offset-15000.json": "3939b25d809574d023141dbb388579e83178052e7ce497cc30ea9193693328f2",
}
TARGETS = (("real", "real", "real"), ("fake", "tta", "audioldm2"))


def checked_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, expected_sha in EXPECTED.items():
        path = PAGES / name
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"source response hash differs: {path}")
        rows.extend(item for item in json.loads(path.read_text())["rows"])
    return rows


def manifest() -> dict[str, Any]:
    by_fsid: dict[int, dict[str, dict[int, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for item in checked_rows():
        row = item["row"]
        if row["dataset"] != "UrbanSound8K":
            continue
        match = FILENAME.fullmatch(row["filename"])
        if match is None:
            raise ValueError(f"unexpected UrbanSound8K filename: {row['filename']}")
        by_fsid[int(match["fsid"])][row["filename"]][item["row_idx"]] = item

    samples: list[dict[str, Any]] = []
    for fsid in sorted(by_fsid)[:32]:
        filename = min(by_fsid[fsid])
        rows = list(by_fsid[fsid][filename].values())
        selected: dict[tuple[str, str, str], dict[str, Any]] = {}
        for item in rows:
            row = item["row"]
            key = (row["label"], row["attack_type"], row["generative_model"])
            selected[key] = item
        for label, attack_type, generator in TARGETS:
            item = selected.get((label, attack_type, generator))
            if item is None:
                raise ValueError(f"missing target for fsID={fsid}, filename={filename}")
            row = item["row"]
            samples.append(
                {
                    "split": "train",
                    "row_idx": item["row_idx"],
                    "fsid": fsid,
                    "filename": filename,
                    "label": label,
                    "attack_type": attack_type,
                    "generative_model": generator,
                    "scene": row["scene"],
                    "event": row["event"],
                    "caption": row["caption"],
                    "audio_url": row["audio"][0]["src"],
                }
            )
    if len(samples) != 64 or len({sample["fsid"] for sample in samples}) != 32:
        raise ValueError("unexpected selected cardinality")
    return {
        "schema_version": 1,
        "purpose": "local_noncommercial_research_exploration_only",
        "dataset": "EnvSDD/EnvSDD",
        "viewer_dataset_revision": "b0832726deb55484dbce869c6f09c41f7e9e5fbb",
        "selection_rule": (
            "From fixed public Viewer pages, select the lexicographically first "
            "sampled filename for each of the 32 lowest numeric UrbanSound8K fsIDs; "
            "select its real row and TTA/Audioldm2 fake row."
        ),
        "group_unit": "UrbanSound8K fsID; all selected samples are assigned to exploration only",
        "license": "UrbanSound8K CC-BY-NC-3.0; retain Freesound attribution",
        "samples": samples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest(), ensure_ascii=False, indent=2) + "\n")
