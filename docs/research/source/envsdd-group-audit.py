"""Audit recording overlap in the fixed EnvSDD source metadata CSV, not media."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


EXPECTED_SHA256 = "cda95f3cafaed74193a6c988904df8de736a48e42ef7b508df47bd0776d8fe0a"
CSV_URL = (
    "https://raw.githubusercontent.com/apple-yinhan/EnvSDD/"
    "d6179d1c773ca78e4939be5933c9914f97815920/datasplit_dev.csv"
)


def audit(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != EXPECTED_SHA256:
        raise ValueError("CSV bytes differ from the fixed source version")
    with path.open(newline="", encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [row for row in all_rows if row["dataset"] == "UrbanSound8K"]
    groups: dict[str, set[str]] = {"train": set(), "validation": set()}
    for row in rows:
        match = re.fullmatch(r"(\d+)-\d+-\d+-\d+\.wav", row["wavename"])
        if match is None or row["usage"] not in groups:
            raise ValueError("unexpected UrbanSound8K filename or usage")
        groups[row["usage"]].add(match[1])
    shared = groups["train"] & groups["validation"]
    return {
        "scope": "fixed source CSV usage; current Hub split alignment not verified",
        "source_url": CSV_URL,
        "source_sha256": EXPECTED_SHA256,
        "source_bytes": len(content),
        "all_source_rows": len(all_rows),
        "urbansound8k_rows": len(rows),
        "usage_row_counts": dict(Counter(row["usage"] for row in rows)),
        "recording_counts": {key: len(value) for key, value in groups.items()},
        "shared_recordings": len(shared),
        "rows_in_shared_recordings": dict(
            Counter(
                row["usage"]
                for row in rows
                if row["wavename"].split("-", 1)[0] in shared
            )
        ),
        "validation_recording_overlap_fraction": len(shared)
        / len(groups["validation"]),
        "naming_contract": "https://urbansounddataset.weebly.com/urbansound8k.html",
        "derivation_unit": "(dataset, original slice filename)",
        "split_unit": "(dataset, fsID of the original Freesound recording)",
        "media_decoded": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().csv), indent=2, sort_keys=True))
