"""Independent offline audit of frozen CSV families and every complete WAV body."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import wave
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[5]
DOC = Path(__file__).parent
DATA = ROOT / "data/envsdd-urban-full-family"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    manifest_path = DOC / "selection.json"
    lock = json.loads((DOC / "selection-lock.json").read_text())
    if digest(manifest_path) != lock["sha256"]:
        raise ValueError("selection lock mismatch")
    manifest = json.loads(manifest_path.read_text())
    csv_path = ROOT / manifest["csv"]["path"]
    if digest(csv_path) != manifest["csv"]["sha256"]:
        raise ValueError("CSV hash mismatch")
    families: dict[int, set[str]] = defaultdict(set)
    with csv_path.open() as stream:
        for row in csv.DictReader(stream):
            if row["dataset"] == "UrbanSound8K":
                families[int(row["wavename"].split("-")[0])].add(row["wavename"])
    viewer: dict[int, dict[str, Any]] = {}
    for page in manifest["source_pages"]:
        path = ROOT / page["path"]
        if digest(path) != page["sha256"]:
            raise ValueError("original page hash mismatch")
        for item in json.loads(path.read_text())["rows"]:
            if item["row"]["dataset"] != "UrbanSound8K":
                raise ValueError("non-Urban source row")
            viewer[item["row_idx"]] = item["row"]
    expected_ids = []
    qualifying = []
    for fsid in sorted(families):
        rows = {
            i: r for i, r in viewer.items() if int(r["filename"].split("-")[0]) == fsid
        }
        if {r["filename"] for r in rows.values()} != families[fsid]:
            continue
        if all(
            {r["label"] for r in rows.values() if r["filename"] == name}
            == {"real", "fake"}
            and len(
                {
                    r["generative_model"]
                    for r in rows.values()
                    if r["filename"] == name and r["label"] == "fake"
                }
            )
            >= 3
            for name in families[fsid]
        ):
            qualifying.append(fsid)
            expected_ids.extend(sorted(rows, key=lambda i: (rows[i]["filename"], i)))
    if (
        len(qualifying),
        sum(len(families[f]) for f in qualifying),
        len(expected_ids),
    ) != (40, 106, 530):
        raise ValueError("independent selection counts differ")
    if expected_ids != [r["row_idx"] for r in manifest["samples"]]:
        raise ValueError("selection rows/order differ")
    runs = sorted(DOC.glob("run-*/execution.json"))
    if not runs:
        raise ValueError("no completed execution report")
    report = json.loads(runs[-1].read_text())
    if report["manifest_sha256"] != lock["sha256"]:
        raise ValueError("execution bound to another manifest")
    fresh_rows = {}
    page_count = 0
    for line in (runs[-1].parent / "journal.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if entry["kind"] != "page" or entry["status"] != "verified":
            continue
        if entry["headers"]["x-revision"] != manifest["revision"]:
            raise ValueError("refresh revision header differs")
        page_path = runs[-1].parent / f"train-offset-{entry['offset']}.json"
        if digest(page_path) != entry["sha256"]:
            raise ValueError("refreshed page hash differs")
        page = json.loads(page_path.read_text())
        if page["partial"] or len(page["rows"]) != 100:
            raise ValueError("partial refreshed page")
        for item in page["rows"]:
            if item["row"]["dataset"] != "UrbanSound8K" or item["truncated_cells"]:
                raise ValueError("non-Urban or truncated refreshed row")
            fresh_rows[item["row_idx"]] = item["row"]
        page_count += 1
    if page_count != 25:
        raise ValueError("missing refreshed pages")
    for item in manifest["samples"]:
        metadata = {
            k: v for k, v in fresh_rows[item["row_idx"]].items() if k != "audio"
        }
        if metadata != item["metadata"]:
            raise ValueError("refreshed metadata differs from selection")
    executed = {r["row_idx"]: r for r in report["items"]}
    if set(executed) != set(expected_ids) or len(report["items"]) != 530:
        raise ValueError("execution missing or duplicated rows")
    decoded = []
    for item in manifest["samples"]:
        row = executed[item["row_idx"]]
        if row["status"] != "verified" or any(row[k] != v for k, v in item.items()):
            raise ValueError("failed row or metadata drift")
        if item["metadata"] != {
            k: v for k, v in viewer[item["row_idx"]].items() if k != "audio"
        }:
            raise ValueError("selection metadata differs from original Viewer")
        path = DATA / item["relative_path"]
        raw = path.read_bytes()
        if digest(path) != row["sha256"] or len(raw) != row["bytes"]:
            raise ValueError("media body hash or byte count differs")
        with wave.open(io.BytesIO(raw), "rb") as reader:
            if (
                reader.getparams()[:4] != (1, 2, 16000, 64000)
                or reader.getcomptype() != "NONE"
            ):
                raise ValueError("WAV format differs")
            pcm = reader.readframes(64001)
        if len(pcm) != 128000:
            raise ValueError("truncated or excessive PCM frames")
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768
        metrics = {
            "rms": float(np.sqrt(np.mean(samples**2))),
            "peak": float(np.max(np.abs(samples))),
            "all_zero": bool(not samples.any()),
            "zero_fraction": float(np.mean(samples == 0)),
            "clipping_fraction": float(np.mean(np.abs(samples) >= 32767 / 32768)),
        }
        if any(row[k] != v for k, v in metrics.items()):
            raise ValueError("recorded signal diagnostics differ")
        decoded.append(row)
    actual = {str(p.relative_to(DATA)) for p in DATA.rglob("*.wav")}
    if actual != {r["relative_path"] for r in manifest["samples"]}:
        raise ValueError("missing or extra local WAV files")
    summary = {
        "at": datetime.now(UTC).isoformat(),
        "verifier_sha256": digest(Path(__file__)),
        "manifest_sha256": lock["sha256"],
        "execution_path": str(runs[-1].relative_to(ROOT)),
        "execution_sha256": digest(runs[-1]),
        "verified": 530,
        "failed": 0,
        "fsids": 40,
        "filenames": 106,
        "bytes": sum(r["bytes"] for r in decoded),
        "labels": dict(Counter(r["metadata"]["label"] for r in decoded)),
        "generators": dict(Counter(r["metadata"]["generative_model"] for r in decoded)),
        "unique_hashes": len({r["sha256"] for r in decoded}),
        "zero_signals": sum(r["all_zero"] for r in decoded),
        "signals_with_clipping": sum(r["clipping_fraction"] > 0 for r in decoded),
        "failed_media_attempts": sum(
            a["status"] == "failed" for r in decoded for a in r["attempts"]
        ),
        "completeness_scope": manifest["completeness_scope"],
        "competition_approved": False,
        "no_model_run": True,
    }
    output = DOC / (
        "verification-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    )
    with output.open("x") as stream:
        stream.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
