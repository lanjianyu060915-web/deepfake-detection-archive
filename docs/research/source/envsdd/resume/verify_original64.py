"""Independent full WAV audit plus recovery of a transient truncated response."""

from __future__ import annotations

import hashlib
import io
import json
import wave
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[5]
DOC = Path(__file__).parent
DATA = ROOT / "data/envsdd-urban"


def decode(raw: bytes) -> dict:
    with wave.open(io.BytesIO(raw), "rb") as reader:
        if reader.getparams()[:4] != (1, 2, 16000, 64000):
            raise ValueError("incorrect WAV shape")
        pcm = reader.readframes(reader.getnframes())
    if len(raw) != 128044 or len(pcm) != 128000:
        raise ValueError("truncated WAV payload")
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "codec": "pcm_s16le",
        "sample_rate": 16000,
        "channels": 1,
        "frames": 64000,
        "duration_seconds": 4.0,
        "rms": float(np.sqrt(np.mean(samples**2))),
        "peak": float(np.max(np.abs(samples))),
        "all_zero": bool(not samples.any()),
        "clipping_fraction": float(np.mean(np.abs(samples) >= 32767 / 32768)),
    }


def main() -> None:
    previous = json.loads((DOC / "original64-execution.json").read_text())
    recovered = ROOT / "cache/envsdd-row15003-diagnostic.wav"
    expected_sha = "6dd736d34a907c58efc36d51bc9d8d93d6dc56f6fb44b9b71864f62d7b3cf23a"
    raw = recovered.read_bytes()
    if decode(raw)["sha256"] != expected_sha:
        raise ValueError("recovery bytes differ from actual second fetch")
    target = DATA / "audio/fake/187075/15003.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(raw)
    rows = []
    for old in previous["items"]:
        info = decode((DATA / old["relative_path"]).read_bytes())
        if old["status"] == "verified" and info["sha256"] != old["sha256"]:
            raise ValueError("previously verified WAV changed")
        row = {k: v for k, v in old.items() if k not in {"error", "status"}}
        row.update(info, status="verified")
        rows.append(row)
    report = {
        "completed_at": datetime.now(UTC).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": previous["manifest_sha256"],
        "fixed_viewer_revision": previous["fixed_viewer_revision"],
        "purpose": previous["purpose"],
        "competition_approved": False,
        "license": previous["license"],
        "planned": 64,
        "verified": len(rows),
        "failed": 0,
        "labels": dict(Counter(r["label"] for r in rows)),
        "fsid_count": len({r["fsid"] for r in rows}),
        "complete_selected_pairs": 32,
        "full_fsid_families": False,
        "verified_bytes": sum(r["bytes"] for r in rows),
        "unique_media_hashes": len({r["sha256"] for r in rows}),
        "zero_signals": sum(r["all_zero"] for r in rows),
        "transient_failure": {
            "row_idx": 15003,
            "first_attempt": "WAV decoded payload had an odd byte count; failed download was not published and bytes were not retained",
            "recovery": "same fixed-revision signed URL fetched again; exact 128044-byte WAV with 128000 PCM bytes",
            "recovery_sha256": expected_sha,
        },
        "no_model_run": True,
        "items": rows,
    }
    (DOC / "original64-verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "items"}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
