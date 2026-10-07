"""Resume the frozen 64-row UrbanSound subset; preserve prior evidence."""

# Research acquisition script keeps a single auditable execution narrative.
# ruff: noqa: C901, PLR0915, TRY301
from __future__ import annotations

import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[5]
DOC = Path(__file__).parent
SOURCE = DOC.parent
DATA = ROOT / "data/envsdd-urban"
REVISION = "b0832726deb55484dbce869c6f09c41f7e9e5fbb"
MANIFEST_SHA = "ade4742cc94a590bf55195ff512e91f6dba00eb7a88c7c17ffe10566757b458b"
KEYS = (
    "filename",
    "label",
    "attack_type",
    "generative_model",
    "scene",
    "event",
    "caption",
)


def sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def get(url: str, limit: int) -> tuple[bytes, dict[str, str]]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "datasets-server.huggingface.co":
        raise ValueError("URL outside fixed read-only public Viewer host")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:  # noqa: S310
                body = response.read(limit + 1)
                if len(body) > limit:
                    raise ValueError("response exceeds byte budget")
                headers = {
                    k: v
                    for k, v in response.headers.items()
                    if k.lower()
                    in {
                        "date",
                        "etag",
                        "x-revision",
                        "x-dataset-revision",
                        "content-length",
                    }
                }
                return body, headers
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def main() -> None:
    started = datetime.now(UTC).isoformat()
    raw = (SOURCE / "manifest-v1-32fsid.json").read_bytes()
    if sha(raw) != MANIFEST_SHA:
        raise ValueError("frozen selection changed")
    manifest = json.loads(raw)
    selected = manifest["samples"]
    paused = json.loads((SOURCE / "acquisition-status-paused.json").read_text())
    old = {r["row_idx"]: r for r in paused["items"] if r["present"]}
    for item in old.values():
        if sha((DATA / item["relative_path"]).read_bytes()) != item["sha256"]:
            raise ValueError("previously verified media changed")
    pages = DOC / "original64-refresh-pages"
    pages.mkdir(exist_ok=True)
    refreshed = {}
    page_evidence = []
    for offset in sorted({r["row_idx"] // 100 * 100 for r in selected}):
        url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
            {
                "dataset": "EnvSDD/EnvSDD",
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": 100,
                "revision": REVISION,
            }
        )
        path = pages / f"train-offset-{offset}.json"
        if path.exists():
            raise FileExistsError(f"fresh evidence directory already used: {path}")
        body, headers = get(url, 2_000_000)
        path.write_bytes(body)
        parsed = json.loads(body)
        if parsed["partial"]:
            raise ValueError("partial viewer response")
        for item in parsed["rows"]:
            refreshed[item["row_idx"]] = item["row"]
        page_evidence.append(
            {
                "path": str(path.relative_to(ROOT)),
                "request_url": url,
                "sha256": sha(body),
                "headers": headers,
            }
        )
    for item in selected:
        row = refreshed[item["row_idx"]]
        if row["dataset"] != "UrbanSound8K" or any(row[k] != item[k] for k in KEYS):
            raise ValueError("refreshed metadata does not match frozen row")
        asset = row["audio"][0]["src"]
        path = urllib.parse.urlparse(asset).path
        expected = f"/cached-assets/EnvSDD/EnvSDD/--/{REVISION}/--/default/train/{item['row_idx']}/audio/audio.wav"
        if (
            path != expected
            or urllib.parse.urlparse(asset).netloc != "datasets-server.huggingface.co"
        ):
            raise ValueError("asset revision or row identity changed")
    (DOC / "original64-refresh-evidence.json").write_text(
        json.dumps(page_evidence, indent=2) + "\n"
    )

    def one(item: dict) -> dict:
        relative = f"audio/{item['label']}/{item['fsid']}/{item['row_idx']}.wav"
        target = DATA / relative
        record = {k: v for k, v in item.items() if k != "audio_url"}
        record.update(relative_path=relative, previously_present=item["row_idx"] in old)
        try:
            if target.exists():
                body = target.read_bytes()
            else:
                body, _ = get(refreshed[item["row_idx"]]["audio"][0]["src"], 256_088)
            with wave.open(io.BytesIO(body), "rb") as audio:
                params = audio.getparams()
                if params[:4] != (1, 2, 16000, 64000) or params.comptype != "NONE":
                    raise ValueError(f"unexpected media format: {params}")
                frames = audio.readframes(params.nframes)
            values = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768
            if len(values) != 64000 or not np.isfinite(values).all():
                raise ValueError("invalid full decoded payload")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".wav.partial")
                with temporary.open("xb") as stream:
                    stream.write(body)
                temporary.replace(target)
            record.update(
                status="verified",
                bytes=len(body),
                sha256=sha(body),
                sample_rate=16000,
                channels=1,
                sample_width_bytes=2,
                frames=64000,
                duration_seconds=4.0,
                codec="pcm_s16le",
                rms=float(np.sqrt(np.mean(values**2))),
                peak=float(np.max(np.abs(values))),
                all_zero=bool(not values.any()),
                clipping_fraction=float(np.mean(np.abs(values) >= 32767 / 32768)),
            )
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        return record

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(one, selected))
    verified = [r for r in records if r["status"] == "verified"]
    report = {
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "manifest_sha256": MANIFEST_SHA,
        "fixed_viewer_revision": REVISION,
        "purpose": "local_noncommercial_research_exploration_only",
        "competition_approved": False,
        "license": "UrbanSound8K CC-BY-NC-3.0; retain Freesound attribution",
        "selected": len(selected),
        "verified": len(verified),
        "failed": len(records) - len(verified),
        "previous_hashes_verified": len(old),
        "newly_downloaded": sum(not r["previously_present"] for r in verified),
        "verified_bytes": sum(r["bytes"] for r in verified),
        "labels": dict(Counter(r["label"] for r in verified)),
        "generators": dict(Counter(r["generative_model"] for r in verified)),
        "fsid_count": len({r["fsid"] for r in verified}),
        "full_fsid_families": False,
        "complete_selected_real_fake_pairs": sum(
            all(r["status"] == "verified" for r in records if r["fsid"] == f)
            for f in {r["fsid"] for r in records}
        ),
        "failure_policy": "Every frozen row is recorded; no silent dropping",
        "items": records,
    }
    (DOC / "original64-execution.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "items"}, ensure_ascii=False
        )
    )
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
