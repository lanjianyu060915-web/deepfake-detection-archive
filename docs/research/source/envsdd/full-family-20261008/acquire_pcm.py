"""Freeze and acquire only complete UrbanSound families in the fixed author CSV."""

# ruff: noqa: C901, PLR0912, PLR0915, TRY301
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
import wave
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[5]
DOC = Path(__file__).parent
SOURCE = DOC.parent
DATA = ROOT / "data/envsdd-urban-full-family"
REVISION = "b0832726deb55484dbce869c6f09c41f7e9e5fbb"
CSV_SHA = "cda95f3cafaed74193a6c988904df8de736a48e42ef7b508df47bd0776d8fe0a"
HOST = "datasets-server.huggingface.co"
LIMIT = 256088


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_new(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def select() -> dict[str, Any]:
    csv_path = SOURCE / "datasplit_dev-d6179d1c.csv"
    if sha(csv_path.read_bytes()) != CSV_SHA:
        raise ValueError("fixed author CSV changed")
    groups: dict[int, set[str]] = defaultdict(set)
    csv_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open() as stream:
        for row in csv.DictReader(stream):
            if row["dataset"] == "UrbanSound8K":
                groups[int(row["wavename"].split("-")[0])].add(row["wavename"])
                csv_rows[row["wavename"]].append(row)
    viewer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    evidence = []
    indices = set()
    pages = sorted((SOURCE / "resume/viewer-pages").glob("*.json"))
    if len(pages) != 33:
        raise ValueError("expected exactly 33 source pages")
    for path in pages:
        raw = path.read_bytes()
        page = json.loads(raw)
        if page["partial"] or len(page["rows"]) != 100:
            raise ValueError("incomplete source page")
        evidence.append({"path": str(path.relative_to(ROOT)), "sha256": sha(raw)})
        for item in page["rows"]:
            row = item["row"]
            if row["dataset"] != "UrbanSound8K" or item["truncated_cells"]:
                raise ValueError("non-Urban or truncated metadata is forbidden")
            if item["row_idx"] in indices:
                raise ValueError("duplicate viewer row index")
            indices.add(item["row_idx"])
            viewer[int(row["filename"].split("-")[0])].append(item)
    samples = []
    families: list[dict[str, Any]] = []
    for fsid, rows in sorted(viewer.items()):
        filenames = {r["row"]["filename"] for r in rows}
        if filenames != groups[fsid]:
            continue
        if not all(
            any(
                r["row"]["label"] == "real"
                for r in rows
                if r["row"]["filename"] == filename
            )
            and len(
                {
                    r["row"]["generative_model"]
                    for r in rows
                    if r["row"]["filename"] == filename and r["row"]["label"] == "fake"
                }
            )
            >= 3
            for filename in filenames
        ):
            continue
        families.append(
            {
                "fsid": fsid,
                "filenames": sorted(filenames),
                "freesound_url": f"https://freesound.org/s/{fsid}/",
            }
        )
        for item in sorted(rows, key=lambda r: (r["row"]["filename"], r["row_idx"])):
            row = item["row"]
            metadata = {k: v for k, v in row.items() if k != "audio"}
            samples.append(
                {
                    "fsid": fsid,
                    "row_idx": item["row_idx"],
                    "metadata": metadata,
                    "author_csv_rows": csv_rows[row["filename"]],
                    "freesound_url": f"https://freesound.org/s/{fsid}/",
                    "relative_path": f"audio/{fsid}/{item['row_idx']}.wav",
                }
            )
    counts = (len(families), sum(len(f["filenames"]) for f in families), len(samples))
    if counts != (40, 106, 530):
        raise ValueError(f"selection differs from approved counts: {counts}")
    return {
        "revision": REVISION,
        "dataset": "EnvSDD/EnvSDD",
        "config": "default",
        "viewer_split": "train",
        "group_unit": "UrbanSound8K fsID",
        "selection_order": "numeric fsID, lexical filename, numeric row_idx",
        "selection_rule": "viewer filenames equal fixed CSV family; real and >=3 fake generators per filename",
        "completeness_scope": "fixed author CSV filenames only, not full original UrbanSound recordings or all EnvSDD derivatives",
        "purpose": "local_noncommercial_research_only",
        "competition_approved": False,
        "license": "UrbanSound8K CC-BY-NC-3.0; EnvSDD CC-BY-4.0 does not supersede upstream terms",
        "license_sources": [
            "https://urbansounddataset.weebly.com/",
            "https://urbansounddataset.weebly.com/urbansound8k.html",
            f"https://huggingface.co/datasets/EnvSDD/EnvSDD/blob/{REVISION}/README.md",
        ],
        "attribution": {
            "dataset_creators": [
                "Justin Salamon",
                "Christopher Jacoby",
                "Juan Pablo Bello",
            ],
            "recording_links": "Each sample retains its Freesound fsID URL",
            "credits_file_status": "FREESOUNDCREDITS.txt not acquired; per-recording creator attribution requires further verification",
        },
        "csv": {
            "path": str(csv_path.relative_to(ROOT)),
            "sha256": CSV_SHA,
            "url": "https://raw.githubusercontent.com/apple-yinhan/EnvSDD/d6179d1c773ca78e4939be5933c9914f97815920/datasplit_dev.csv",
        },
        "source_pages": evidence,
        "source_rows": len(indices),
        "families": families,
        "samples": samples,
        "counts": {"fsids": 40, "filenames": 106, "rows": 530},
        "media_request_budget_bytes": len(samples) * 3 * (LIMIT + 1),
        "max_workers": 3,
        "timeout_seconds": 45,
        "attempts_per_row": 3,
    }


def get(url: str, limit: int) -> tuple[bytes, dict[str, str]]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != HOST:
        raise ValueError("URL outside fixed public Viewer host")
    with urllib.request.urlopen(url, timeout=45) as response:  # noqa: S310
        if urllib.parse.urlparse(response.url).netloc != HOST:
            raise ValueError("redirect outside Viewer host")
        body = response.read(limit + 1)
        if len(body) > limit:
            raise ValueError("response exceeds byte budget")
        headers = {
            k.lower(): v
            for k, v in response.headers.items()
            if k.lower() in {"date", "etag", "x-revision", "content-length"}
        }
        if "content-length" in headers and len(body) != int(headers["content-length"]):
            raise ValueError("truncated HTTP body")
        return body, headers


def decode(body: bytes) -> dict[str, Any]:
    with wave.open(io.BytesIO(body), "rb") as audio:
        params = audio.getparams()
        if (
            params.nchannels != 1
            or params.sampwidth not in {1, 2, 3, 4}
            or params.framerate != 16000
            or params.nframes != 64000
            or params.comptype != "NONE"
        ):
            raise ValueError(f"unexpected WAV format: {params}")
        frames = audio.readframes(64000)
        if len(frames) != 64000 * params.sampwidth or audio.readframes(1):
            raise ValueError("incomplete or excess PCM payload")
    width = params.sampwidth
    scale = 1 << (width * 8 - 1)
    if width == 1:
        values = (
            np.frombuffer(frames, dtype=np.uint8).astype(np.float64) - 128
        ) / scale
    elif width == 3:
        octets = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3).astype(np.int64)
        integers = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        signed = (integers ^ (1 << 23)) - (1 << 23)
        values = signed.astype(np.float64) / scale
    else:
        values = np.frombuffer(frames, dtype=f"<i{width}").astype(np.float64) / scale
    return {
        "sha256": sha(body),
        "bytes": len(body),
        "sample_rate": 16000,
        "channels": 1,
        "sample_width_bytes": width,
        "frames": 64000,
        "codec": "pcm_u8" if width == 1 else f"pcm_s{width * 8}le",
        "duration_seconds": 4,
        "rms": float(np.sqrt(np.mean(values**2))),
        "peak": float(np.max(np.abs(values))),
        "all_zero": bool(not values.any()),
        "zero_fraction": float(np.mean(values == 0)),
        "clipping_fraction": float(np.mean(np.abs(values) >= (scale - 1) / scale)),
    }


def acquire(manifest: dict[str, Any], manifest_sha: str) -> None:
    run = DOC / ("run-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    run.mkdir()
    write_new(
        run / "start.json",
        {
            "at": now(),
            "manifest_sha256": manifest_sha,
            "script_sha256": sha(Path(__file__).read_bytes()),
        },
    )
    journal = run / "journal.jsonl"

    def log(record: dict[str, Any]) -> None:
        with journal.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    refreshed = {}
    for offset in sorted({r["row_idx"] // 100 * 100 for r in manifest["samples"]}):
        url = f"https://{HOST}/rows?" + urllib.parse.urlencode(
            {
                "dataset": "EnvSDD/EnvSDD",
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": 100,
                "revision": REVISION,
            }
        )
        for attempt in range(1, 4):
            try:
                body, headers = get(url, 2_000_000)
                if headers.get("x-revision") != REVISION:
                    raise ValueError("x-revision mismatch or missing")
                page = json.loads(body)
                if page["partial"] or len(page["rows"]) != 100:
                    raise ValueError("partial refreshed page")
                for item in page["rows"]:
                    if (
                        item["row"]["dataset"] != "UrbanSound8K"
                        or item["truncated_cells"]
                    ):
                        raise ValueError("non-Urban or truncated refreshed row")
                    refreshed[item["row_idx"]] = item["row"]
                (run / f"train-offset-{offset}.json").write_bytes(body)
                log(
                    {
                        "kind": "page",
                        "offset": offset,
                        "attempt": attempt,
                        "status": "verified",
                        "url": url,
                        "sha256": sha(body),
                        "headers": headers,
                        "at": now(),
                    }
                )
                print(f"refreshed offset={offset}", flush=True)
                break
            except Exception as exc:
                log(
                    {
                        "kind": "page",
                        "offset": offset,
                        "attempt": attempt,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "at": now(),
                    }
                )
                if attempt == 3:
                    raise
                time.sleep(attempt)
    for item in manifest["samples"]:
        row = refreshed[item["row_idx"]]
        if {k: v for k, v in row.items() if k != "audio"} != item["metadata"]:
            raise ValueError("refreshed metadata differs from frozen selection")
        asset = urllib.parse.urlparse(row["audio"][0]["src"])
        expected = f"/cached-assets/EnvSDD/EnvSDD/--/{REVISION}/--/default/train/{item['row_idx']}/audio/audio.wav"
        if asset.scheme != "https" or asset.netloc != HOST or asset.path != expected:
            raise ValueError("audio asset identity differs")
    previous = {}
    for path in sorted(DOC.glob("run-*/journal.jsonl")):
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            if entry["kind"] == "media" and entry["status"] == "verified":
                previous[entry["row_idx"]] = entry

    def one(item: dict[str, Any]) -> dict[str, Any]:
        target = DATA / item["relative_path"]
        result = dict(item, kind="media", attempts=[])
        if target.exists():
            try:
                info = decode(target.read_bytes())
                if (
                    item["row_idx"] not in previous
                    or info["sha256"] != previous[item["row_idx"]]["sha256"]
                ):
                    raise ValueError(
                        "existing media has no matching verified journal hash"
                    )
                return dict(result, **info, status="verified", reused=True, at=now())
            except Exception as exc:
                return dict(result, status="failed", error=str(exc), at=now())
        for attempt in range(1, 4):
            try:
                url = refreshed[item["row_idx"]]["audio"][0]["src"]
                body, headers = get(url, LIMIT)
                info = decode(body)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".wav.partial")
                with temporary.open("wb") as stream:
                    stream.write(body)
                temporary.replace(target)
                result["attempts"].append(
                    {
                        "attempt": attempt,
                        "status": "verified",
                        "at": now(),
                        "headers": headers,
                    }
                )
                return dict(result, **info, status="verified", reused=False, at=now())
            except Exception as exc:
                result["attempts"].append(
                    {
                        "attempt": attempt,
                        "status": "failed",
                        "at": now(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if attempt < 3:
                    time.sleep(attempt)
        return dict(result, status="failed", at=now())

    records = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(one, item) for item in manifest["samples"]]
        for future in as_completed(futures):
            record = future.result()
            log(record)
            records.append(record)
            if len(records) % 10 == 0 or record["status"] == "failed":
                print(
                    f"media {len(records)}/530 failures={sum(r['status'] == 'failed' for r in records)}",
                    flush=True,
                )
    records.sort(key=lambda r: (r["fsid"], r["metadata"]["filename"], r["row_idx"]))
    verified = [r for r in records if r["status"] == "verified"]
    report = {
        "at": now(),
        "manifest_sha256": manifest_sha,
        "planned": 530,
        "verified": len(verified),
        "failed": 530 - len(verified),
        "bytes": sum(r["bytes"] for r in verified),
        "labels": dict(Counter(r["metadata"]["label"] for r in verified)),
        "generators": dict(
            Counter(r["metadata"]["generative_model"] for r in verified)
        ),
        "items": records,
    }
    write_new(run / "execution.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "items"}), flush=True)
    if len(verified) != 530:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "fetch"])
    args = parser.parse_args()
    calculated = select()
    path = DOC / "selection.json"
    if args.mode == "freeze":
        write_new(path, calculated)
        write_new(
            DOC / "selection-lock.json",
            {"sha256": sha(path.read_bytes()), "created_at": now()},
        )
        print(json.dumps(calculated["counts"]))
    else:
        raw = path.read_bytes()
        if sha(raw) != json.loads((DOC / "selection-lock.json").read_text())["sha256"]:
            raise ValueError("frozen selection hash changed")
        if json.loads(raw) != calculated:
            raise ValueError("source evidence no longer reproduces frozen selection")
        acquire(calculated, sha(raw))


if __name__ == "__main__":
    main()
