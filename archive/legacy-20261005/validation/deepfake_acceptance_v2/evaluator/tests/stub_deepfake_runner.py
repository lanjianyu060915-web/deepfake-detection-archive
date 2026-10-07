"""Deterministic adapter-shaped runner used only by evaluator tests.

The order enumerates eight distinct required behaviours even though it refers
to "seven" modes in prose. All eight are implemented, plus two narrow schema
failure modes that exercise missing-output and missing-label handling.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


MODES = (
    "all_correct",
    "partial_wrong",
    "all_real",
    "all_fake",
    "crash",
    "invalid_json",
    "status_error",
    "timeout",
    "no_output",
    "missing_label",
)


def _load_fixture(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = payload.get("samples", payload)
    if not isinstance(samples, dict):
        raise ValueError("fixture map must contain an object named samples")
    return samples


def _sample_record(query: Path, fixture: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sample_id = query.stem
    if sample_id in fixture:
        return fixture[sample_id]
    if sample_id.startswith("R"):
        return {"truth": "REAL", "partial_label": "real", "score": 0.1}
    if sample_id.startswith("F"):
        return {"truth": "FAKE", "partial_label": "deepfake", "score": 0.9}
    raise ValueError(f"stub cannot infer fixture truth for {sample_id!r}")


def _success_payload(query: Path, label: str, score: float, mode: str) -> dict[str, Any]:
    return {
        "status": "success",
        "query_image": str(query.resolve(strict=False)),
        "is_deepfake": label == "deepfake",
        "score": score,
        "label": label,
        "runtime_seconds": 0.001,
        "details": {
            "stub": True,
            "stub_mode": mode,
            "score_semantics": "test fixture: larger values mean FAKE",
        },
        "error_message": None,
    }


def run(args: argparse.Namespace) -> int:
    query = Path(args.query).expanduser().resolve(strict=False)
    output = Path(args.output).expanduser().resolve(strict=False)
    fixture = _load_fixture(args.fixture_map)

    if args.stub_mode == "crash":
        return 23
    if args.stub_mode == "timeout":
        time.sleep(args.sleep_seconds)
    if args.stub_mode == "no_output":
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.stub_mode == "invalid_json":
        output.write_text('{"status": "success", this is not JSON', encoding="utf-8")
        return 0
    if args.stub_mode == "status_error":
        output.write_text(
            json.dumps(
                {
                    "status": "error",
                    "query_image": str(query),
                    "is_deepfake": None,
                    "score": None,
                    "label": None,
                    "runtime_seconds": 0.001,
                    "details": {"stub": True, "stub_mode": args.stub_mode},
                    "error_message": "intentional stub status error",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    record = _sample_record(query, fixture)
    truth = record.get("truth")
    if truth not in {"REAL", "FAKE"}:
        raise ValueError(f"fixture truth must be REAL or FAKE for {query.stem}")
    if args.stub_mode == "all_correct":
        label = "deepfake" if truth == "FAKE" else "real"
        score = 0.9 if truth == "FAKE" else 0.1
    elif args.stub_mode == "partial_wrong":
        label = str(record.get("partial_label"))
        score = float(record.get("score"))
    elif args.stub_mode == "all_real":
        label, score = "real", 0.1
    elif args.stub_mode == "all_fake":
        label, score = "deepfake", 0.9
    elif args.stub_mode == "missing_label":
        payload = _success_payload(query, "real", 0.1, args.stub_mode)
        payload["label"] = None
        payload["is_deepfake"] = None
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0
    else:
        raise ValueError(f"unsupported stub mode: {args.stub_mode}")

    output.write_text(
        json.dumps(_success_payload(query, label, score, args.stub_mode), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D-v2 evaluator-only deterministic stub")
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stub-mode", choices=MODES, default="all_correct")
    parser.add_argument("--fixture-map")
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))

