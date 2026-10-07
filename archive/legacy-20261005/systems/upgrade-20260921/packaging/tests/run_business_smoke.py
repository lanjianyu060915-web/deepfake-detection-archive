from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-root", required=True)
    parser.add_argument("--local-app-data", required=True)
    parser.add_argument("--query", default="query_001.jpg")
    parser.add_argument("--expected-identity", default="person_001")
    parser.add_argument("--expected-source", default="122.jpg")
    parser.add_argument("--summary")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    app_root = Path(args.app_root).resolve()
    local_app_data = Path(args.local_app_data).resolve()
    query_path = app_root / "samples" / "queries" / args.query
    config_path = app_root / "config.json"
    local_app_data.mkdir(parents=True, exist_ok=True)
    os.environ["LOCALAPPDATA"] = str(local_app_data)
    os.environ["FSTS_CONFIG_PATH"] = str(config_path)
    os.environ.pop("PYTHONHOME", None)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from orchestrator import run_task

    status = run_task(query_path, config_path=config_path, progress_callback=print)
    task_dir = Path(status["task_dir"])
    identity = json.loads((task_dir / "identity_result.json").read_text(encoding="utf-8"))
    source = json.loads((task_dir / "source_result.json").read_text(encoding="utf-8"))
    visualizations = [path for path in (task_dir / "visualizations").glob("*") if path.is_file()]

    identity_top1 = identity["candidates"][0]["identity_id"] if identity.get("candidates") else None
    source_top1 = (
        Path(source["candidates"][0]["file_path"]).name if source.get("candidates") else None
    )
    assertions = {
        "overall_success": status.get("status") == "success",
        "identity_success": identity.get("status") == "success",
        "source_success": source.get("status") == "success",
        "identity_candidate_count": len(identity.get("candidates", [])) == 5,
        "source_candidate_count": len(source.get("candidates", [])) == 5,
        "identity_rank1": identity_top1 == args.expected_identity,
        "source_rank1": source_top1 == args.expected_source,
        "visualization_count": len(visualizations) == 5,
        "visualizations_readable": all(path.stat().st_size > 0 for path in visualizations),
    }
    summary = {
        "status": "PASS" if all(assertions.values()) else "FAIL",
        "query": str(query_path),
        "task_id": status.get("task_id"),
        "task_dir": str(task_dir),
        "identity_rank1": identity_top1,
        "identity_score": identity.get("candidates", [{}])[0].get("score"),
        "source_rank1": source_top1,
        "source_score": source.get("candidates", [{}])[0].get("final_score"),
        "identity_runtime_seconds": status.get("identity", {}).get("runtime_seconds"),
        "source_runtime_seconds": status.get("source", {}).get("runtime_seconds"),
        "total_runtime_seconds": status.get("runtime_seconds"),
        "assertions": assertions,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    print(rendered)
    if args.summary:
        summary_path = Path(args.summary).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(rendered, encoding="utf-8")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
