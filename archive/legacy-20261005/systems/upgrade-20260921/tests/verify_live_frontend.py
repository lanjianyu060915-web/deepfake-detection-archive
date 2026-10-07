from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PRODUCT_ROOT / "frontend" / "app.py"
QUERY_PATH = (
    PRODUCT_ROOT.parent
    / "first_week"
    / "data"
    / "换脸查询图"
    / "query_001.jpg"
)
TASKS_ROOT = PRODUCT_ROOT / "runtime" / "tasks"
INBOX_ROOT = PRODUCT_ROOT / "runtime" / "inbox"


def _files(root: Path) -> set[Path]:
    if not root.is_dir():
        return set()
    return {path.resolve() for path in root.iterdir() if path.is_file()}


def main() -> int:
    before_tasks = {
        path.resolve() for path in TASKS_ROOT.iterdir() if path.is_dir()
    }
    before_inbox = _files(INBOX_ROOT)

    app = AppTest.from_file(APP_PATH, default_timeout=240)
    app.run()
    assert not app.exception
    assert app.radio[0].value == "Live Analysis"
    assert len(app.file_uploader) == 1
    assert any(button.label == "开始真实分析" for button in app.button)

    app.file_uploader[0].set_value(
        (QUERY_PATH.name, QUERY_PATH.read_bytes(), "image/jpeg")
    )
    next(
        button for button in app.button if button.label == "开始真实分析"
    ).click()
    app.run(timeout=240)

    assert not app.exception
    assert not app.error
    status = app.session_state["live_task_status"]
    loaded = app.session_state["loaded_results"]
    identity = loaded["identity"]
    source = loaded["source"]

    assert status["status"] == "success"
    assert status["identity"]["status"] == "success"
    assert status["source"]["status"] == "success"
    assert loaded["mode"] == "Live Analysis"
    assert len(identity["candidates"]) == 5
    assert len(source["candidates"]) == 5
    assert identity["candidates"][0]["identity_id"] == "person_001"
    assert Path(source["candidates"][0]["file_path"]).name == "122.jpg"
    assert all(
        Path(candidate["reference_image"]).is_file()
        for candidate in identity["candidates"]
    )
    assert all(
        Path(candidate["file_path"]).is_file()
        and Path(candidate["visualization_path"]).is_file()
        for candidate in source["candidates"]
    )
    assert len(app.tabs) == 3
    assert len(app.metric) >= 13

    after_tasks = {
        path.resolve() for path in TASKS_ROOT.iterdir() if path.is_dir()
    }
    new_tasks = after_tasks - before_tasks
    assert new_tasks == {Path(status["task_dir"]).resolve()}
    assert _files(INBOX_ROOT) == before_inbox

    evidence = {
        "status": "PASS",
        "task_id": status["task_id"],
        "task_dir": status["task_dir"],
        "default_mode": "Live Analysis",
        "identity_candidates_rendered": len(identity["candidates"]),
        "source_candidates_rendered": len(source["candidates"]),
        "identity_rank1": identity["candidates"][0]["identity_id"],
        "source_rank1": Path(source["candidates"][0]["file_path"]).name,
        "identity_references_renderable": 5,
        "source_candidates_renderable": 5,
        "visualizations_renderable": 5,
        "metrics": len(app.metric),
        "tabs": len(app.tabs),
        "page_exceptions": len(app.exception),
        "page_errors": len(app.error),
        "inbox_cleanup": True,
    }
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

