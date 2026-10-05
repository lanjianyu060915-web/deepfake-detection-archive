from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "test_results" / "regression_matrix"
BASELINE = ROOT / "docs" / "upgrade_v0.4.0" / "baseline_case_20260831_114829_c45e.json"
AIGC_NORMAL = (
    ROOT
    / "test_results"
    / "aigc_visual_evidence"
    / "case_20260831_114829_c45e_final"
    / "detection_result.json"
)
AIGC_NO_FACE = (
    ROOT
    / "test_results"
    / "aigc_visual_evidence"
    / "no_face"
    / "detection_result.json"
)
INSTALLED_TASK = (
    Path.home()
    / "AppData"
    / "Local"
    / "FaceSourceTrace"
    / "tasks"
    / "case_20260831_114829_c45e"
)


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), path
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_close(actual: float, expected: float, tolerance: float) -> float:
    delta = abs(float(actual) - float(expected))
    assert delta <= tolerance, (actual, expected, delta, tolerance)
    return delta


def _validate_manifest(result_path: Path, payload: dict) -> dict:
    manifest = payload.get("artifact_manifest")
    assert isinstance(manifest, list) and manifest, result_path
    maximum = 0
    kinds = []
    for item in manifest:
        assert isinstance(item, dict)
        relative = PurePosixPath(str(item["path"]))
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        assert "\\" not in str(item["path"])
        artifact = result_path.parent.joinpath(*relative.parts)
        assert artifact.is_file(), artifact
        assert artifact.stat().st_size == int(item["size_bytes"])
        assert _sha256(artifact).lower() == str(item["sha256"]).lower()
        maximum = max(maximum, artifact.stat().st_size)
        kinds.append(str(item.get("kind")))
    return {
        "artifact_count": len(manifest),
        "kinds": sorted(set(kinds)),
        "largest_artifact_bytes": maximum,
        "all_paths_task_relative_posix": True,
        "all_hashes_and_sizes_verified": True,
    }


def _read_progress(path: Path) -> dict:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events, path
    percentages = [
        float(event["progress_percent"])
        for event in events
        if event.get("progress_percent") is not None
    ]
    assert percentages == sorted(percentages), path
    assert percentages[-1] == 100.0, path
    return {
        "event_count": len(events),
        "first_stage": events[0]["stage"],
        "last_stage": events[-1]["stage"],
        "last_status": events[-1]["status"],
        "monotonic_progress": True,
    }


def _junit(path: Path) -> dict:
    tree = ET.parse(path)
    cases = tree.findall(".//testcase")
    failures = tree.findall(".//failure")
    errors = tree.findall(".//error")
    skipped = tree.findall(".//skipped")
    assert cases and not failures and not errors, path
    return {
        "testcases": len(cases),
        "failures": len(failures),
        "errors": len(errors),
        "skipped": len(skipped),
    }


def main() -> int:
    baseline = _read(BASELINE)
    tolerance = float(baseline["maximum_allowed_absolute_error"])
    expected_hash = baseline["input"]["sha256"].lower()

    aigc = _read(AIGC_NORMAL)
    identity_path = MATRIX / "normal_identity" / "identity_result.json"
    source_path = MATRIX / "normal_source" / "source_result.json"
    identity = _read(identity_path)
    source = _read(source_path)
    assert aigc["status"] == identity["status"] == source["status"] == "success"

    normal_deltas = {
        "aigc_fusion": _assert_close(
            aigc["score"], baseline["deepfake"]["fusion_score"], tolerance
        ),
        "aigc_spatial": _assert_close(
            aigc["metrics"]["spatial_score"],
            baseline["deepfake"]["spatial_score"],
            tolerance,
        ),
        "aigc_frequency": _assert_close(
            aigc["metrics"]["frequency_score"],
            baseline["deepfake"]["frequency_score"],
            tolerance,
        ),
        "aigc_threshold": _assert_close(
            aigc["metrics"]["decision_threshold"],
            baseline["deepfake"]["decision_threshold"],
            tolerance,
        ),
    }
    actual_identity_ranking = [
        [item["identity_id"], float(item["score"])]
        for item in identity["candidates"]
    ]
    assert [item[0] for item in actual_identity_ranking] == [
        item[0] for item in baseline["identity"]["ranking"]
    ]
    for actual, expected in zip(
        actual_identity_ranking, baseline["identity"]["ranking"], strict=True
    ):
        normal_deltas[f"identity_{actual[0]}"] = _assert_close(
            actual[1], expected[1], tolerance
        )
    normal_deltas["identity_margin"] = _assert_close(
        identity["metrics"]["top1_top2_margin"],
        baseline["identity"]["top1_top2_margin"],
        tolerance,
    )
    assert identity["candidates"][0]["best_reference_file_name"] == baseline[
        "identity"
    ]["best_reference_file"]

    actual_source_ranking = [
        [
            Path(item["file_path"]).name,
            float(item["global_score"]),
            float(item["local_score"]),
            float(item["final_score"]),
        ]
        for item in source["candidates"]
    ]
    assert [item[0] for item in actual_source_ranking] == [
        item[0] for item in baseline["source"]["ranking"]
    ]
    for actual, expected in zip(
        actual_source_ranking, baseline["source"]["ranking"], strict=True
    ):
        for index, label in ((1, "global"), (2, "local"), (3, "final")):
            normal_deltas[f"source_{actual[0]}_{label}"] = _assert_close(
                actual[index], expected[index], tolerance
            )
    top_source = source["candidates"][0]
    assert top_source["good_match_count"] == baseline["source"][
        "top1_good_match_count"
    ]
    assert top_source["ransac_inlier_count"] == baseline["source"][
        "top1_ransac_inlier_count"
    ]
    normal_deltas["source_ransac_ratio"] = _assert_close(
        top_source["ransac_inlier_ratio"],
        baseline["source"]["top1_ransac_inlier_ratio"],
        tolerance,
    )
    normal_deltas["source_margin"] = _assert_close(
        source["metrics"]["top1_top2_margin"],
        baseline["source"]["top1_top2_margin"],
        tolerance,
    )
    assert max(normal_deltas.values()) <= tolerance
    assert aigc["provenance"]["input_sha256"].lower() == expected_hash
    assert source["provenance"]["input_sha256"].lower() == expected_hash

    no_face = _read(AIGC_NO_FACE)
    assert no_face["status"] == "success"
    assert no_face["label"] == "no_face" and no_face["score"] is None
    assert no_face["visual_evidence"]["deepfake"]["face_detection"]["detected_face_count"] == 0

    multi = _read(MATRIX / "multi_face_identity" / "identity_result.json")
    assert multi["status"] == "error" and multi["metrics"]["query_face_count"] == 2
    assert "2 张人脸" in multi["error_message"]

    empty_identity = _read(MATRIX / "empty_identity" / "identity_result.json")
    assert empty_identity["status"] == "error"
    assert empty_identity["metrics"]["library_identity_count"] == 0
    empty_source = _read(MATRIX / "empty_source" / "source_result.json")
    assert empty_source["status"] == "error" and not empty_source["candidates"]

    corrupt_identity_path = MATRIX / "corrupt_identity" / "identity_result.json"
    corrupt_identity = _read(corrupt_identity_path)
    assert corrupt_identity["status"] == "success"
    assert corrupt_identity["metrics"]["skipped_library_image_count"] == 1
    skipped_identity = corrupt_identity["candidates"][0]["skipped_reference_images"]
    assert skipped_identity[0]["reason"] == "ImageDecodeError"
    corrupt_source_path = MATRIX / "corrupt_source" / "source_result.json"
    corrupt_source = _read(corrupt_source_path)
    assert corrupt_source["status"] == "success"
    assert corrupt_source["metrics"]["library_file_count"] == 3
    assert corrupt_source["metrics"]["library_decoded"] == 2
    assert corrupt_source["metrics"]["library_skipped_count"] == 1

    frontend_state = _read(MATRIX / "frontend_state.json")
    assert frontend_state["status"] == "passed"
    assert frontend_state["page_switch_did_not_create_task"] is True
    assert frontend_state["fresh_session_reload"]["passed"] is True

    report_txt = INSTALLED_TASK / "检测结果_case_20260831_114829_c45e.txt"
    report_html = INSTALLED_TASK / "检测结果_case_20260831_114829_c45e.html"
    for report in (report_txt, report_html):
        assert report.is_file() and report.stat().st_size > 0, report
        assert "case_20260831_114829_c45e" in report.read_text(encoding="utf-8")

    static_log = _read(MATRIX / "packaging_static.log.json")
    assert static_log["exit_code"] == 0
    assert "PACKAGING_STATIC_VALIDATION=PASSED" in static_log["stdout"]
    static_result = _read(ROOT / "packaging_v0.3" / "static_validation.json")
    assert static_result["status"] == "passed"
    assert static_result["product_version"] == "0.4.0"
    assert static_result["fixed_drive_path_findings"] == 0
    packaged_config = _read(ROOT / "packaging_v0.3" / "config.packaged.json")
    assert packaged_config["build_flavor"] == "formal"
    assert packaged_config["features"]["real_modules_default"] is True
    assert packaged_config["features"]["mock_enabled"] is False
    assert packaged_config["identity"]["offline"] is True
    assert (ROOT / "packaging_v0.3" / "resources" / "portable_python" / "python.exe").is_file()

    junit = {
        "aigc_offline_evidence": _junit(
            MATRIX / "aigc_offline_evidence.junit.xml"
        ),
        "identity_evidence": _junit(MATRIX / "identity_evidence.junit.xml"),
        "source_edges": _junit(MATRIX / "source_edges.junit.xml"),
        "orchestrator_launcher": _junit(
            MATRIX / "orchestrator_launcher.junit.xml"
        ),
    }
    report_tests = _read(MATRIX / "frontend_reports.log.json")
    assert report_tests["exit_code"] == 0
    assert "Ran 19 tests" in report_tests["stderr"]
    assert "OK" in report_tests["stderr"]

    output = {
        "schema_version": "1.0",
        "status": "passed",
        "baseline_task_id": baseline["task_id"],
        "maximum_allowed_absolute_error": tolerance,
        "maximum_observed_absolute_error": max(normal_deltas.values()),
        "normal_case": {
            "query_sha256": expected_hash,
            "three_module_status": {
                "deepfake": aigc["status"],
                "identity": identity["status"],
                "source": source["status"],
            },
            "deltas": normal_deltas,
            "deepfake_manifest": _validate_manifest(AIGC_NORMAL, aigc),
            "identity_manifest": _validate_manifest(identity_path, identity),
            "source_manifest": _validate_manifest(source_path, source),
            "identity_progress": _read_progress(
                MATRIX / "normal_identity" / "logs" / "progress_events.jsonl"
            ),
            "source_progress": _read_progress(
                MATRIX / "normal_source" / "logs" / "progress_events.jsonl"
            ),
        },
        "edge_cases": {
            "no_face": {"status": no_face["status"], "label": no_face["label"]},
            "multi_face": {
                "status": multi["status"],
                "query_face_count": multi["metrics"]["query_face_count"],
            },
            "identity_empty_library": empty_identity["status"],
            "source_empty_library": empty_source["status"],
            "identity_corrupt_image": {
                "status": corrupt_identity["status"],
                "skipped": corrupt_identity["metrics"]["skipped_library_image_count"],
                "reason": skipped_identity[0]["reason"],
                "manifest": _validate_manifest(
                    corrupt_identity_path, corrupt_identity
                ),
            },
            "source_corrupt_image": {
                "status": corrupt_source["status"],
                "decoded": corrupt_source["metrics"]["library_decoded"],
                "skipped": corrupt_source["metrics"]["library_skipped_count"],
                "manifest": _validate_manifest(corrupt_source_path, corrupt_source),
            },
        },
        "frontend_state": frontend_state,
        "reports": {
            "txt": {
                "path": str(report_txt),
                "size_bytes": report_txt.stat().st_size,
                "sha256": _sha256(report_txt),
            },
            "html": {
                "path": str(report_html),
                "size_bytes": report_html.stat().st_size,
                "sha256": _sha256(report_html),
            },
            "current_builder_tests": 19,
        },
        "packaging_static": static_result,
        "junit": junit,
    }
    destination = MATRIX / "verification.json"
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
