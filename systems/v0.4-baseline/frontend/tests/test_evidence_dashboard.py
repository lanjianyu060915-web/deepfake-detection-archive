from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


FRONTEND_ROOT = Path(__file__).resolve().parents[1]
if str(FRONTEND_ROOT) not in sys.path:
    sys.path.insert(0, str(FRONTEND_ROOT))

from components.data_loader import ResultDataError, validate_detection_data
from components.evidence_dashboard import (
    build_task_overview,
    collect_artifact_manifest,
    resolve_evidence_path,
)
from components.path_security import ResourcePathPolicy


class EvidenceDashboardPureTests(unittest.TestCase):
    def test_overview_uses_current_result_values(self) -> None:
        deepfake = {
            "query_image": "C:/task/query.jpg",
            "runtime_seconds": 1.25,
            "details": {
                "input_sha256": "a" * 64,
                "actual_device": "cpu",
                "media_metadata": {"width": 640, "height": 480},
            },
        }
        identity = {"runtime_seconds": 2.5}
        source = {"runtime_seconds": 3.75}
        overview = build_task_overview(
            deepfake,
            identity,
            source,
            task_status={"task_id": "case_dynamic", "status": "success", "runtime_seconds": 9.0},
            product_version="0.4.0",
        )
        self.assertEqual(overview["task_id"], "case_dynamic")
        self.assertEqual(overview["query_file"], "query.jpg")
        self.assertEqual(overview["dimensions"], "640 × 480")
        self.assertEqual(overview["input_sha256"], "a" * 64)
        self.assertEqual(overview["total_runtime"], 9.0)
        self.assertEqual(overview["deepfake_runtime"], 1.25)

    def test_manifest_is_dynamic_and_deduplicated(self) -> None:
        first = {
            "module": "deepfake",
            "artifact_manifest": [
                {"id": "heat", "kind": "heatmap", "path": "visualizations/heat.png"},
                {"id": "heat-copy", "kind": "heatmap", "path": "visualizations/heat.png"},
            ],
        }
        second = {
            "module": "identity",
            "visual_evidence": {
                "artifact_manifest": [
                    {"id": "aligned", "kind": "aligned_face", "path": "visualizations/aligned.png"}
                ]
            },
        }
        records = collect_artifact_manifest(first, second)
        self.assertEqual(len(records), 2)
        self.assertEqual({item["module"] for item in records}, {"deepfake", "identity"})

    def test_relative_evidence_path_stays_in_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            artifact = task / "visualizations" / "evidence.png"
            artifact.parent.mkdir()
            artifact.write_bytes(b"image")
            policy = ResourcePathPolicy(task_roots=(task,))
            resolved = resolve_evidence_path(
                "visualizations/evidence.png",
                policy,
                label="fixture",
            )
            self.assertEqual(resolved, artifact.resolve())
            self.assertIsNone(
                resolve_evidence_path("../escape.png", policy, label="escape")
            )


class ArtifactManifestValidationTests(unittest.TestCase):
    def _payload(
        self,
        query: Path,
        artifact_path: str | None,
        *,
        artifact_status: str | None = None,
    ) -> dict:
        return {
            "status": "success",
            "query_image": str(query),
            "is_deepfake": False,
            "score": 0.1,
            "label": "real",
            "runtime_seconds": 0.1,
            "details": {},
            "error_message": None,
            "visual_evidence": {},
            "artifact_manifest": [
                {
                    "id": "evidence",
                    "kind": "test",
                    "path": artifact_path,
                    "sha256": "a" * 64,
                    "size_bytes": 1,
                    **({"status": artifact_status} if artifact_status else {}),
                }
            ],
        }

    def test_task_relative_manifest_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            query = task / "query.jpg"
            query.write_bytes(b"q")
            policy = ResourcePathPolicy(task_roots=(task,))
            payload = self._payload(query, "visualizations/missing-but-safe.png")
            self.assertEqual(validate_detection_data(payload, policy)["status"], "success")

    def test_manifest_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            query = task / "query.jpg"
            query.write_bytes(b"q")
            policy = ResourcePathPolicy(task_roots=(task,))
            with self.assertRaises(ResultDataError):
                validate_detection_data(self._payload(query, "../escape.png"), policy)

    def test_optional_not_generated_artifact_may_have_no_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            query = task / "query.jpg"
            query.write_bytes(b"q")
            policy = ResourcePathPolicy(task_roots=(task,))
            payload = self._payload(query, None, artifact_status="not_generated")
            self.assertEqual(validate_detection_data(payload, policy)["status"], "success")

    def test_generated_artifact_without_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            query = task / "query.jpg"
            query.write_bytes(b"q")
            policy = ResourcePathPolicy(task_roots=(task,))
            with self.assertRaises(ResultDataError):
                validate_detection_data(self._payload(query, None), policy)


if __name__ == "__main__":
    unittest.main()
