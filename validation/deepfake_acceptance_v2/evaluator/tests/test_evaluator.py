from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EVALUATOR_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = EVALUATOR_ROOT / "evaluate_deepfake.py"
STUB = Path(__file__).resolve().with_name("stub_deepfake_runner.py")
sys.path.insert(0, str(EVALUATOR_ROOT))

from metrics import compute_binary_metrics, compute_roc_auc  # noqa: E402


FIXTURE_ROWS = [
    ("R001", "REAL", True, "", "", "not_applicable", "real", 0.10),
    ("F001", "FAKE", True, "", "", "stub_faceswap", "deepfake", 0.90),
    ("R002", "REAL", False, "R001", "JPEG", "not_applicable", "real", 0.20),
    ("F002", "FAKE", False, "F001", "JPEG", "stub_faceswap", "deepfake", 0.85),
    ("R003", "REAL", False, "R001", "Scale", "not_applicable", "real", 0.30),
    ("F003", "FAKE", False, "F001", "Scale", "stub_faceswap", "deepfake", 0.70),
    ("R004", "REAL", False, "R001", "Blur", "not_applicable", "real", 0.40),
    ("F004", "FAKE", False, "F001", "Blur", "stub_faceswap", "real", 0.35),
    ("R005", "REAL", False, "R001", "Brightness", "not_applicable", "deepfake", 0.80),
    ("F005", "FAKE", False, "F001", "Brightness", "stub_faceswap", "real", 0.25),
]


class EvaluatorTests(unittest.TestCase):
    def _make_fixture(self, root: Path, limit: int = 10) -> tuple[Path, Path]:
        data = root / "data"
        data.mkdir(parents=True)
        manifest = root / "ground_truth.csv"
        fixture_map = root / "fixture_map.json"
        fields = [
            "sample_id",
            "file_path",
            "label",
            "source",
            "generation_method",
            "source_face_id",
            "target_face_id",
            "is_original",
            "parent_sample_id",
            "perturbation_type",
            "perturbation_level",
            "width",
            "height",
            "sha256",
            "notes",
        ]
        map_payload: dict[str, dict[str, object]] = {}
        with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, row in enumerate(FIXTURE_ROWS[:limit], start=1):
                sample_id, truth, original, parent, kind, method, partial_label, score = row
                image = data / f"{sample_id}.png"
                image.write_bytes(b"D-V2-STUB-ONLY-" + bytes([index]))
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "file_path": str(image.relative_to(root)),
                        "label": truth,
                        "source": "synthetic evaluator fixture",
                        "generation_method": method,
                        "is_original": str(original).lower(),
                        "parent_sample_id": parent,
                        "perturbation_type": kind,
                        "perturbation_level": "fixture" if kind else "",
                        "width": 1,
                        "height": 1,
                        "sha256": "fixture-only",
                        "notes": "never passed to the real model",
                    }
                )
                map_payload[sample_id] = {
                    "truth": truth,
                    "partial_label": partial_label,
                    "score": score,
                }
        fixture_map.write_text(
            json.dumps({"samples": map_payload}, indent=2) + "\n", encoding="utf-8"
        )
        return manifest, fixture_map

    def _run_evaluator(
        self,
        root: Path,
        mode: str,
        *,
        limit: int = 10,
        timeout_seconds: float = 5.0,
        sleep_seconds: float = 2.0,
        score_semantics: str = "unconfirmed",
    ) -> tuple[dict, Path, subprocess.CompletedProcess[str]]:
        manifest, fixture_map = self._make_fixture(root, limit=limit)
        output = root / "evaluation_run"
        command = [
            sys.executable,
            "-S",
            "-B",
            str(EVALUATOR),
            "--python",
            sys.executable,
            "--runner",
            str(STUB),
            "--dataset",
            str(manifest),
            "--output-dir",
            str(output),
            "--timeout-seconds",
            str(timeout_seconds),
            "--score-semantics",
            score_semantics,
            "--python-arg=-S",
            "--python-arg=-B",
            f"--runner-arg=--stub-mode={mode}",
            f"--runner-arg=--fixture-map={fixture_map}",
            f"--runner-arg=--sleep-seconds={sleep_seconds}",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(completed.returncode, 0, msg=f"stdout={completed.stdout}\nstderr={completed.stderr}")
        report = json.loads((output / "evaluation_report.json").read_text(encoding="utf-8"))
        return report, output, completed

    def test_known_metric_fixture(self) -> None:
        pairs = (
            [("FAKE", "FAKE")] * 3
            + [("REAL", "FAKE")]
            + [("REAL", "REAL")] * 4
            + [("FAKE", "REAL")] * 2
        )
        metrics = compute_binary_metrics(pairs)
        self.assertEqual(
            (metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"]),
            (3, 1, 4, 2),
        )
        self.assertAlmostEqual(metrics["Accuracy"]["value"], 0.7)
        self.assertAlmostEqual(metrics["Precision"]["value"], 0.75)
        self.assertAlmostEqual(metrics["Recall"]["value"], 0.6)
        self.assertAlmostEqual(metrics["Specificity"]["value"], 0.8)
        self.assertAlmostEqual(metrics["F1"]["value"], 2 / 3)

    def test_zero_denominator_is_not_silently_zero(self) -> None:
        metrics = compute_binary_metrics([("REAL", "REAL")] * 5)
        self.assertIsNone(metrics["Precision"]["value"])
        self.assertIsNone(metrics["Recall"]["value"])
        self.assertIsNone(metrics["F1"]["value"])
        self.assertEqual(metrics["Precision"]["status"], "NOT_COMPUTED_ZERO_DENOMINATOR")
        self.assertEqual(metrics["Specificity"]["value"], 1.0)

    def test_auc_conditions_and_expected_value(self) -> None:
        labels = ["REAL"] * 5 + ["FAKE"] * 5
        scores = [0.10, 0.20, 0.30, 0.40, 0.80, 0.90, 0.85, 0.70, 0.35, 0.25]
        unconfirmed = compute_roc_auc(
            labels, scores, score_semantics="unconfirmed", all_predictions_usable=True
        )
        self.assertIsNone(unconfirmed["value"])
        confirmed = compute_roc_auc(
            labels,
            scores,
            score_semantics="higher_is_more_fake",
            all_predictions_usable=True,
        )
        self.assertAlmostEqual(confirmed["value"], 0.76)
        incomplete = compute_roc_auc(
            labels,
            scores[:-1] + [None],
            score_semantics="higher_is_more_fake",
            all_predictions_usable=True,
        )
        self.assertIsNone(incomplete["value"])

    def test_partial_wrong_end_to_end_and_grouping(self) -> None:
        with tempfile.TemporaryDirectory(prefix="d_v2_eval_partial_") as temporary:
            report, output, _ = self._run_evaluator(
                Path(temporary), "partial_wrong", score_semantics="higher_is_more_fake"
            )
            metrics = report["metrics"]
            self.assertEqual((metrics["N"], metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"]), (10, 3, 1, 4, 2))
            self.assertAlmostEqual(metrics["Accuracy"]["value"], 0.7)
            self.assertAlmostEqual(metrics["Precision"]["value"], 0.75)
            self.assertAlmostEqual(metrics["Recall"]["value"], 0.6)
            self.assertAlmostEqual(metrics["Specificity"]["value"], 0.8)
            self.assertAlmostEqual(metrics["F1"]["value"], 2 / 3)
            self.assertAlmostEqual(metrics["ROC_AUC"]["value"], 0.76)
            groups = {row["group"]: row for row in report["groups"]["perturbation"]}
            self.assertEqual(groups["Original"]["Accuracy"]["value"], 1.0)
            self.assertEqual(groups["JPEG"]["Accuracy"]["value"], 1.0)
            self.assertEqual(groups["Scale"]["Accuracy"]["value"], 1.0)
            self.assertEqual(groups["Blur"]["Accuracy"]["value"], 0.5)
            self.assertEqual(groups["Blur"]["Recall"]["value"], 0.0)
            self.assertEqual(groups["Blur"]["Specificity"]["value"], 1.0)
            self.assertEqual(groups["Brightness"]["Accuracy"]["value"], 0.0)
            generation = {
                row["group"]: row for row in report["groups"]["generation_method"]
            }
            self.assertEqual(generation["not_applicable"]["N"], 5)
            self.assertEqual(generation["not_applicable"]["Specificity"]["value"], 0.8)
            self.assertIsNone(generation["not_applicable"]["Recall"]["value"])
            self.assertEqual(generation["stub_faceswap"]["N"], 5)
            self.assertEqual(generation["stub_faceswap"]["Recall"]["value"], 0.6)
            self.assertIsNone(generation["stub_faceswap"]["Specificity"]["value"])
            self.assertTrue(all(row["descriptive_only"] for row in groups.values()))
            self.assertAlmostEqual(report["parent_aggregation"]["parent_macro_accuracy"], 0.7)
            with (output / "predictions.csv").open("r", encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 10)
            with (output / "false_negatives.csv").open("r", encoding="utf-8-sig", newline="") as handle:
                self.assertEqual({row["sample_id"] for row in csv.DictReader(handle)}, {"F004", "F005"})
            self.assertTrue((output / "run_manifest.json").is_file())
            for sample_id, *_ in FIXTURE_ROWS:
                sample_dir = output / "per_sample" / sample_id
                self.assertTrue((sample_dir / "result.json").is_file())
                self.assertTrue((sample_dir / "stdout.log").is_file())
                self.assertTrue((sample_dir / "stderr.log").is_file())
                self.assertTrue((sample_dir / "status.json").is_file())

    def test_nonfailure_stub_modes(self) -> None:
        expected = {
            "all_correct": (5, 0, 5, 0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "all_real": (0, 0, 5, 5, 0.5, None, 0.0, 1.0, 0.0),
            "all_fake": (5, 5, 0, 0, 0.5, 0.5, 1.0, 0.0, 2 / 3),
        }
        for mode, values in expected.items():
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix=f"d_v2_{mode}_") as temporary:
                report, _, _ = self._run_evaluator(Path(temporary), mode)
                metrics = report["metrics"]
                actual_counts = (metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"])
                self.assertEqual(actual_counts, values[:4])
                for metric_name, expected_value in zip(
                    ("Accuracy", "Precision", "Recall", "Specificity", "F1"),
                    values[4:],
                    strict=True,
                ):
                    actual = metrics[metric_name]["value"]
                    if expected_value is None:
                        self.assertIsNone(actual)
                    else:
                        self.assertAlmostEqual(actual, expected_value)

    def test_each_system_failure_keeps_fixed_denominator(self) -> None:
        expected_codes = {
            "crash": "RUNNER_NONZERO_EXIT",
            "invalid_json": "RESULT_INVALID_JSON",
            "status_error": "RESULT_STATUS_NOT_SUCCESS",
            "timeout": "RUNNER_TIMEOUT",
            "no_output": "RESULT_MISSING",
            "missing_label": "RESULT_LABEL_MISSING",
        }
        for mode, failure_code in expected_codes.items():
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix=f"d_v2_{mode}_") as temporary:
                report, output, _ = self._run_evaluator(
                    Path(temporary),
                    mode,
                    limit=2,
                    timeout_seconds=0.25 if mode == "timeout" else 5.0,
                    sleep_seconds=1.0,
                    score_semantics="higher_is_more_fake",
                )
                metrics = report["metrics"]
                self.assertEqual(metrics["N"], 2)
                self.assertEqual((metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"]), (0, 1, 0, 1))
                self.assertEqual(metrics["Accuracy"]["value"], 0.0)
                self.assertEqual(report["failures"]["count"], 2)
                self.assertEqual(report["failures"]["by_code"], {failure_code: 2})
                self.assertIsNone(metrics["ROC_AUC"]["value"])
                with (output / "predictions.csv").open("r", encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 2)
                self.assertEqual({row["effective_prediction"] for row in rows}, {"REAL", "FAKE"})

    def test_label_mapping_is_exact_not_fuzzy(self) -> None:
        # A standalone stub invocation makes a valid template, then changes only
        # the label spelling. The evaluator must count it wrong, not normalize it.
        with tempfile.TemporaryDirectory(prefix="d_v2_strict_label_") as temporary:
            root = Path(temporary)
            manifest, fixture_map = self._make_fixture(root, limit=2)
            custom_stub = root / "uppercase_stub.py"
            custom_stub.write_text(
                "import argparse,json,pathlib\n"
                "p=argparse.ArgumentParser();p.add_argument('--query');p.add_argument('--output');"
                "a,_=p.parse_known_args();q=pathlib.Path(a.query).resolve();o=pathlib.Path(a.output);"
                "o.parent.mkdir(parents=True,exist_ok=True);"
                "o.write_text(json.dumps({'status':'success','query_image':str(q),'label':'FAKE',"
                "'is_deepfake':True,'score':0.9}),encoding='utf-8')\n",
                encoding="utf-8",
            )
            output = root / "evaluation_run"
            command = [
                sys.executable,
                "-S",
                "-B",
                str(EVALUATOR),
                "--python",
                sys.executable,
                "--runner",
                str(custom_stub),
                "--dataset",
                str(manifest),
                "--output-dir",
                str(output),
                "--python-arg=-S",
                "--python-arg=-B",
                f"--runner-arg=--fixture-map={fixture_map}",
            ]
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            report = json.loads((output / "evaluation_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["failures"]["by_code"], {"RESULT_LABEL_INVALID": 2})


if __name__ == "__main__":
    unittest.main()
