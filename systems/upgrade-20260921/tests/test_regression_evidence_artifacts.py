from pathlib import Path

from tools.verify_regression_evidence import run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V3_TASK = (
    PROJECT_ROOT
    / "docs"
    / "regression_test_results"
    / "case_20260826_001607_552e"
)


def test_archived_real_query001_regression_reports_and_progress() -> None:
    """The archived run is real evidence, not a mock fixture."""
    result = run(PROJECT_ROOT, V3_TASK, write_outputs=False)

    assert result["comparison"]["uses_mock"] is False
    assert len(result["comparison"]["baseline_inventory"]) == 6
    assert result["comparison"]["deepfake"]["passed"] is True
    assert result["comparison"]["identity"]["passed"] is True
    assert result["comparison"]["source"]["passed"] is True
    assert result["validation"]["reports"]["passed"] is True
    assert result["validation"]["progress"]["passed"] is True
    assert result["overall_passed"] is True
