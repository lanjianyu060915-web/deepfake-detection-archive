"""Run the frozen audit with result documents redirected to this directory."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "docs/research/source/vif-pilot/prepare_pilot.py"
spec = importlib.util.spec_from_file_location("vif_frozen_audit", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load frozen audit source")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.__dict__["DOC"] = Path(__file__).resolve().parent

if __name__ == "__main__":
    module.main()
