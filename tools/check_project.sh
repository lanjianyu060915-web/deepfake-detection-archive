#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

for script in scripts/devctl scripts/install-global-direnv-hook.sh tools/check_project.sh; do
  bash -n "$script"
done

model_tools=(
  tools/profile_models.py
  tools/prepare_asvspoof2019.py
  tools/finalize_asvspoof2019.py
  tools/prepare_vif_panda.py
  tools/prepare_asv_main.py
  tools/prepare_video_control.py
  tools/prepare_robustness.py
  tools/test_prepare_video_control.py
)

uv run --no-sync ruff check src tests tools/check_environment.py "${model_tools[@]}"
uv run --no-sync ruff format --check src tests tools/check_environment.py "${model_tools[@]}"
uv run --no-sync mypy
uv run --no-sync mypy "${model_tools[@]}"
uv run --no-sync pytest -q
uv run --no-sync python -B -m unittest discover \
  -s tools -p 'test_prepare_video_control.py' -v
uv run --no-sync python -B -m unittest discover \
  -s archive/legacy-20261005/validation/deepfake_acceptance_v2/evaluator/tests \
  -p 'test_*.py' -v
uv build --no-sources
