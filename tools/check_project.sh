#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

model_tools=(
  tools/profile_models.py
  tools/prepare_asvspoof2019.py
  tools/finalize_asvspoof2019.py
  tools/prepare_vif_panda.py
)

uv run --no-sync ruff check src tests tools/check_environment.py "${model_tools[@]}"
uv run --no-sync ruff format --check src tests tools/check_environment.py "${model_tools[@]}"
uv run --no-sync mypy
uv run --no-sync mypy "${model_tools[@]}"
uv run --no-sync pytest -q
uv run --no-sync python -B -m unittest discover \
  -s archive/legacy-20261005/validation/deepfake_acceptance_v2/evaluator/tests \
  -p 'test_*.py' -v
uv build --no-sources
