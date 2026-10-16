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
  tools/prepare_pretrained.py
  tools/prepare_ufd.py
  tools/prepare_waverep.py
  tools/prepare_video_fusion.py
  tools/prepare_asvspoof2019.py
  tools/prepare_asv_parquet.py
  tools/prepare_asv_eval.py
  tools/prepare_in_the_wild.py
  tools/finalize_asvspoof2019.py
  tools/prepare_vif_panda.py
  tools/prepare_vif_bench.py
  tools/prepare_asv_main.py
  tools/prepare_doppelganger.py
  tools/prepare_fakesound.py
  tools/prepare_envsdd.py
  tools/prepare_aime.py
  tools/prepare_ai_openbmat.py
  tools/prepare_audio_mixtures.py
  tools/prepare_universal_audio.py
  tools/prepare_video_control.py
  tools/prepare_video_multigen.py
  tools/prepare_video_size_control.py
  tools/report_video_pairs.py
  tools/prepare_video_augmentation.py
  tools/prepare_robustness.py
  tools/compare_scores.py
  tools/prepare_offline_package.py
  tools/build_release.py
  tools/test_prepare_asv_eval.py
  tools/test_prepare_in_the_wild.py
  tools/test_prepare_video_control.py
  tools/test_prepare_video_multigen.py
  tools/test_prepare_video_size_control.py
  tools/test_prepare_video_augmentation.py
)

uv run --no-sync ruff check src tests tools/check_environment.py "${model_tools[@]}"
uv run --no-sync ruff format --check src tests tools/check_environment.py "${model_tools[@]}"
uv run --no-sync mypy
uv run --no-sync mypy "${model_tools[@]}"
uv run --no-sync pytest -q
uv run --no-sync python -B -m unittest discover \
  -s tools -p 'test_*.py' -v
uv build --no-sources
