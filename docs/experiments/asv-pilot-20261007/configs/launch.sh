#!/usr/bin/env bash
# Prepared for later use only. This planning task does not invoke training.
set -euo pipefail

root='/home/maoxiao/WorkSpace/deepfake-detection-archive'
planning="$root/outputs/experiment-planning/asv-pilot"
run_name="${1:?Usage: bash launch.sh <logmel-seed42|aasist-l-seed42|logmel-seed43|aasist-l-seed43>}"
case "$run_name" in
  logmel-seed42|aasist-l-seed42|logmel-seed43|aasist-l-seed43) ;;
  *) printf 'Unknown planned run: %s\n' "$run_name" >&2; exit 2 ;;
esac
config="$planning/$run_name.json"
experiment="$root/outputs/experiments/asv-pilot/$run_name"
telemetry="$planning/telemetry/$run_name"
if [[ -e "$experiment" || -e "$telemetry" ]]; then
  printf 'Existing experiment or telemetry found; use a newly reviewed run ID/config.\n' >&2
  exit 2
fi
command -v nvidia-smi >/dev/null
command -v uv >/dev/null
test -x /usr/bin/time
test -f "$root/data/asvspoof2019-la-pilot/manifest-decoded.jsonl"
test -f "$root/data/asvspoof2019-la-pilot/sources.jsonl"
cd "$root"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/aigc-uv-cache}"
uv run --no-sync python - "$config" <<'PY'
import json
import sys
from pathlib import Path
from deepfake_detection_archive.data.manifest import load_manifest, load_sources
from deepfake_detection_archive.data.split import connected_groups, validate_splits
from deepfake_detection_archive.experiments import load_experiment_config
config = load_experiment_config(Path(sys.argv[1]))
records = load_manifest(Path(config.manifest))
assert len(records) == 1024, 'pilot requires exactly 1024 records'
assert all(row.modality == 'audio' and row.content_type == 'speech' for row in records)
assert all(row.split in ('train', 'dev') for row in records), 'holdout/eval input rejected'
assert all(row.decode_status == 'ok' and row.sha256 != 'unknown' for row in records)
assert not validate_splits(records, require_class_coverage=False)
groups = connected_groups(records)
for split, count in [('train', 20), ('dev', 10)]:
    selected = [row for row in records if row.split == split]
    assert len({groups[row.sample_id] for row in selected}) == count
    for label in (0, 1):
        labelled = [row for row in selected if row.label == label]
        assert len(labelled) == 256, (split, label, len(labelled))
        assert len({row.identity_id for row in labelled}) == count
    assert {row.generator for row in selected if row.label == 1} == {f'A{i:02}' for i in range(1, 7)}
sources = load_sources(Path(config.sources))
for source in {row.source for row in records}:
    entry = sources[source]
    assert entry.downloaded_at != 'unknown', 'real acquisition date is not registered'
    assert entry.license_status == 'approved' and 'training' in entry.allowed_uses
summary = json.loads((Path(config.manifest).parent / 'import-summary.json').read_text())
assert summary['official_counts_match'] is True and summary['eval_accessed'] is False
assert summary['limit_per_class_per_split'] == 256
print('Preflight verified 512 train / 512 dev, 20 / 10 groups, known decode/permission, no holdout.')
PY
mkdir -p "$planning/telemetry"
mkdir "$telemetry"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export LC_ALL=C

sha256sum "$config" "$planning/plan.json" > "$telemetry/planned-inputs.sha256"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$telemetry/started-at.txt"
git rev-parse HEAD > "$telemetry/git-head.txt"
git status --porcelain > "$telemetry/git-status.txt"
free -b > "$telemetry/memory-before.txt"
nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv --loop-ms=1000 > "$telemetry/gpu.csv" 2> "$telemetry/gpu-monitor.stderr" &
monitor_pid=$!
cleanup() {
  if kill -0 "$monitor_pid" 2>/dev/null; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if /usr/bin/time -v -o "$telemetry/process-time.txt" \
    uv run --no-sync deepfake-detection-archive train --config "$config" \
    > "$telemetry/train.stdout" 2> "$telemetry/train.stderr"; then
  exit_code=0
else
  exit_code=$?
fi
printf '%s\n' "$exit_code" > "$telemetry/exit-code.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$telemetry/finished-at.txt"
free -b > "$telemetry/memory-after.txt"
if [[ -s "$telemetry/gpu-monitor.stderr" ]]; then
  printf 'GPU monitor reported a problem; retain and inspect gpu-monitor.stderr.\n' >&2
fi
exit "$exit_code"
