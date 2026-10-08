#!/usr/bin/env bash
# Prepared for later use only. This planning task does not invoke training.
set -euo pipefail

root='/home/maoxiao/WorkSpace/deepfake-detection-archive'
planning="$root/outputs/experiment-planning/asv-pilot-quarantine"
run_name="${1:?Usage: bash launch.sh <logmel-seed42|aasist-l-seed42|logmel-seed43|aasist-l-seed43>}"
case "$run_name" in
  logmel-seed42|aasist-l-seed42|logmel-seed43|aasist-l-seed43) ;;
  *) printf 'Unknown planned run: %s\n' "$run_name" >&2; exit 2 ;;
esac
config="$planning/$run_name.json"
experiment="$root/outputs/experiments/asv-pilot-quarantine/$run_name"
telemetry="$planning/telemetry/$run_name"
if [[ -e "$experiment" || -e "$telemetry" ]]; then
  printf 'Existing experiment or telemetry found; use a newly reviewed run ID/config.\n' >&2
  exit 2
fi
command -v nvidia-smi >/dev/null
command -v uv >/dev/null
test -x /usr/bin/time
test -f "$root/data/asvspoof2019-la-pilot-quarantine/manifest-decoded.jsonl"
test -f "$root/data/asvspoof2019-la-pilot/sources.jsonl"
cd "$root"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/aigc-uv-cache}"
uv run --no-sync python "$root/docs/experiments/asv-pilot-quarantine-20261008/source/check_pilot.py" --config "$config"
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
