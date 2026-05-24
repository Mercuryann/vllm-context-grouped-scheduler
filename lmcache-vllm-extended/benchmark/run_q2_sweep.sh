#!/bin/bash
# Q2 full cache sweep: 4 cache sizes × 2 experiments = 8 JSON runs, but only 4 vLLM restarts.
#
# Usage (from repo root, venv active):
#   bash lmcache-vllm-extended/benchmark/run_q2_sweep.sh
#
# Optional env:
#   CACHE_SIZES="0.05 0.1 0.2 0.4"
#   SKIP_RESTART=1          # you restarted vLLM yourself before each block
#   VLLM_HEALTH_URL=http://127.0.0.1:8000/health
#   LMCACHE_CONFIG=lmcache-vllm-extended/configuration.yaml

set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="${LMCACHE_CONFIG:-lmcache-vllm-extended/configuration.yaml}"
CACHE_SIZES="${CACHE_SIZES:-0.05 0.1 0.2 0.4}"
TASK1="python lmcache-vllm-extended/benchmark/run_task1.py"
GAP="python lmcache-vllm-extended/benchmark/run_repeat_gap.py"
PLOT="python lmcache-vllm-extended/benchmark/plot_q2_first_repeat_lines.py"
HEALTH="${VLLM_HEALTH_URL:-http://127.0.0.1:8000/health}"

patch_cache() {
  local gb="$1"
  if grep -q '^max_local_cache_size:' "$CONFIG"; then
    sed -i "s/^max_local_cache_size:.*/max_local_cache_size: ${gb}/" "$CONFIG"
  else
    echo "max_local_cache_size: ${gb}" >> "$CONFIG"
  fi
  echo "  -> $CONFIG now max_local_cache_size=${gb}"
}

wait_for_vllm() {
  local tries=60
  for ((i = 1; i <= tries; i++)); do
    if curl -sf "$HEALTH" >/dev/null 2>&1; then
      echo "  vLLM ready ($HEALTH)"
      return 0
    fi
    sleep 2
  done
  echo "  WARN: vLLM health check failed; continue anyway if API is up."
}

echo "=== Q2 cache sweep ==="
echo "Config: $CONFIG"
echo "Sizes:  $CACHE_SIZES"
echo ""

for gb in $CACHE_SIZES; do
  echo "########################################"
  echo "### cache ${gb} GB"
  echo "########################################"
  patch_cache "$gb"

  if [[ "${SKIP_RESTART:-0}" != "1" ]]; then
    echo ""
    echo ">>> Restart vLLM (+ LMCache server if needed), then press Enter."
    echo "    Example:"
    echo "      LMCACHE_CONFIG_FILE=$CONFIG CUDA_VISIBLE_DEVICES=0 \\"
    echo "        python lmcache-vllm-extended/lmcache_vllm/script.py serve ..."
    read -r _
    wait_for_vllm
  fi

  echo "--- q2 back-to-back (56 reqs) ---"
  $TASK1 q2 --cache-gb "$gb" --no-plot --no-token-count

  echo "--- repeat_gap two-phase (28 reqs) ---"
  $GAP --cache-gb "$gb" --no-plot --no-token-count

  echo ""
done

echo "=== Plot all (response time cache sweep) ==="
$PLOT

echo "=== q2_cache_sweep.png (throughput + first/repeat) ==="
python -c "
from pathlib import Path
import sys
sys.path.insert(0, 'lmcache-vllm-extended/benchmark')
import run_task1 as r
r.plot_q2_cache_sweep(Path('lmcache-vllm-extended/benchmark/results'))
"

echo "Done: results/q2_cache*.json, repeat_gap_cache*.json,"
echo "      q2_avg_response_*.png, q2_cache_sweep.png"
