#!/bin/bash
# Task 1 full run — project root, venv active, server :65432 + vLLM :8000 running.

set -e
cd "$(dirname "$0")/../.."
source ./venv/bin/activate
B="python lmcache-vllm-extended/benchmark/run_task1.py"
CACHE=0.2

echo "========== Q1: length vs full response time =========="
$B q1 --cache-gb "$CACHE"

echo ""
echo "========== Q2: repeat back-to-back, full response time (cache=$CACHE GB) =========="
$B q2 --cache-gb "$CACHE"

echo ""
echo "========== Q2 cache sweep (optional) =========="
echo "For each size: edit configuration.yaml -> restart vLLM -> run:"
echo "  $B q2 --cache-gb 0.05"
echo "  $B q2 --cache-gb 0.1"
echo "  $B q2 --cache-gb 0.4"
echo "(Already ran 0.2 above; sweep PNG appears when >=2 q2_*.json exist)"

echo ""
echo "========== Q3: diversity =========="
$B q3 --cache-gb "$CACHE"

echo ""
echo "Output: lmcache-vllm-extended/benchmark/results/"
echo "  q1_cache${CACHE}.json  q1_cache${CACHE}.png"
echo "  q2_cache${CACHE}.*  [+ q2_cache_sweep.png]"
echo "  (gap repeat: python lmcache-vllm-extended/benchmark/run_repeat_gap.py)"
echo "  q3_cache${CACHE}.json  q3_cache${CACHE}.png"
