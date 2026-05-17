#!/usr/bin/env bash
set -euo pipefail

REPO="/export/home3/phy/projects/VeriForm"
PY="$REPO/veriform/bin/python"
VLLM="$REPO/veriform/bin/vllm"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8101}"
# Never /tmp: node-local, ~2 GB, fills and crashes the node.
LOG_DIR="${LOG_DIR:-/dataset/phy/qlambda/tmp/veriform_phase2_workers/diag_persistent}"
mkdir -p "$LOG_DIR"

INPUT_DIR="$REPO/data/output/brokenmath-soft-perturbed/formalized"
OUTPUT_DIR="$REPO/data/output/_diag/brokenmath-soft-persistent-repl"
mkdir -p "$OUTPUT_DIR"

vllm_log="$LOG_DIR/vllm.log"
echo "=== starting vllm DeepSeek-Prover-V2-7B on GPU $GPU_ID port $PORT ==="
CUDA_VISIBLE_DEVICES="$GPU_ID" nohup "$VLLM" serve deepseek-ai/DeepSeek-Prover-V2-7B \
    --port "$PORT" --tensor-parallel-size 1 --dtype bfloat16 \
    --max-model-len 16384 --max-num-seqs 512 --enable-prefix-caching \
    > "$vllm_log" 2>&1 &
vpid=$!
trap "kill $vpid 2>/dev/null || true" EXIT

while kill -0 $vpid 2>/dev/null; do
    if grep -q "Application startup complete" "$vllm_log" 2>/dev/null \
        && curl -sf -o /dev/null "http://localhost:${PORT}/v1/models"; then
        break
    fi
    sleep 5
done
echo "=== vllm ready (pid=$vpid) ==="

run_log="$LOG_DIR/prove.log"
cd "$REPO"
"$PY" -u scripts/run_prove.py \
    --formalizer herald --p 1.0 --port "$PORT" \
    --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" \
    --num_samples 5 --shuffle --seed 999 \
    --concurrency 8 --lean_concurrency 16 \
    2>&1 | tee "$run_log"

echo "=== stopping vllm (pid=$vpid) ==="
kill $vpid 2>/dev/null || true
echo "=== DONE ==="
