#!/usr/bin/env bash
# Phase 1 driver: run scripts/run_formalize.py for stepfun, kimina, goedel
# at p=1.0 and p=0.0, on a single GPU. Brings vLLM up per formalizer,
# runs both p values against it, then shuts it down before the next.
#
# Usage:  bash scripts/run_phase1_all.sh [GPU_ID] [PORT]
# Defaults: GPU_ID=1, PORT=8002.
#
# Resume-safe: run_formalize.py skips chains whose pickle already exists,
# so re-invoking this script picks up where it left off.

set -euo pipefail

GPU_ID="${1:-1}"
PORT="${2:-8002}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/veriform/bin/python"
VLLM="$REPO/veriform/bin/vllm"
LOG_DIR="${LOG_DIR:-/tmp/veriform_logs}"
mkdir -p "$LOG_DIR"

declare -A MODEL=(
    [stepfun]="stepfun-ai/StepFun-Formalizer-7B"
    [kimina]="AI-MO/Kimina-Autoformalizer-7B"
    [goedel]="Goedel-LM/Goedel-Formalizer-V2-8B"
)

start_vllm() {
    local model="$1" log="$2"
    CUDA_VISIBLE_DEVICES="$GPU_ID" nohup "$VLLM" serve "$model" \
        --port "$PORT" \
        --tensor-parallel-size 1 \
        --dtype bfloat16 \
        --trust-remote-code \
        --enable-prefix-caching \
        --max-num-seqs 256 \
        > "$log" 2>&1 &
    echo $!
}

wait_ready() {
    local pid="$1" log="$2"
    while kill -0 "$pid" 2>/dev/null; do
        if grep -q "Application startup complete" "$log" 2>/dev/null \
            && curl -sf -o /dev/null "http://localhost:${PORT}/v1/models"; then
            return 0
        fi
        sleep 5
    done
    echo "vllm exited before becoming ready (see $log)" >&2
    return 1
}

stop_vllm() {
    local pid="$1"
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$pid" 2>/dev/null || return 0
            sleep 1
        done
        kill -9 "$pid" 2>/dev/null || true
    fi
}

cd "$REPO"

for name in stepfun kimina goedel; do
    model="${MODEL[$name]}"
    vllm_log="$LOG_DIR/vllm_${name}.log"
    echo "=== [$name] starting vllm ($model) on GPU $GPU_ID port $PORT ==="
    pid=$(start_vllm "$model" "$vllm_log")
    trap 'stop_vllm '"$pid"' || true' EXIT
    if ! wait_ready "$pid" "$vllm_log"; then
        stop_vllm "$pid" || true
        exit 1
    fi
    echo "=== [$name] vllm ready (pid=$pid) ==="

    for p in 1.0 0.0; do
        run_log="$LOG_DIR/${name}_p$(printf '%03d' $(awk -v x="$p" 'BEGIN{printf "%d", x*100+0.5}')).log"
        echo "=== [$name] run_formalize.py --p $p  (log: $run_log) ==="
        "$PY" scripts/run_formalize.py \
            --formalizer "$name" \
            --p "$p" \
            --port "$PORT" \
            --concurrency 64 \
            --sampling deterministic \
            2>&1 | tee "$run_log"
    done

    echo "=== [$name] stopping vllm (pid=$pid) ==="
    stop_vllm "$pid"
    trap - EXIT
done

echo "=== all formalizers done ==="
