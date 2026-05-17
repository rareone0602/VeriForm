#!/usr/bin/env bash
# Phase 1 driver for the BrokenMath perturbation arm. Mirrors
# run_phase1_all.sh but: (1) passes --perturber brokenmath, (2) routes output
# to data/output/brokenmath-<severity>-perturbed/formalized/, (3) only runs at
# p=1.0 equivalent — BrokenMath is full-coverage by construction (the JSONL
# contains a row per formalizable step).
#
# The unperturbed (p=0.0) baseline reuses the existing
# data/output/regex-perturbed/formalized/<f>/0/ pickles — same DAGs untouched.
#
# Usage:  bash scripts/run_phase1_brokenmath.sh <severity> [GPU_ID] [PORT] [formalizer ...]
# severity must be one of {soft, medium, hard}; selects both the JSONL file
# (new_perturbations_gpt5.2_<severity>.jsonl at repo root) and the output sub-tree.
# Defaults: GPU_ID=1, PORT=8002, all four formalizers.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <severity> [GPU_ID] [PORT] [formalizer ...]" >&2
    echo "       severity in {soft, medium, hard}" >&2
    exit 2
fi
SEVERITY="$1"; shift
case "$SEVERITY" in
    soft|medium|hard) ;;
    *) echo "invalid severity: $SEVERITY (expected soft|medium|hard)" >&2; exit 2 ;;
esac

GPU_ID="${1:-1}"
PORT="${2:-8002}"
shift 2 2>/dev/null || true
NAMES=("$@")
if [ ${#NAMES[@]} -eq 0 ]; then
    NAMES=(stepfun kimina goedel herald)
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/veriform/bin/python"
VLLM="$REPO/veriform/bin/vllm"
LOG_DIR="${LOG_DIR:-/tmp/veriform_logs}"
mkdir -p "$LOG_DIR"

OUTPUT_DIR="$REPO/data/output/brokenmath-${SEVERITY}-perturbed/formalized"
JSONL_PATH="$REPO/new_perturbations_gpt5.2_${SEVERITY}.jsonl"

if [ ! -f "$JSONL_PATH" ]; then
    echo "JSONL not found: $JSONL_PATH" >&2
    exit 2
fi

declare -A MODEL=(
    [stepfun]="stepfun-ai/StepFun-Formalizer-7B"
    [kimina]="AI-MO/Kimina-Autoformalizer-7B"
    [goedel]="Goedel-LM/Goedel-Formalizer-V2-8B"
    [herald]="FrenzyMath/Herald_translator"
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

for name in "${NAMES[@]}"; do
    model="${MODEL[$name]}"
    vllm_log="$LOG_DIR/vllm_${name}_brokenmath_${SEVERITY}.log"
    echo "=== [$name] starting vllm ($model) on GPU $GPU_ID port $PORT ==="
    pid=$(start_vllm "$model" "$vllm_log")
    trap 'stop_vllm '"$pid"' || true' EXIT
    if ! wait_ready "$pid" "$vllm_log"; then
        stop_vllm "$pid" || true
        exit 1
    fi
    echo "=== [$name] vllm ready (pid=$pid) ==="

    run_log="$LOG_DIR/${name}_brokenmath_${SEVERITY}_p100.log"
    echo "=== [$name] run_formalize.py --perturber brokenmath  (log: $run_log) ==="
    "$PY" scripts/run_formalize.py \
        --formalizer "$name" \
        --perturber brokenmath \
        --brokenmath_jsonl "$JSONL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --port "$PORT" \
        --concurrency 64 \
        --sampling deterministic \
        2>&1 | tee "$run_log"

    echo "=== [$name] stopping vllm (pid=$pid) ==="
    stop_vllm "$pid"
    trap - EXIT
done

echo "=== all formalizers done (brokenmath phase 1, severity=${SEVERITY}) ==="
