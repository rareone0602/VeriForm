#!/usr/bin/env bash
# Phase 2 driver for the BrokenMath perturbation arm. Mirrors
# run_phase2_all.sh but reads phase-1 pickles from
# data/output/brokenmath-<severity>-perturbed/formalized/ and writes proved
# pickles to data/output/brokenmath-<severity>-perturbed/proved/, only at p=1.0
# (BrokenMath is full-coverage; no probabilistic p=0.0 arm).
#
# Usage:  bash scripts/run_phase2_brokenmath.sh <severity> [GPU_ID] [PORT] [formalizer ...]
# severity must be one of {soft, medium, hard}.
# Defaults: GPU_ID=1, PORT=8001, all four formalizers.

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
PORT="${2:-8001}"
shift 2 2>/dev/null || true
NAMES=("$@")
if [ ${#NAMES[@]} -eq 0 ]; then
    NAMES=(stepfun kimina goedel herald)
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/veriform/bin/python"
VLLM="$REPO/veriform/bin/vllm"
# Never default to /tmp: it is node-local and small, and a full /tmp takes
# the whole compute node down. Use the canonical user-scratch path on the
# shared 186 TB pool.
LOG_DIR="${LOG_DIR:-/dataset/phy/qlambda/tmp/veriform_phase2_workers}"
mkdir -p "$LOG_DIR"

# vLLM/Triton/Inductor default to /tmp; node-local /tmp is ~2 GB and fills
# fast. Job 492038 (stepfun on node15) died with "No space left on device"
# during vLLM init for exactly this reason. Redirect to shared pool.
export TMPDIR="${TMPDIR:-/dataset/phy/qlambda/tmp/vllm_phase2_tmp/${SLURM_JOB_ID:-$$}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$TMPDIR/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$TMPDIR/vllm}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

INPUT_DIR="$REPO/data/output/brokenmath-${SEVERITY}-perturbed/formalized"
OUTPUT_DIR="$REPO/data/output/brokenmath-${SEVERITY}-perturbed/proved"

if [ ! -d "$INPUT_DIR" ]; then
    echo "phase-1 input dir not found: $INPUT_DIR" >&2
    echo "did you run phase 1 for severity=$SEVERITY first?" >&2
    exit 2
fi

PROVER_MODEL="deepseek-ai/DeepSeek-Prover-V2-7B"

start_vllm() {
    local model="$1" log="$2"
    # No --gpu-memory-utilization: defer to vLLM's default (0.9). DSP-V2
    # owns the GPU in this loop, so the prior explicit 0.5 was leaving
    # ~70 GB of KV cache on the table — removing it lets vLLM use what
    # it would have anyway.
    # Honor SLURM's cgroup GPU assignment (CUDA_VISIBLE_DEVICES is preset
    # by SLURM under sbatch/srun); only override for bare-shell use.
    local devs="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
    CUDA_VISIBLE_DEVICES="$devs" nohup "$VLLM" serve "$model" \
        --port "$PORT" \
        --tensor-parallel-size 1 \
        --dtype bfloat16 \
        --max-model-len 16384 \
        --max-num-seqs 512 \
        --enable-prefix-caching \
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

# Preflight: rebind CUDA_VISIBLE_DEVICES only if SLURM gave us a BUSY GPU.
# See run_phase2_all.sh preflight_rebind() for the rationale; logic
# duplicated here so brokenmath jobs (Phase C) inherit the fix.
# Priority: (1) respect SLURM's clean assignment. (2)/(3) fall back to a
# clean GPU on the node. Critically: do NOT rebind on the basis of "this
# other GPU is slightly cleaner" — sibling jobs would all stampede onto it.
if command -v nvidia-smi >/dev/null 2>&1; then
    _probe=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)
    if [[ -n "$_probe" ]]; then
        _cur="${CUDA_VISIBLE_DEVICES:-}"
        _cur_mem=""
        if [[ "$_cur" =~ ^[0-9]+$ ]]; then
            _cur_mem=$(echo "$_probe" | awk -F, -v i="$_cur" 'BEGIN{i+=0}{idx=$1+0; if(idx==i){print $2+0; exit}}')
        fi
        if [[ -z "$_cur_mem" || "$_cur_mem" -gt 500 ]]; then
            _best_idx=""; _best_mem=10000000
            while IFS=, read -r _idx _mem; do
                _idx="${_idx// /}"; _mem="${_mem// /}"
                [[ -z "$_idx" || -z "$_mem" ]] && continue
                if (( _mem < _best_mem )); then _best_mem=$_mem; _best_idx=$_idx; fi
            done <<< "$_probe"
            if [[ -z "$_best_idx" || $_best_mem -gt 500 ]]; then
                echo "PREFLIGHT: no clean GPU on $(hostname -s); SLURM_GPU=${_cur:-unset} (${_cur_mem:-?}MiB), cheapest=GPU${_best_idx:-?} with ${_best_mem}MiB. Aborting." >&2
                exit 1
            fi
            mkdir -p ~/logs
            printf '%s\tjobid=%s\tnode=%s\trebind_from=%s(mem=%sMiB)\trebind_to=%s(mem=%sMiB)\treason=preflight_double_alloc\n' \
                "$(date -Is)" "${SLURM_JOB_ID:-?}" "$(hostname -s)" "${_cur:-unset}" "${_cur_mem:-?}" "$_best_idx" "$_best_mem" \
                >> ~/logs/slurm_gpu_collisions.log
            echo "PREFLIGHT: rebinding CUDA_VISIBLE_DEVICES ${_cur:-unset} -> $_best_idx (orig had ${_cur_mem:-?}MiB used; new has ${_best_mem}MiB)" >&2
            export CUDA_VISIBLE_DEVICES="$_best_idx"
        fi
    fi
fi

vllm_log="$LOG_DIR/vllm_prover_brokenmath_${SEVERITY}.log"
echo "=== starting vllm ($PROVER_MODEL) on GPU $GPU_ID port $PORT (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}) ==="
pid=$(start_vllm "$PROVER_MODEL" "$vllm_log")
trap 'stop_vllm '"$pid"' || true' EXIT
if ! wait_ready "$pid" "$vllm_log"; then
    stop_vllm "$pid" || true
    exit 1
fi
echo "=== vllm ready (pid=$pid) ==="

for name in "${NAMES[@]}"; do
    run_log="$LOG_DIR/prove_brokenmath_${SEVERITY}_${name}_p100.log"
    echo "=== [$name] run_prove.py --p 1.0  (log: $run_log) ==="
    # Each Lean4ServerProcess can hold ~2000 mmap'd Mathlib oleans; with
    # 64 workers we hit the system-wide ENFILE cap (file-max=131072) and
    # every TC call returns pass=False → 100% TC_FAIL. Cap at 16 to stay
    # well under that ceiling.
    "$PY" scripts/run_prove.py \
        --formalizer "$name" \
        --p 1.0 \
        --port "$PORT" \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --concurrency 8 \
        --lean_concurrency 16 \
        ${NUM_SAMPLES:+--num_samples "$NUM_SAMPLES"} \
        2>&1 | tee "$run_log"
done

echo "=== stopping vllm (pid=$pid) ==="
stop_vllm "$pid"
trap - EXIT

echo "=== all formalizers done (brokenmath phase 2, severity=${SEVERITY}) ==="
