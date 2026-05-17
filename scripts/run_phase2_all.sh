#!/usr/bin/env bash
# Phase 2 driver: run scripts/run_prove.py for stepfun, kimina, goedel,
# herald at p=1.0 and p=0.0, on a single GPU. The prover model is the same
# (DeepSeek-Prover-V2-7B) regardless of which formalizer produced phase-1,
# so we boot vLLM ONCE at the top of the script and reuse it for every
# (formalizer, p) combo.
#
# Usage:  bash scripts/run_phase2_all.sh [GPU_ID] [PORT] [formalizer ...]
# Defaults: GPU_ID=0, PORT=8001, all four formalizers.
# Example: bash scripts/run_phase2_all.sh 0 8001 herald
#
# Resume-safe: run_prove.py skips chains whose output pickle already exists.

set -euo pipefail

GPU_ID="${1:-0}"
PORT="${2:-8001}"
shift 2 2>/dev/null || true
NAMES=("$@")
if [ ${#NAMES[@]} -eq 0 ]; then
    NAMES=(stepfun kimina goedel herald)
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/veriform/bin/python"
VLLM="$REPO/veriform/bin/vllm"
# Never /tmp: node-local, ~2 GB, fills and crashes the node.
LOG_DIR="${LOG_DIR:-/dataset/phy/qlambda/tmp/veriform_phase2_workers}"
mkdir -p "$LOG_DIR"

# vLLM/Triton/Inductor default to /tmp for engine temp files and JIT cache.
# Node-local /tmp is ~2 GB and fills fast (job 492038 on node15 died with
# "No space left on device" mid-vLLM-init). Redirect to the shared pool so
# we don't depend on whichever node SLURM happens to land us on.
export TMPDIR="${TMPDIR:-/dataset/phy/qlambda/tmp/vllm_phase2_tmp/${SLURM_JOB_ID:-$$}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$TMPDIR/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$TMPDIR/vllm}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

PROVER_MODEL="deepseek-ai/DeepSeek-Prover-V2-7B"

preflight_rebind() {
    # SLURM on this cluster occasionally hands out a GPU that's already in
    # use by another job (cgroup isolation isn't strict; nvidia-smi from
    # inside the allocation can still see all GPUs on the node). When that
    # happens, vLLM fails with "Free memory on device cuda:0 ... GiB". The
    # admin-policy default is to log + abort, but for this Phase 2 rollout
    # the user has authorised silent rebinding. Every rebind is appended
    # to ~/logs/slurm_gpu_collisions.log so the admin signal stays intact.
    #
    # IMPORTANT: only rebind if SLURM's assigned GPU is *itself* busy.
    # If SLURM gave us a clean GPU (even if it's not the "cheapest" on the
    # node), respect that — otherwise sibling jobs on the same node all
    # pick the same "cheapest" GPU and collide on it.
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local probe
    probe=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null) || return 0
    local cur="${CUDA_VISIBLE_DEVICES:-}"
    local cur_mem=""
    if [[ "$cur" =~ ^[0-9]+$ ]]; then
        cur_mem=$(echo "$probe" | awk -F, -v i="$cur" 'BEGIN{i+=0}{idx=$1+0; if(idx==i){print $2+0; exit}}')
    fi
    # SLURM's GPU is clean → no action needed.
    if [[ -n "$cur_mem" ]] && (( cur_mem <= 500 )); then
        return 0
    fi
    # SLURM's GPU is busy (or we have no assigned GPU yet): find cheapest.
    local best_idx="" best_mem=10000000
    while IFS=, read -r idx mem; do
        idx="${idx// /}"; mem="${mem// /}"
        [[ -z "$idx" || -z "$mem" ]] && continue
        if (( mem < best_mem )); then best_mem=$mem; best_idx=$idx; fi
    done <<< "$probe"
    if [[ -z "$best_idx" ]]; then return 0; fi
    if (( best_mem > 500 )); then
        echo "PREFLIGHT: no clean GPU on $(hostname -s); SLURM_GPU=$cur (${cur_mem:-?}MiB), cheapest=GPU$best_idx with ${best_mem}MiB used. Aborting." >&2
        exit 1
    fi
    mkdir -p ~/logs
    printf '%s\tjobid=%s\tnode=%s\trebind_from=%s(mem=%sMiB)\trebind_to=%s(mem=%sMiB)\treason=preflight_double_alloc\n' \
        "$(date -Is)" "${SLURM_JOB_ID:-?}" "$(hostname -s)" "${cur:-unset}" "${cur_mem:-?}" "$best_idx" "$best_mem" \
        >> ~/logs/slurm_gpu_collisions.log
    echo "PREFLIGHT: rebinding CUDA_VISIBLE_DEVICES ${cur:-unset} -> $best_idx (orig had ${cur_mem:-?}MiB used; new has ${best_mem}MiB)" >&2
    export CUDA_VISIBLE_DEVICES="$best_idx"
}

start_vllm() {
    local model="$1" log="$2"
    # Honor SLURM's cgroup GPU assignment (CUDA_VISIBLE_DEVICES is preset
    # by SLURM under sbatch/srun); only override for bare-shell use.
    local devs="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
    # vllm args mirror the comment block at the top of deepseek_prover.py.
    CUDA_VISIBLE_DEVICES="$devs" nohup "$VLLM" serve "$model" \
        --port "$PORT" \
        --tensor-parallel-size 1 \
        --dtype bfloat16 \
        --max-model-len 16384 \
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

preflight_rebind

vllm_log="$LOG_DIR/vllm_prover.log"
echo "=== starting vllm ($PROVER_MODEL) on GPU $GPU_ID port $PORT (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}) ==="
pid=$(start_vllm "$PROVER_MODEL" "$vllm_log")
trap 'stop_vllm '"$pid"' || true' EXIT
if ! wait_ready "$pid" "$vllm_log"; then
    stop_vllm "$pid" || true
    exit 1
fi
echo "=== vllm ready (pid=$pid) ==="

for name in "${NAMES[@]}"; do
    for p in 1.0 0.0; do
        run_log="$LOG_DIR/prove_${name}_p$(printf '%03d' $(awk -v x="$p" 'BEGIN{printf "%d", x*100+0.5}')).log"
        echo "=== [$name] run_prove.py --p $p  (log: $run_log) ==="
        # See run_phase2_brokenmath.sh comment: cap lean workers at 16 to
        # avoid system-wide ENFILE (Mathlib mmaps ~2k files per REPL).
        "$PY" scripts/run_prove.py \
            --formalizer "$name" \
            --p "$p" \
            --port "$PORT" \
            --concurrency 8 \
            --lean_concurrency 16 \
            2>&1 | tee "$run_log"
    done
done

echo "=== stopping vllm (pid=$pid) ==="
stop_vllm "$pid"
trap - EXIT

echo "=== all formalizers done ==="
