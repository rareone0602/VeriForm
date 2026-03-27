#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

# Format: "{formalizer}:{p}:{port}"
tasks=(
  "goedel:0.0:8001"
  "goedel:1.0:8001"
  "goedel:0.0:8002"
  "goedel:1.0:8002"
  "goedel:0.0:8001"
  "goedel:1.0:8001"
  "goedel:0.0:8002"
  "goedel:1.0:8002"
  "goedel:0.0:8001"
  "goedel:1.0:8001"
  "goedel:0.0:8002"
  "goedel:1.0:8002"
)

# ------------------------------------------------------------------
# RUN TASKS CONCURRENTLY (STAGGERED START)
# ------------------------------------------------------------------

for task in "${tasks[@]}"; do
  # Split the string into formalizer, p, and port
  IFS=":" read -r formalizer p port <<< "$task"

  log_file="logs/${formalizer}_p${p}_port${port}.log"

  echo "Starting: formalizer=${formalizer}, p=${p}, port=${port}"

  # Assuming you want to pass 'p' to the python script directly.
  # Note: I removed CUDA_VISIBLE_DEVICES here because 'p' (e.g., 0.0, 1.0)
  # does not look like a GPU ID. If you need to assign GPUs, you might
  # need to add a 4th parameter or handle it differently.
  python scripts/all_proof.py \
    --p "$p" \
    --formalizer "$formalizer" \
    --port "$port" \
    >"$log_file" 2>&1 &

  sleep 10
done

# Wait for all background jobs to finish
wait

echo "All tasks completed."