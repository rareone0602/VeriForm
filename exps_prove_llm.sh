#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

# Define your parameters here
formalizers=("goedel" "herald" "kimina" "stepfun")
efforts=("low" "medium" "high")
ports=(8002)
p="1.0"

# ------------------------------------------------------------------
# RUN TASKS CONCURRENTLY (STAGGERED START)
# ------------------------------------------------------------------

# Loop through each parameter
for port in "${ports[@]}"; do
  for effort in "${efforts[@]}"; do
    for formalizer in "${formalizers[@]}"; do
      
      # Define the log file name
      log_file="logs/${formalizer}_p${p}_port${port}_${effort}.log"

      echo "Starting: formalizer=${formalizer}, p=${p}, port=${port}, effort=${effort}"

      # Run the python script in the background
      python scripts/all_proof_llm_perturbed.py \
        --p "$p" \
        --formalizer "$formalizer" \
        --port "$port" \
        --effort "$effort" \
        >"$log_file" 2>&1 &

      # Wait 10 seconds before starting the next one
      sleep 10

    done
  done
done

# Wait for all background jobs to finish
wait

echo "All tasks completed."