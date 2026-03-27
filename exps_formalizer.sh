#!/bin/bash

# --- Configuration Arguments ---
PROBABILITY="1.0"
# NUM_RUNS=16 means 16 high + 16 low = 32 concurrent processes
NUM_RUNS=32
SLEEP_TIME=5
PORT=8002
# -------------------------------

# 1. Define an associative array mapping names to HuggingFace paths
declare -A MODELS=(
    ["stepfun"]="stepfun-ai/StepFun-Formalizer-7B"
    ["herald"]="FrenzyMath/Herald_translator"
    ["kimina"]="AI-MO/Kimina-Autoformalizer-7B"
    ["goedel"]="Goedel-LM/Goedel-Formalizer-V2-8B"
)

# Ordered list of formalizers to loop through
FORMALIZERS=("goedel")

# Helper function to poll the server until it's ready
wait_for_server() {
    echo "Waiting for vLLM server on port $PORT to be ready..."
    # The loop continues until curl gets a successful response
    while ! curl -s "http://localhost:$PORT/v1/models" > /dev/null; do
        sleep 5
    done
    echo "Server is up and running!"
}

# -------------------------------
# Main Loop
# -------------------------------
for FORMALIZER in "${FORMALIZERS[@]}"; do
    MODEL_PATH="${MODELS[$FORMALIZER]}"
    
    echo "=========================================================="
    echo "Starting pipeline for: $FORMALIZER"
    echo "Model path: $MODEL_PATH"
    echo "=========================================================="

    # 1. Spin up the vLLM server in the background using '&'
    CUDA_VISIBLE_DEVICES=2 vllm serve "$MODEL_PATH" \
        --port $PORT \
        --tensor-parallel-size 1 \
        --dtype bfloat16 \
        --trust-remote-code \
        --enable-prefix-caching & 
        
    # $! captures the PID of the last background process
    VLLM_PID=$! 
    echo "vLLM started behind the scenes with PID: $VLLM_PID"
    
    # 2. Wait until the server is actually ready to accept requests
    wait_for_server
    
    echo "Launching 32 concurrent Python processes..."

    # 3. Run the python clients
    for ((i=1; i<=NUM_RUNS; i++)); do
        # Run high effort in the background
        python scripts/all_formalizer_llm_perturbed.py \
            --p "$PROBABILITY" \
            --formalizer "$FORMALIZER" \
            --effort "high" &
            
        sleep "$SLEEP_TIME"
        
        # Run low effort in the background
        python scripts/all_formalizer_llm_perturbed.py \
            --p "$PROBABILITY" \
            --formalizer "$FORMALIZER" \
            --effort "low" &
        
        sleep "$SLEEP_TIME"
    done

    # 'wait' pauses the script until ALL background python scripts wrap up
    echo "All scripts launched. Waiting for them to finish..."
    wait
    echo "Clients finished for $FORMALIZER."

    # 4. Terminate the vLLM server to free the GPU
    echo "Shutting down vLLM server (PID: $VLLM_PID)..."
    kill $VLLM_PID
    
    # Wait for the OS to reclaim the GPU memory before starting the next loop
    sleep 10 
done

echo "=========================================================="
echo "All formalizers have been processed successfully."