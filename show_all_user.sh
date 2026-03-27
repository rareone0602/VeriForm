#!/bin/bash

# 1. Get the list of PIDs (Process IDs) using the GPU
# We request the GPU Index, PID, and Used Memory
# idioms: We're separating the wheat from the chaff here.
info=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)

# Check if the string is empty (i.e., no processes running)
if [ -z "$info" ]; then
    echo "No active GPU processes found. The coast is clear!"
    exit 0
fi

# Print a header
printf "%-5s %-10s %-15s %-15s %s\n" "GPU" "PID" "USER" "MEM(MiB)" "COMMAND"
echo "------------------------------------------------------------------"

# 2. Iterate through each line of the nvidia-smi output
while IFS=, read -r gpu_uuid pid used_mem; do
    # Trim whitespace
    pid=$(echo "$pid" | xargs)
    used_mem=$(echo "$used_mem" | xargs)
    
    # 3. Use 'ps' to find the username associated with this PID
    # We suppress the header with 'h' and ask for user and command
    user_info=$(ps -p "$pid" -o user=,comm=)
    
    # Extract user and command from the ps output
    user=$(echo "$user_info" | awk '{print $1}')
    cmd=$(echo "$user_info" | awk '{$1=""; print $0}' | xargs)

    # Print the formatted row
    # If the process finished before we could check, ps returns empty, so we handle that.
    if [ -n "$user" ]; then
        printf "%-5s %-10s %-15s %-15s %s\n" "0" "$pid" "$user" "$used_mem" "$cmd"
    fi

done <<< "$info"