#!/bin/bash

# Set bash script to fail at first error, and echo all commands
set -eo pipefail

# Set path for instances exist
CUSTOM_INSTANCE_PATH="data"

# Define the array of node sizes
node_sizes=(20)

# Define the array of n_layers
n_layers=(15)

max_feval=20000

# Initialize counter
total_jobs=0

# Loop through all .pkl files in the directory
for instance in $CUSTOM_INSTANCE_PATH/*.pkl; do
    for node_size in "${node_sizes[@]}"; do
        for layer in "${n_layers[@]}"; do
            # Set NodeMemory based on node_size
            if [ "$node_size" -lt 15 ]; then
                NodeMemory="16GB"
            else
                NodeMemory="40GB"
            fi

            echo "Allocating node $NodeMemory memory for instance: $instance, Node Size: $node_size, Layer: $layer"
            log_file="logs/qaoa_${instance}_node_${node_size}_layer_${layer}.log"
            echo "Results will be logged into $log_file"

            # Submit the job to Slurm
            sbatch --chdir=$(pwd) --mem $NodeMemory --output="$log_file" bin/maxcut/qaoa_layers_vanilla/run_maxcut.slurm $node_size "$instance" $layer $max_feval

            # Increment the counter
            total_jobs=$((total_jobs+1))
            echo "Job number $total_jobs submitted for instance $instance"
        done
    done
done

# Display the total number of jobs submitted
echo "Total number of jobs submitted: $total_jobs"
