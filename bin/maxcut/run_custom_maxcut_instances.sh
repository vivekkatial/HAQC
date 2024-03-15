#!/bin/bash

# Set bash script to fail at first error, and echo all commands
set -eo pipefail

# Define the directory where the .pkl files will be downloaded
BASE_DIR="instances_final"
REPO_URL="https://github.com/vivekkatial/instance-generators.git"

download_instances() {
    if [ ! -d "$BASE_DIR" ]; then
        echo "Downloading instances from GitHub..."
        git clone --depth 1 --filter=blob:none --sparse $REPO_URL
        cd instance-generators
        git sparse-checkout add $BASE_DIR
    else
        echo "Instances already downloaded."
    fi
}

# Download the instances to the local directory
download_instances

# Define the array of node sizes
node_sizes=(12)

# Define the array of n_layers
n_layers=(3)

# Initialize counter
total_jobs=0

# Loop through all .pkl files in the directory
for instance in instance-generators/*.pkl; do
    for node_size in "${node_sizes[@]}"; do
        for layer in "${n_layers[@]}"; do
            # Set NodeMemory based on node_size
            if [ "$node_size" -lt 15 ]; then
                NodeMemory="16GB"
            else
                NodeMemory="40GB"
            fi

            echo "Allocating node $NodeMemory memory for instance: $instance, Node Size: $node_size, Layer: $layer"
            log_file="logs/qaoa_${graph_type}_node_${node_size}_layer_${layer}.log"
            echo "Results will be logged into $log_file"

            # Submit the job to Slurm
            # sbatch --chdir=$(pwd) --mem $NodeMemory --output="$log_file" ../bin/maxcut/run_maxcut.slurm $node_size "$graph_type" $layer

            # Increment the counter
            total_jobs=$((total_jobs+1))
            echo "Job number $total_jobs submitted for instance $instance"
        done
    done
done

# Display the total number of jobs submitted
echo "Total number of jobs submitted: $total_jobs"
