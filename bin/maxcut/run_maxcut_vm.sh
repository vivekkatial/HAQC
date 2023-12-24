#!/bin/bash

# Save the current working directory
working_dir=$(pwd)

# Define the array of node sizes
node_sizes=(8)

# Define the array of n_layers
n_layers=(1 2 3 4 5 6 7 8 9 10)

# Define the array of graph types
graph_types=("Nearly Complete BiPartite" "Uniform Random" "Power Law Tree" "Watts-Strogatz small world" "3-Regular Graph" "4-Regular Graph" "Geometric")

# Log file setup
log_file="$working_dir/run_log.txt"
> "$log_file"

# Function to run a single job
run_job() {
    local size=$1
    local layer=$2
    local graph=$3
    local iteration=$4

    # Log the parameters
    echo "Running iteration $iteration with size: $size, layer: $layer, graph type: $graph" >> "$log_file"

    # Check the OS and run appropriate command with lower priority
    if [[ $(uname) == 'Darwin' ]]; then
        # macOS specific command
        nice -n 10 poetry run python run_maxcut_isa.py -T True -G "$graph" -n "$size" -l "$layer" >> "$log_file" 2>&1 &
    else
        # Linux (Ubuntu) specific command
        nice -n 10 source ~/.bashrc
        nice -n 10 apptainer run \
          --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
          --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
          --pwd /mnt \
          --bind $working_dir:/mnt \
          --app run_qaoa_maxcut_params_conc haqc.sif $size "$layer" $graph >> "$log_file" 2>&1 &
    fi
}

# Fixed number of parallel jobs
max_jobs=4

# Main processing loop
for iteration in {1..50}; do
    for size in "${node_sizes[@]}"; do
        for layer in "${n_layers[@]}"; do
            for graph in "${graph_types[@]}"; do
                run_job $size $layer "$graph" $iteration
                while (( $(jobs -p | wc -l) >= max_jobs )); do
                    sleep 1
                done
            done
        done
    done
done

# Wait for all jobs to complete
wait

echo "All jobs completed."
