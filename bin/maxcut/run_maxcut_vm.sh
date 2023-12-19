#!/bin/bash

# Define the array of node sizes
node_sizes=(8)

# Define the array of n_layers
n_layers=(1)

# Define the array of graph types
graph_types=("Nearly Complete BiPartite" "Uniform Random" "Power Law Tree" "Watts-Strogatz small world" "3-Regular Graph" "4-Regular Graph" "Geometric")

# Create a log file
log_file="run_log.txt"

# Clear the log file at the start of the script
> "$log_file"

# Function to run a single job
run_job() {
    local size=$1
    local layer=$2
    local graph=$3
    local i=$4

    # Logging the parameters
    echo "Running iteration $i with size: $size, layer: $layer, graph type: $graph" >> "$log_file"

    # Check OS and run appropriate command
    if [[ $(uname) == 'Darwin' ]]; then
        # macOS specific command
        poetry run python run_maxcut_isa.py -T True -G "$graph" -n "$size" -l "$layer" >> "$log_file" 2>&1
    else
        # Ubuntu (Linux) specific command
        source ~/.bashrc
        apptainer run \
          --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
          --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
          --pwd /mnt \
          --bind $(pwd):/mnt \
          --app run_qaoa_maxcut_params_conc haqc.sif $size "$layer" $graph >> "$log_file" 2>&1
    fi
}

# Counter for jobs
job_count=0

# Outer loop for 100 iterations
for i in {1..100}; do
    # Loop through each combination of parameters
    for size in "${node_sizes[@]}"; do
        for layer in "${n_layers[@]}"; do
            for graph in "${graph_types[@]}"; do
                # Run the job in the background
                run_job $size $layer "$graph" $i &

                # Increment job counter
                ((job_count++))

                # Limit the number of concurrent jobs
                if ((job_count >= 10)); then
                    wait -n
                    ((job_count--))
                fi
            done
        done
    done
done

# Wait for all background jobs to finish
wait

echo "All jobs completed."