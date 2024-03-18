#!/bin/bash

# Exit immediately if a command exits with a non-zero status, and print each command that is executed
set -eo pipefail

# ASCII Art Header
printf "\n       \\               ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ \n        \ji            ♥ Running MAXCUT Experiments ♥\n        /.(((          ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ \n       (,/\"(((__,--. \n           \  ) _( /{  \n           !|| \" :||    \n           !||   :||  \n           '''   '''  \n"

# Define experiment parameters
node_sizes=(6 8 10 12 14)
graph_types=("Nearly Complete BiPartite" "Uniform Random" "Power Law Tree" "Watts-Strogatz small world" "3-Regular Graph" "4-Regular Graph" "Geometric")
n_layers=(20)
max_feval=20000
total_jobs=0

# Main experiment loop
for node_size in "${node_sizes[@]}"; do
   for i in {1..300}; do
        for graph_type in "${graph_types[@]}"; do
            for layer in "${n_layers[@]}"; do
                # Adjust NodeMemory based on node_size
                if [ "$node_size" -lt 13 ]; then
                    NodeMemory="16GB"
                else
                    NodeMemory="40GB"
                fi

                # Logging and job submission
                echo "Allocating node $NodeMemory memory for run number: $i, Node Size: $node_size, Graph Type: $graph_type, Layer: $layer"
                log_file="logs/qaoa_maxcut_node_${node_size}_graph_${graph_type}_layer_${layer}_run_$i.log"
                echo "Results will be logged into $log_file"
                sbatch --chdir=$(pwd) --mem=$NodeMemory --output="$log_file" bin/maxcut/qaoa_layers_vanilla/run_maxcut.slurm "$node_size" "$graph_type" "$layer" "$max_feval"

                # Increment job counter
                total_jobs=$((total_jobs + 1))
                echo "Job number $total_jobs submitted"
            done
        done
    done
done

# Final job count
echo "Total number of jobs submitted: $total_jobs"
