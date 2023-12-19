#!/bin/bash

set -eo pipefail
printf "\n       \\               ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ \n        \ji            ♥ Running VQE MAXCUT Experiments ♥\n        /.(((          ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ \n       (,/\"(((__,--. \n           \  ) _( /{  \n           !|| \" :||    \n           !||   :||  \n           '''   '''  \n"

# Define the array of node sizes
node_sizes=(12)

# Define the array of graph types
graph_types=("Nearly Complete BiPartite" "Uniform Random" "Power Law Tree" "Watts-Strogatz small world" "3-Regular Graph" "4-Regular Graph" "Geometric")

# Define the array of n_layers
n_layers=(3)

# Initialize counter
total_jobs=0

# Main loop (running 100 instances of each node size, graph type and layer)
for i in {1..300}; do
   for node_size in "${node_sizes[@]}"; do
      for graph_type in "${graph_types[@]}"; do
         for layer in "${n_layers[@]}"; do
               # Set NodeMemory based on node_size and aad
               if [ "$node_size" -lt 10 ]; then
                  NodeMemory="16GB"
               else
                  NodeMemory="40GB"
               fi

               echo "Allocating node $NodeMemory memory for run number: $i, Node Size: $node_size, Graph Type: $graph_type, Layer: $layer"
               log_file="logs/qaoa_maxcut_node_${node_size}_graph_${graph_type}_layer_${layer}_run_$i.log"
               echo "Results will be logged into $log_file"
               sbatch --chdir=$(pwd) --mem $NodeMemory --output="$log_file" bin/maxcut/run_maxcut.slurm $node_size "$graph_type" $layer

               # Increment the counter
               total_jobs=$((total_jobs+1))
               echo "Job number $total_jobs submitted"
         done
      done
   done
done

# Display the total number of jobs submitted
echo "Total number of jobs submitted: $total_jobs"
