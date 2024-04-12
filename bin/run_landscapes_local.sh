#!/bin/bash

ls

# Create array of Graph Types
graph_types=( 
    # "Uniform Random" 
    "Power Law Tree" 
    "Watts-Strogatz small world" 
    "Geometric" 
    "Nearly Complete BiPartite" 
    "3-Regular Graph" 
    # "4-Regular Graph" 
    "3-Regular (no triangle)" 
    "2-Regular (ring)"
    )

# Run landscapes locally
for graph_type in "${graph_types[@]}"
do
    echo "Running landscapes for ${graph_type} graph"
    poetry run python parameter-fixing-simple-instances.py --graph_type "${graph_type}"
done
