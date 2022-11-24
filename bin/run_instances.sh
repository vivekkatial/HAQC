#!/bin/bash

set -eo pipefail
printf "\n       \\               ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ \n        \ji            ♥ Running AQTED Experiments ♥\n        /.(((          ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ \n       (,/\"(((__,--. \n           \  ) _( /{  \n           !|| \" :||    \n           !||   :||  \n           '''   '''  \n"


# Set variables
export INSTANCE_DIRECTORY="data/*"
export NodeMemory=40GB

# Build random instances
# echo "♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥"
# echo "♥ Building Random Instances ... ♥"
# echo "♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥"

# sbatch --output=$log_file bin/run-experiments.slurm $exp_run_params


echo "♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥"
echo "♥ Run Each Random Instance  ... ♥"
echo "♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥"

for file in $INSTANCE_DIRECTORY  
do    
    inst="${file##*/}"
    echo "Allocating node $NodeMemory memory for instance: $inst"
    # Run experiment as an instance of the singularity container
    log_file="logs/$inst.log"
    echo $log_file
    sbatch --mem $NodeMemory --output=$log_file bin/run_instance.slurm $inst
done