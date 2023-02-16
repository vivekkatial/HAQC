#!/bin/bash

set -eo pipefail
printf "\n       \\               ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ \n        \ji            ♥ Running VQE MAXCUT Experiments ♥\n        /.(((          ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ \n       (,/\"(((__,--. \n           \  ) _( /{  \n           !|| \" :||    \n           !||   :||  \n           '''   '''  \n"


# Set variables
export NodeMemory=40GB

# Build random instances
# echo "♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥"
# echo "♥ Building Random Instances ... ♥"
# echo "♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥"

# sbatch --output=$log_file bin/run-experiments.slurm $exp_run_params


# echo "♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥"
# echo "♥ Run Each Random Instance  ... ♥"
# echo "♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥"


for i in {1..1000}
do
   echo "Allocating node $NodeMemory memory for run number: $i"
   log_file="logs/qaoa_param_conc_maxcut_run_all_instance_$i.log"
   echo "Results will be logged into $log_file"
   sbatch --mem $NodeMemory --output=$log_file bin/maxcut/qaoa_param_conc/run_maxcut.slurm
   # singularity run --app run_vqe_maxcut vqe_maxcut.img
done