#!/bin/bash

set -eo pipefail
printf "\n       \\               ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ \n        \ji            ♥ Running AQTED Experiments ♥\n        /.(((          ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ \n       (,/\"(((__,--. \n           \  ) _( /{  \n           !|| \" :||    \n           !||   :||  \n           '''   '''  \n"


# Set variables
export INSTANCE_DIRECTORY="data/*"

echo "♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥"
echo "♥ Run Each Random Instance  ... ♥"
echo "♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥"

for file in $INSTANCE_DIRECTORY  
do    
    inst="${file##*/}"
    echo $inst
    singularity run --app run_experiment qaoa_vrp.img $inst
done