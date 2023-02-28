#!/bin/bash
for i in {1..150}
do
   echo "Welcome this run:$i for MAXCUT"
   singularity run --app run_qaoa_maxcut_optimizers vqe_maxcut.img
done