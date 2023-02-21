#!/bin/bash
for i in {1..150}
do
   echo "Welcome this run:$i for MAXCUT QAOA parameter concentration"
   singularity run --app run_qaoa_maxcut_params_conc vqe_maxcut.img
done