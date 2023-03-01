#!/bin/bash
for i in {1..150}
do
   echo "Welcome this run:$i for MAXCUT QAOA Initialisation Methods"
   singularity run --app run_qaoa_maxcut_initialisation_techniques vqe_maxcut.img
done