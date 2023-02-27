#!/bin/bash
for i in {1..150}
do
   echo "Welcome this run:$i for MAXCUT QAOA number of Layers"
   singularity run --app run_qaoa_maxcut_layers vqe_maxcut.img
done