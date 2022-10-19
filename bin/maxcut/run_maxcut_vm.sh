#!/bin/bash
for i in {1..150}
do
   echo "Welcome this run:$i for MAXCUT VQE"
   singularity run --app run_vqe_maxcut vqe_maxcut.img
done