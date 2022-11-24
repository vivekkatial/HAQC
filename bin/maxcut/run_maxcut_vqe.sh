#!/bin/bash
for i in {1..150}
do
   echo "Welcome this run:$i for MAXCUT VQE"
   python maxcut_vqe.py -T True
done