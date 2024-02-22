#!/bin/bash
for i in {1..5}
do
   echo "Welcome this run:$i for MAXCUT QAOA number of Layers"
   apptainer run \
    --env MLFLOW_TRACKING_USERNAME=$MLFLOW_TRACKING_USERNAME \
    --env MLFLOW_TRACKING_PASSWORD=$MLFLOW_TRACKING_PASSWORD \
    --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    --pwd /mnt \
    --bind $(pwd):/mnt \
    --app run_qaoa_maxcut_n_layers haqc.sif 12 "Nearly Complete BiPartite" 3 1000
done