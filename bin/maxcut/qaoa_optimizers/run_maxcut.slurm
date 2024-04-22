#!/bin/bash

#SBATCH --job-name=run-experiment
###SBATCH -p cloud

# For R need to run on single CPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# set your minimum acceptable walltime=hours:minutes:seconds
#SBATCH --time=3-24:00:00  

# Specify your email address to be notified of progress.
### SBATCH --mail-user=vkatial@student.unimelb.edu.au
### SBATCH --mail-type=ALL

module purge
module load GCCcore/11.3.0
module load Apptainer/1.1.8


# Load in environment modules
source ~/.bashrc
echo $1 $2 $3 $4

# The command to run the job, using parameters passed to the script
apptainer run \
    --env MLFLOW_TRACKING_USERNAME=$MLFLOW_TRACKING_USERNAME \
    --env MLFLOW_TRACKING_PASSWORD=$MLFLOW_TRACKING_PASSWORD \
    --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    --pwd /mnt \
    --bind $(pwd):/mnt \
    --app run_qaoa_maxcut_classical_optimisers haqc.sif $1 "$2" $3 $4
