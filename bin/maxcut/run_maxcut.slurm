#!/bin/bash

#SBATCH --job-name=run-experiment
###SBATCH -p cloud

# For R need to run on single CPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# set your minimum acceptable walltime=hours:minutes:seconds
#SBATCH -t 07-23:59:59

# Specify your email address to be notified of progress.
### SBATCH --mail-user=vkatial@student.unimelb.edu.au
### SBATCH --mail-type=ALL

# Load in environment modules
source ~/.bashrc

# The command to run the job, using parameters passed to the script
apptainer run \
    --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    --app run_qaoa_maxcut_params_conc haqc.sif $1 $2 $3
