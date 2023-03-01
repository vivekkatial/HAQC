#!/bin/bash

#SBATCH --job-name=run-experiment
###SBATCH -p cloud

# For R need to run on single CPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# set your minimum acceptable walltime=hours:minutes:seconds
#SBATCH -t 00-23:59:59

# Specify your email address to be notified of progress.
### SBATCH --mail-user=vkatial@student.unimelb.edu.au
### SBATCH --mail-type=ALL

# Load in Singularity Model
module load singularity/3.5.3

# The command to actually run the job
singularity run --app run_qaoa_maxcut_initialisation_techniques vqe_maxcut.img