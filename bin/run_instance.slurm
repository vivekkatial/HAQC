#!/bin/bash

#SBATCH --job-name=run-experiment
###SBATCH -p cloud

# For R need to run on single CPU
#SBATCH --ntasks=1

# set your minimum acceptable walltime=hours:minutes:seconds
#SBATCH -t 00-23:59:59

# Specify your email address to be notified of progress.
### SBATCH --mail-user=vkatial@student.unimelb.edu.au
### SBATCH --mail-type=ALL

# Load in Singularity Model
module load Singularity/3.5.3-spartan_gcc-6.2.0

command_input=$1

echo "Running Experiment on log file $command_input"

# The command to actually run the job
singularity run --app run_experiment qaoa_vrp.img $command_input