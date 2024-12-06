#!/bin/bash
#SBATCH -N 1            # number of nodes
#SBATCH -c 2            # number of cores
#SBATCH -G a100:1            # number of GPU
#SBATCH --mem=12G       # memory requested 
#SBATCH -t 0-4:00:00    # time in d-hh:mm:ss
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

module load mamba/latest

source activate GNN_PyTorch

protein=$1

python endpoints/MLP_5fold.py -p $protein 
