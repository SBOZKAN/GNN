#!/bin/bash
#SBATCH -N 1            # number of nodes
#SBATCH -c 2            # number of cores
#SBATCH -G a100:1            # number of GPU
#SBATCH --mem=12G       # memory requested 
#SBATCH -t 0-4:00:00    # time in d-hh:mm:ss
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --export=NONE   # Purge the job-submitting shell environment

res=$1
#module load python/3.7.1
for res in $(seq 56)
do
mkdir $res
cd $res
cp ../wt.pdb .
python ~/Scripts/dualDFI_v1_1/dualDFI.py --pdb wt.pdb --fdfi A$res
cd ../
done

