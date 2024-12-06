#!/bin/bash                                                     
                                                                                
#SBATCH -N 1                                                                    
#SBATCH -n 1                                                                   
#SBATCH -p physicsgpu1,physicsgpu2                                                          
#SBATCH -q physicsgpu1                                                          
#SBATCH --gres=gpu:1                                                            
#SBATCH -t 1-00:00:00                                                               
#SBATCH -o slurm.%N.%j.out                                                      
#SBATCH -e slurm.%N.%j.err 




res=$1

module load python/3.7.1
python ~/Scripts/dualDFI_v1_1/dualDFI.py --pdb wt.pdb --fdfi A$res
