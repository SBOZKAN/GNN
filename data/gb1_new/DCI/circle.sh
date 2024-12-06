#!/bin/bash

for i in $(seq 56)
do
rm -r $i
mkdir $i
cd $i
cp ../{2qmt.pdb,submit.sh} .
sbatch submit.sh $i
cd ../
done
