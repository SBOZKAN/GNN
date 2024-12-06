#!/bin/bash

for i in $(seq 26 290)
do
rm -r $i
mkdir $i
cd $i
cp ../{wt.pdb,submit2.sh} .
sbatch submit2.sh $i
cd ../
done
