#!/bin/bash

## Only 2 GPU total nodes in cluster. Can use only 1. So no use specifying -n option. Because it splits tasks over multiple nodes. Hence used -c which ensures multithreading on same machine.
#SBATCH -p pfen3
#SBATCH --gres=gpu:1

#SBATCH -c 6 #24 cores total on 1 machine. So use 6 cores for 1 task.
#SBATCH --mem=60G

#SBATCH --job-name differential
#SBATCH --output differential-log-%J.txt

time Rscript newcode.R
#time ./fileclean.sh
