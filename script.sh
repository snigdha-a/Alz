#!/bin/bash

## Only 2 GPU total nodes in cluster. Can use only 1. So no use specifying -n option. Because it splits tasks over multiple nodes. Hence used -c which ensures multithreading on same machine.
#SBATCH -p pfen3
#SBATCH --gres=gpu:1

#SBATCH -c 1 #24 cores total on 1 machine. So use 6 cores for 1 task.
#SBATCH --mem=25G

#SBATCH --job-name negRscript
#SBATCH --output negRscript-log-%J.txt

python signal_extraction.py
#Rscript nullSet.R label_1
#time ./fileclean.sh
