#!/bin/bash

## Only 2 GPU total nodes in cluster. Can use only 1. So no use specifying -n option. Because it splits tasks over multiple nodes. Hence used -c which ensures multithreading on same machine.
#SBATCH -p pfen3
#SBATCH --gres=gpu:1

#SBATCH -c 1 #24 cores total on 1 machine. So use 6 cores for 1 task.
#SBATCH --mem=50G

#SBATCH --job-name differential
#SBATCH --output differential-log-%J.txt

python3 cnn.py --xtrain trainData/trainInput.npy --ytrain trainData/trainLabels.npy --xvalid trainData/validationInput.npy --yvalid trainData/validationLabels.npy --model-out output-newDrop --pool-dropout-rate 0.5
#time Rscript newcode.R
#time ./fileclean.sh
