#!/bin/bash
#SBATCH --job-name=gpu_nvidia_smi
#SBATCH --partition=gpuq
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1 
##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=00:05:00 
#SBATCH --output=gpu_nvidia_smi_%j.out
#SBATCH --error=gpu_nvidia_smi_%j.out

# Cleans out the modules loaded in interactive and inherited by default 
module purge

# Loading of modules
#module load ...

# Echo of launched commands
set -x

# Code execution
nvidia-smi -L
nvidia-smi -l 10

