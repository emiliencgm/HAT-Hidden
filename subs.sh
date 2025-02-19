#!/bin/bash
#SBATCH --job-name=predict_HAT_gpu
#SBATCH --partition=gpuq
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1 
##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=02:00:00 
#SBATCH --output=predict_HAT_gpu_%j.out
#SBATCH --error=predict_HAT_gpu_%j.out

# source activate base
source /home/echen/anaconda3/etc/profile.d/conda.sh
export CONDA_OVERRIDE_CUDA=11.3
conda activate predict_HAT

# Cleans out the modules loaded in interactive and inherited by default 
module purge

# Loading of modules
#module load ...

# Echo of launched commands
set -x

# Code execution
nvidia-smi -L
nvidia-smi
python -u main-emilien.py --dataset in_house
python -u main-emilien.py --dataset omega
python -u main-emilien.py --dataset omega_exp
python -u main-emilien.py --dataset hong
python -u main-emilien.py --dataset omega_bietti_hong
python -u main-emilien.py --dataset tantillo
python -u main-emilien.py --dataset rmechdb
# python -u reproducibility.py
nvidia-smi

