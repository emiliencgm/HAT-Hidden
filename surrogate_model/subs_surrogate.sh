#!/bin/bash
#SBATCH --job-name=retrain_surrogate
#SBATCH --partition=gpuq
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1 
##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=30
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=168:00:00 
#SBATCH --output=retrain_surrogate_%j.out
#SBATCH --error=retrain_surrogate_%j.out

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

# Code execution, 
# During the original hyperparams search: 300 to 1800 for hidden_size, while batch_size and epochs are not tuned.
for hidden_size in $(seq 100 100 2000); do
    python -u train_emilien.py --hidden_size $hidden_size --batch_size 50 --epochs 100
done

