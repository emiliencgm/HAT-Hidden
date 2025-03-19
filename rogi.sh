#!/bin/bash
#SBATCH --job-name=rogi_xd_hidden_and_desc
#SBATCH --partition=gpuq
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1 
##SBATCH --gres=gpu:1 #TODO why needs to be removed?
#SBATCH --cpus-per-task=10 # TODO add more CPUs
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=300:00:00 
#SBATCH --output=rogi_xd_hidden_and_desc_%j.out
#SBATCH --error=rogi_xd_hidden_and_desc_%j.out

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
# 不建议同时运行多个作业，这会导致log文件的混乱！！！
for hidden_size in $(seq 100 100 2000); do
    python -u rogi_.py --dataset in_house --chk_path_hidden $hidden_size
    python -u rogi_.py --dataset omega --chk_path_hidden $hidden_size
    python -u rogi_.py --dataset omega_exp --chk_path_hidden $hidden_size
    python -u rogi_.py --dataset hong --chk_path_hidden $hidden_size
    python -u rogi_.py --dataset omega_bietti_hong --chk_path_hidden $hidden_size
    python -u rogi_.py --dataset tantillo --chk_path_hidden $hidden_size
    python -u rogi_.py --dataset rmechdb --chk_path_hidden $hidden_size
done


nvidia-smi

