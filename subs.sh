#!/bin/bash
#SBATCH --job-name=predict_HAT_gpu
#SBATCH --partition=gpuq
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1 
##SBATCH --gres=gpu:1 #TODO why needs to be removed?
#SBATCH --cpus-per-task=10 # TODO add more CPUs
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=72:00:00 
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

# python -u main-emilien.py --dataset in_house --chk_path_hidden 1200
# python -u main-emilien.py --dataset omega
# python -u main-emilien.py --dataset omega_exp
# python -u main-emilien.py --dataset hong
# python -u main-emilien.py --dataset omega_bietti_hong
# python -u main-emilien.py --dataset tantillo
# python -u main-emilien.py --dataset rmechdb
# python -u reproducibility.py


for hidden_size in $(seq 100 100 2000); do
    # 在不同hidden size的M1下，按照1200-dim M1下最优的M2超参数，重跑M2
    # NOTE 若在不同M1下调整M2的超参数，参考作业脚本hyper_opt_subs.sh的注释掉的代码
    python -u main-emilien.py --dataset in_house --chk_path_hidden $hidden_size
    python -u main-emilien.py --dataset omega --chk_path_hidden $hidden_size
    python -u main-emilien.py --dataset omega_exp --chk_path_hidden $hidden_size
    python -u main-emilien.py --dataset hong --chk_path_hidden $hidden_size
    python -u main-emilien.py --dataset omega_bietti_hong --chk_path_hidden $hidden_size
    python -u main-emilien.py --dataset tantillo --chk_path_hidden $hidden_size
    python -u main-emilien.py --dataset rmechdb --chk_path_hidden $hidden_size
    # 在不同hidden size的M1下，按默认超参数，复现Javier的M2模型
    # TODO 复现tantillo上的最佳模型FFNN，需在所有程序运行结束后修改代码，见eproducibility_emilien.py的NOTE
    python -u reproducibility_emilien.py --hidden_size_M1 $hidden_size
done

nvidia-smi

