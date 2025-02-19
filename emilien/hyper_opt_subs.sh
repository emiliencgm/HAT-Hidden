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

# 激活 Conda 环境
source /home/echen/anaconda3/etc/profile.d/conda.sh
export CONDA_OVERRIDE_CUDA=11.3
conda activate predict_HAT

# 清理默认加载的模块
module purge

# 打印执行的命令
set -x

# 获取 GPU 信息
nvidia-smi -L
nvidia-smi

# 定义超参数搜索范围
layers=(0 1 2)
hidden_sizes=(128 192 256 384 512 1024 1500)
# dropouts=(0.0 0.1)
dropouts=(0.0)
# lrs=(0.005 0.01 0.02 0.0277 0.035)
lrs=(0.0277)

# 运行任务的计数器
count=0
max_parallel_jobs=3  # 最多并行 3 个任务
pids=()  # 存储任务的 PID

for layer in "${layers[@]}"; do
    for hidden_size in "${hidden_sizes[@]}"; do
        for dropout in "${dropouts[@]}"; do
            for lr in "${lrs[@]}"; do
                echo "Running with layers=$layer, hidden_size=$hidden_size, dropout=$dropout, lr=$lr"
                
                srun --ntasks=1 --cpus-per-task=10 python -u /home/echen/code/energy_predictor_HAT/main-emilien.py \
                    --layers "$layer" --hidden_size "$hidden_size" --dropout "$dropout" --lr "$lr" &

                pids+=($!)  # 记录任务的 PID
                ((count++))

                if [[ $count -ge $max_parallel_jobs ]]; then
                    wait -n  # 等待最早完成的任务
                    count=$((count - 1))  # 释放一个任务计数
                fi
            done
        done
    done
done

# 等待剩余的任务完成
wait

# 运行完成后再次查看 GPU 使用情况
nvidia-smi
