#!/bin/bash

#SBATCH --job-name=maisi_vae_s1_brats
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=P2
#SBATCH --exclude=b31
#SBATCH --time=0-12:00:00
#SBATCH --mem=200GB
#SBATCH --signal=B:SIGUSR1@180
#SBATCH -o /shared/s1/lab06/wonyoung/maisi/logs_ex/%x-%j.txt
#SBATCH --open-mode=append

source /home/s1/wonyoungjang/.bashrc
source /home/s1/wonyoungjang/anaconda3/bin/activate
conda activate wdm

# --- 환경 변수 설정 (기존과 동일) ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=1

echo "BraTS - Starting MAISI VAE MULTI-NODE DDP training on 2 nodes (8 GPUs)..."
echo "MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT"

max_restarts=1000
scontext=$(scontrol show job ${SLURM_JOB_ID})
restarts=$(echo ${scontext} | grep -o 'Restarts=[0-9]*****' | cut -d= -f2)

function resubmit()
{
    if [[ $restarts -lt $max_restarts ]]; then
        scontrol requeue ${SLURM_JOB_ID}
        exit 0
    else
        echo "Your job is over the Maximum restarts limit"
        exit 1
    fi
}

trap 'resubmit' SIGUSR1

srun --cpu-bind=none,v --accel-bind=gn torchrun \
    --nproc_per_node=4 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    maisi_train_VAE_brats.py \
      --model_config_path configs/config_maisi3d-rflow_brats.json \
      --train_config_path configs/config_maisi_vae_train_brats_stage1.json \
      --run_name ${SLURM_JOB_NAME} \
      --cpus_per_task ${SLURM_CPUS_PER_TASK} \
      --resume &
wait
exit 0