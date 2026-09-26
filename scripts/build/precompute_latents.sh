#!/bin/bash
#SBATCH --job-name=precompute_latents
#SBATCH --output=logs/precompute_latents.%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH --time=04:00:00
#SBATCH --qos=cscc-gpu-qos

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /apps/local/anaconda3/conda_init.sh
conda activate wm

python scripts/precompute_latents.py --config configs/default.yaml --force
