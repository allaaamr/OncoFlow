#!/bin/bash
#SBATCH --job-name=tage_diff_train
#SBATCH --output=logs/tagediff/output_MU_ConcatLast_CA_ours.%A_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH --time=12:00:00
#SBATCH --qos=cscc-gpu-qos

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /apps/local/anaconda3/conda_init.sh
conda activate wm   

    

python scripts/train.py \
--config configs/default.yaml \
--training_mode diffusion \
--resume /home/alaa.mohamed/OncoFlow+/ckpts/OncoFlow3D_latent_miu_fm_fixed/diffusion/ckpt_e0100.pt
