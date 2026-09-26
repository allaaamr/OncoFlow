#!/bin/bash
#SBATCH --job-name=oncoflow_latent_fm
#SBATCH --output=logs/horizon/MU_SoftDil_STLoss_HighRes.%A_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH --time=12:00:00
#SBATCH --qos=cscc-gpu-qos

source /apps/local/anaconda3/conda_init.sh
cd /home/alaa.mohamed/OncoFlow+
conda activate wm   

python scripts/test3.py --config configs/cfb_voxel_HD_ST.yaml \
    --checkpoint ./new_ckpts/CFB/HorizonDecay_HighRes/flow_matching/voxel/ckpt_e0720.pt\
    --patient-id PatientID_109 \
    --steps 50 --use-ema \
    --save-dir ./evaluations/MU/RegAware/HorizonDecay_HighRes_720/PatientID_109

python scripts/test3.py --config configs/cfb_voxel_HD_ST.yaml \
    --checkpoint ./new_ckpts/CFB/HorizonDecay_HighRes/flow_matching/voxel/ckpt_e0520.pt\
    --patient-id PatientID_109 \
    --steps 50 --use-ema \
    --save-dir ./evaluations/MU/RegAware/HorizonDecay_HighRes_520/PatientID_109

python scripts/test3.py --config configs/cfb_voxel_HD_ST.yaml \
    --checkpoint ./new_ckpts/CFB/HorizonDecay_HighRes/flow_matching/voxel/ckpt_e0420.pt\
    --patient-id PatientID_109 \
    --steps 50 --use-ema \
    --save-dir ./evaluations/MU/RegAware/HorizonDecay_HighRes_420/PatientID_109

python scripts/test3.py --config configs/cfb_voxel_HD_ST.yaml \
    --checkpoint ./new_ckpts/CFB/HorizonDecay_HighRes/flow_matching/voxel/ckpt_e0720.pt\
    --patient-id PatientID_199 \
    --steps 50 --use-ema \
    --save-dir ./evaluations/MU/RegAware/HorizonDecay_HighRes_720/PatientID_199

    python scripts/test3.py --config configs/cfb_voxel_HD_ST.yaml \
    --checkpoint ./new_ckpts/CFB/HorizonDecay_HighRes/flow_matching/voxel/ckpt_e0720.pt\
    --patient-id PatientID_065 \
    --steps 50 --use-ema \
    --save-dir ./evaluations/MU/RegAware/HorizonDecay_HighRes_720/PatientID_065
