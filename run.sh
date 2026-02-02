#!/bin/bash
#SBATCH --job-name=picogpt
#SBATCH --partition=gpu
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

source /gpfs-flash/junlab/liuyifan24/miniconda3/etc/profile.d/conda.sh
conda activate nanogpt
python -u train.py