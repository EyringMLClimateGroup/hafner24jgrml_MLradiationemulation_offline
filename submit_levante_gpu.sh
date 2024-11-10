#!/bin/sh
#SBATCH --account=bdXXXX

#SBATCH --job-name=training
#SBATCH --partition=gpu
#SBTACH --gpus=4
#SBATCH --mem=0
#SBATCH --constraint=a100_80
#SBATCH --exclusive
#SBATCH --time=04:00:00

source ~/.bashrc
conda activate hafner_ml_rad

python train_coarse_levante_profile.py -c $1 
wait
python eval_coarse_levante_profile.py -c $1 