#!/bin/sh
#SBATCH --job-name=MS_10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 
#SBATCH --mem=64gb
#SBATCH --time=96:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:geforce:1
#SBATCH --output=RSAC_muscle.out
pwd; hostname; date

export PATH=/home/lazzarijohn/.conda/envs/mouse/bin:$PATH

python /home/lazzarijohn/mouse_project/main.py --cuda

date
