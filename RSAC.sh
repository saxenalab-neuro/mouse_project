#!/bin/sh
#SBATCH --job-name=mouse
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 
#SBATCH --mem=64gb
#SBATCH --time=144:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=mouse.out
pwd; hostname; date

export PATH=/home/lazzarijohn/.conda/envs/mouse/bin:$PATH

python /home/lazzarijohn/mouse_without_ddp/mouse_project/main.py --cuda

date
