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
#SBATCH --output=RSAC.out
pwd; hostname; date

export PATH=/blue/shreya.saxena/share/mujoco/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64:/home/malmani/.mujoco/mujoco200/bin

python /home/malmani/blue_dir/malmani/Monkey_Speed_10/Monkey_RSAC_1300/RSAC5/SAC/main.py --cuda

date
