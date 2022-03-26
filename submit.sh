#!/bin/bash
#SBATCH --job-name=edward-pigan

#SBATCH --partition=rtx6000

#SBATCH --gres=gpu:2

#SBATCH --qos=normal

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=8G

#SBATCH --output=slurm-%j.out

#SBATCH --error=slurm-%j.err

# Create Checkpoint Directory
mkdir -p /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYEDPURGE/checkpoint/
# Local symbolic link
ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYEDPURGE $PWD/checkpoint

# prepare your environment here
# module load pytorch1.7.1-cuda11.0-python3.6
module purge
source /h/edwardl/pigan/pigan_env/bin/activate

# put your command here
# python train.py
CUDA_VISIBLE_DEVICES=0,1 CUDA_HOME=/pkgs/cuda-10.2/ python3 train.py --curriculum CARLA --output_dir /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYEDPURGE --checkpoint_dir /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYEDPURGE/checkpoint --n_epochs 3000 --sample_interval 1000 --model_save_interval 5000 --eval_freq 5000

# copy over checkpoint files
mkdir -p /h/edwardl/pigan/output/${SLURM_JOB_ID}/
cp -r /checkpoint/${USER}/${SLURM_JOB_ID}/* /h/edwardl/pigan/output/${SLURM_JOB_ID}/