#!/bin/bash
#
#SBATCH --time=48:00:00   # walltime
#SBATCH --gres=gpu:3
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=3
#SBATCH -J "run_GPU_training"   # job name
#SBATCH --mail-user=mlevine@caltech.edu   # email address

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python test_run.py --use_gpu True --lr 0.1 --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/ --output_path ./gpu_test_output_lr0.1 &
srun --exclusive -N 1 -n 1 python test_run.py --use_gpu True --lr 0.05 --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/ --output_path ./gpu_test_output_lr0.01 &
srun --exclusive -N 1 -n 1 python test_run.py --use_gpu True --lr 0.2 --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/ --output_path ./gpu_test_output_lr0.2
wait