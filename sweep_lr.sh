#!/bin/bash
# https://s3-us-west-2.amazonaws.com/imss-hpc/index.html to generate script
#SBATCH --time=48:00:00   # walltime
#SBATCH --gres=gpu:1
#SBATCH --gid=andersonlab

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python test_run.py --output_path ./test_output_1000epochs_lr0.2 --lr 0.2 --use_gpu True --num_epochs 1000 --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/ &
srun --exclusive -N 1 -n 1 python test_run.py --output_path ./test_output_1000epochs_lr0.05 --lr 0.05 --use_gpu True --num_epochs 1000 --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/  &
srun --exclusive -N 1 -n 1 python test_run.py --output_path ./test_output_1000epochs_lr0.1 --lr 0.1 --use_gpu True --num_epochs 1000 --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/  &
srun --exclusive -N 1 -n 1 python test_run.py --output_path ./test_output_1000epochs_lr0.5 --lr 0.5 --use_gpu True --num_epochs 1000 --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/
wait