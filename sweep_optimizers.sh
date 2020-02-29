#!/bin/bash
# https://s3-us-west-2.amazonaws.com/imss-hpc/index.html to generate script
#SBATCH --time=48:00:00   # walltime
#SBATCH --gres=gpu:1
#SBATCH --gid=andersonlab

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 --gpus 1 python test_run.py --output_path ./test_output_SGD --optimizer SGD --use_gpu True --num_epochs 100 --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/  &
srun --exclusive -N 1 -n 1 --gpus 1 python test_run.py --output_path ./test_output_RMSprop --optimizer RMSprop --use_gpu True --num_epochs 100 --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/  &
srun --exclusive -N 1 -n 1 --gpus 1 python test_run.py --output_path ./test_output_Adam --optimizer Adam --use_gpu True --num_epochs 100 --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/  &
srun --exclusive -N 1 -n 1 --gpus 1 python test_run.py --output_path ./test_output_LBFGS --optimizer LBFGS --use_gpu True --num_epochs 100 --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/
wait