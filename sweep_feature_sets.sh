#!/bin/bash
# https://s3-us-west-2.amazonaws.com/imss-hpc/index.html to generate script
#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=10G   # memory per CPU core
#SBATCH -c 1 # 1 core per task

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python test_run.py --output_path ./test_output_featurestyle_all --feature_style 'all' --lr 0.1 --use_gpu False --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/  &
srun --exclusive -N 1 -n 1 python test_run.py --output_path ./test_output_featurestyle_keypoints_only --feature_style 'keypoints_only' --lr 0.1 --use_gpu False --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/  &
srun --exclusive -N 1 -n 1 python test_run.py --output_path ./test_output_featurestyle_selective --feature_style 'selective' --lr 0.1 --use_gpu False --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/
wait