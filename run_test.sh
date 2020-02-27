#!/bin/bash
#
#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -c 1 # 1 core per task
#SBATCH --mem-per-cpu=2G   # memory per CPU core
#SBATCH -J "run_training"   # job name
#SBATCH --mail-user=mlevine@caltech.edu   # email address

source activate mars_tf
cd ~/MARS_train_infer_CMS/

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python test_run.py --lr 0.1 --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/ --output_path /groups/Andersonlab/CMS273/rnn_test_output_lr0.1 &
srun --exclusive -N 1 -n 1 python test_run.py --lr 0.05 --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/ --output_path /groups/Andersonlab/CMS273/rnn_test_output_lr0.01 &
srun --exclusive -N 1 -n 1 python test_run.py --lr 0.2 --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/ --output_path /groups/Andersonlab/CMS273/rnn_test_output_lr0.2
wait