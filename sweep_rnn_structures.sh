#!/bin/bash
# https://s3-us-west-2.amazonaws.com/imss-hpc/index.html to generate script
#SBATCH --time=48:00:00   # walltime
#SBATCH --gres=gpu:1
#SBATCH --gid=andersonlab

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python test_run.py --output_path ./test_output_rnn_LSTM_backwards_v3 --model_name LSTMTagger --bidirectional True --lr 0.1 --use_gpu True --num_epochs 100 --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/  &
srun --exclusive -N 1 -n 1 python test_run.py --output_path ./test_output_rnn_GRU_v3 --model_name GRUTagger --bidirectional False --lr 0.1 --use_gpu True --num_epochs 100 --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/  &
srun --exclusive -N 1 -n 1 python test_run.py --output_path ./test_output_rnn_GRU_backwards_v3 --model_name GRUTagger --bidirectional True --lr 0.1 --use_gpu True --num_epochs 100 --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/  &
srun --exclusive -N 1 -n 1 python test_run.py --output_path ./test_output_rnn_LSTM_v3 --model_name LSTMTagger --bidirectional False --lr 0.1 --use_gpu True --num_epochs 100 --train_path /groups/Andersonlab/CMS273/TRAIN_lite/ --test_path /groups/Andersonlab/CMS273/TEST_lite/
wait