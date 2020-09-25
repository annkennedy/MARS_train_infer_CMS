#!/bin/bash

cd /home/kennedya/MARS_train_infer_CMS/
source /home/kennedya/anaconda2/bin/activate mars_tf

rundate=date '%(%Y%m%d_%H)T'

rundate=$(date +'%Y%m%d_%H')
echo $1
python run_training.py --behavior $1 --trainset $2 --evalset $3 --testset $4 --earlystopping 50 --suffix $rundate > "/home/kennedya/zmars_log_orig/${2}--${4}_${1}.txt"