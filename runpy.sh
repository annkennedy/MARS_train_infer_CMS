#!/bin/bash

cd /home/kennedya/MARS_train_infer_CMS/
source /home/kennedya/anaconda2/bin/activate mars_tf

if (($3==1))
then
    python run_training.py $1 --earlystopping $2 \
                > "/home/kennedya/zmars_log_attack/mars_log_attack_N-$1_S-$2.txt"
else
    python run_training.py $1 --earlystopping $2 \
                                    --dowavelet \
                > "/home/kennedya/zmars_log_attack/mars_log_attack_N-$1_S-$2_CWT.txt"
fi