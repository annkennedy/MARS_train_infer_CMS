#!/bin/bash

cd /home/kennedya/MARS_train_infer_CMS/
source /home/kennedya/anaconda2/bin/activate mars_tf

if (($4==1))
then
    python run_training.py --earlystopping $1 --maxdepth $2 --minchild $3 \
                > "/home/kennedya/zmars_log_attack/mars_log_attack_STOP$1_DEPTH$2_CHILD$3.txt"
else
    python run_training.py --earlystopping $1 --maxdepth $2 --minchild $3 \
                                    --dowavelet \
                > "/home/kennedya/zmars_log_attack/mars_log_attack_STOP$1_DEPTH$2_CHILD$3_CWT.txt"
fi