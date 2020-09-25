#!/bin/bash

cd /home/kennedya/MARS_train_infer_CMS/
source /home/kennedya/anaconda2/bin/activate mars_tf

printf -v date '%(%Y%m%d_%H-%M-%S)T\n' -1

python run_training_orig.py --suffix $date > "/home/kennedya/zmars_log_orig/mars_log_attack.txt"