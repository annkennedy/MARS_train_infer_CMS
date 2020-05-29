#!/bin/bash

cd /home/kennedya/MARS_train_infer_CMS/
source /home/kennedya/anaconda2/bin/activate mars_tf


python run_training_orig.py > "/home/kennedya/zmars_log_orig/mars_log_attack.txt"