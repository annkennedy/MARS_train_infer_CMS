#!/bin/bash
#
#SBATCH --time=60:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=192G   # memory per CPU core
#SBATCH -J "run_training"   # job name
#SBATCH --mail-user=mlevine@caltech.edu   # email address

source activate mars_tf
cd ~/MARS_train_infer_CMS/
python run_training.py sniff_face both > test_output/mars_log.txt
