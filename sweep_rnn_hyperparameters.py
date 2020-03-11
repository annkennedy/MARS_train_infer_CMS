import os
import itertools

# Adapted from https://vsoch.github.io/lessons/sherlock-jobs/

OUTPUT_DIR = '/groups/Andersonlab/CMS273/rnn_outputs/hyperparameter_sweep_v0'

SETTINGS = {'num_epochs': [100],
                'hidden_dim': [5,10,25],
                'num_rnn_layers': [1,2,3,4],
                'feature_style': ['keypoints_only','selective','all'],
                'use_glm_scores': [True, False],
                'learn_glm_bias': [True, False],
                'model_name': ['GRUTagger'],
                'use_gpu': [True],
                'train_path': ['/groups/Andersonlab/CMS273/TRAIN_lite/'],
                'test_path': ['/groups/Andersonlab/CMS273/TEST_lite/'],
                'bidirectional': [True],
                'lr': [0.1]
            }

SHORT_NAMES = {'hidden_dim': 'hd',
                'num_rnn_layers': 'nlayers',
                'use_glm_scores': 'glmScores',
                'learn_glm_bias': 'glmBias',
                'feature_style': 'featureStyle'}


def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

def main(output_dir=OUTPUT_DIR, settings=SETTINGS):

    # Make top level directories
    mkdir_p(output_dir)
    job_directory = os.path.join(output_dir,'.job')
    mkdir_p(job_directory)

    var_names = [key for key in settings.keys() if len(settings[key]) > 1]# list of keys that are the experimental variables here
    keys, values = zip(*settings.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for exp_dict in experiments:
        # create run_dir and job_file pathnames
        nametag = ""
        for v in var_names:
            nametag += "{0}{1}_".format(SHORT_NAMES[v],exp_dict[v])
        nametag = nametag.rstrip('_')
        run_dir = os.path.join(output_dir, nametag)
        job_file = os.path.join(job_directory,"{0}.job".format(nametag))

        if os.path.exists(run_dir):
            print('Skipping:', run_dir)
            continue
        else:
            mkdir_p(run_dir)


        # build sbatch job script and write to file
        sbatch_str = ""
        sbatch_str += "#!/bin/bash\n"
        sbatch_str += "#SBATCH --account=andersonlab\n" # account name
        sbatch_str += "#SBATCH --job-name=%s.job\n" % nametag
        sbatch_str += "#SBATCH --output=.out/%s.out\n" % nametag
        sbatch_str += "#SBATCH --error=.out/%s.err\n" % nametag
        sbatch_str += "#SBATCH --time=48:00:00\n" # 48hr
        # sbatch_str += "#SBATCH --mem=12000\n"
        sbatch_str += "#SBATCH --gres=gpu:1\n"
        sbatch_str += "#SBATCH --mail-type=ALL\n"
        sbatch_str += "#SBATCH --mail-user=$USER@caltech.edu\n"
        sbatch_str += "conda activate mars_tf\n"

        sbatch_str += "python $HOME/MARS_train_infer_CMS/test_run.py"
        for key in exp_dict:
            sbatch_str += ' --{0} {1}'.format(key, exp_dict[key])
        sbatch_str += '\n'

        with open(job_file, 'w') as fh:
            fh.writelines(sbatch_str)

        # run the sbatch job script
        os.system("sbatch %s" %job_file)


if __name__ == '__main__':
    main()

