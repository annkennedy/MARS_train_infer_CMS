import os, sys
import MARS_train_test as mars
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('ntrees', type=int,
                    help='number of XGBoost trees to use (default: 1000')

parser.add_argument('--earlystopping', dest='earlystopping', default=10,
                    help='number of early stopping steps (default: 10)')
                    
parser.add_argument('--dowavelet', dest='do_cwt', action='store_true',
                    default=False,
                    help='use wavelet transform on features (default: false)')
                    
parser.add_argument('--testonly', dest='testonly', action='store_true',
                    default=False,
                    help='skip classifier training (default: false)')

parser.add_argument('--behavior', dest='behavior', default='attack',
                    help = 'behavior to train (default: attack)')

args = parser.parse_args()

beh = mars.get_beh_dict(args.behavior)

# this tells the script where our training and test sets are located- you shouldn't need to change anything here.
video_path = '/groups/Andersonlab/CMS273/'
train_videos = [os.path.join('TRAIN',v) for v in os.listdir(video_path+'TRAIN')]
eval_videos = [os.path.join('EVAL',v) for v in os.listdir(video_path+'EVAL')]
test_videos = [os.path.join('TEST',v) for v in os.listdir(video_path+'TEST')]


# these are the parameters that define our classifier.
do_wnd = True if not args.do_cwt else False
clf_params = dict(clf_type='xgb', n_trees=args.ntrees, feat_type='top', do_cwt=args.do_cwt, do_wnd=do_wnd, early_stopping=args.earlystopping)

if not args.testonly:
    mars.train_classifier(behs, video_path, train_videos, eval_videos, clf_params=clf_params, verbose=1)

mars.test_classifier(behs, video_path, test_videos, clf_params=clf_params, verbose=1)