import os, sys
import MARS_train_test as mars
import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--earlystopping', dest='earlystopping', default=10,
                    help='number of early stopping steps (default: 10)')

parser.add_argument('--maxdepth', dest='max_depth', default=3,
                    help='max tree depth (default: 3)')
                    
parser.add_argument('--minchild', dest='minchild', default=1,
                    help='min_child_weight (default: 1)')
                    
parser.add_argument('--dowavelet', dest='do_cwt', action='store_true',
                    default=False,
                    help='use wavelet transform on features (default: false)')
                    
parser.add_argument('--testonly', dest='testonly', action='store_true',
                    default=False,
                    help='skip classifier training (default: false)')

parser.add_argument('--behavior', dest='behavior', default='attack',
                    help = 'behavior to train (default: attack)')

parser.add_argument('--clf', dest='clf_pth', default='',
                    help = 'path to a specific classifier to test (otherwise will determine from other arguments)')

parser.add_argument('--videos', dest='vid_pth', default='/groups/Andersonlab/CMS273/',
                    help = 'path to directory containing folders TRAIN_ORIG and TEST_ORIG with training/test data (default: current directory)')

args = parser.parse_args()

behs = mars.get_beh_dict(args.behavior)

# this tells the script where our training and test sets are located- you shouldn't need to change anything here.
video_path = args.vid_pth
train_videos = [os.path.join('TRAIN_ORIG',v) for v in os.listdir(video_path+'TRAIN_ORIG')]
eval_videos = [os.path.join('TEST_ORIG',v) for v in os.listdir(video_path+'TEST_ORIG')]
test_videos = [os.path.join('TEST_ORIG',v) for v in os.listdir(video_path+'TEST_ORIG')]


# these are the parameters that define our classifier.
do_wnd = True if not args.do_cwt else False
clf_params = dict(clf_type='xgb', feat_type='top', do_cwt=args.do_cwt, do_wnd=do_wnd, 
                  early_stopping=args.earlystopping, max_depth=args.max_depth,
                  min_child_weight=args.minchild, clf_path_hardcoded=args.clf_pth)

if not args.testonly:
    mars.train_classifier(behs, video_path, train_videos, eval_videos, clf_params=clf_params, verbose=1)

mars.test_classifier(behs, video_path, test_videos, clf_params=clf_params, verbose=1)