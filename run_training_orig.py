import os, sys
import MARS_train_test as mars
import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--earlystopping', dest='earlystopping', default=10,
                    help='number of early stopping steps (default: 10)')

parser.add_argument('--maxdepth', dest='max_depth', default=9,
                    help='max tree depth (default: 9)')
                    
parser.add_argument('--minchild', dest='minchild', default=4,
                    help='min_child_weight (default: 4)')
                    
parser.add_argument('--dowavelet', dest='do_cwt', action='store_true',
                    default=False,
                    help='use wavelet transform on features (default: false)')
                    
parser.add_argument('--testonly', dest='testonly', action='store_true',
                    default=False,
                    help='skip classifier training (default: false)')

parser.add_argument('--behavior', dest='behavior', default='attack',
                    help = 'behavior to train (default: attack)')

args = parser.parse_args()

behs = mars.get_beh_dict(args.behavior)

# this tells the script where our training and test sets are located- you shouldn't need to change anything here.
video_path = '/groups/Andersonlab/CMS273/'
train_videos = [os.path.join('TRAIN_ORIG',v) for v in os.listdir(video_path+'TRAIN_ORIG')]
eval_videos = [os.path.join('TEST_ORIG',v) for v in os.listdir(video_path+'TEST_ORIG')]
test_videos = [os.path.join('TEST_ORIG',v) for v in os.listdir(video_path+'TEST_ORIG')]


# these are the parameters that define our classifier.
do_wnd = True if not args.do_cwt else False
clf_params = dict(clf_type='xgb', feat_type='top', do_cwt=args.do_cwt, do_wnd=do_wnd, 
                  early_stopping=args.earlystopping, max_depth=args.max_depth,
                  min_child_weight=args.minchild)

if not args.testonly:
    mars.train_classifier(behs, video_path, train_videos, eval_videos, clf_params=clf_params, verbose=1)

mars.test_classifier(behs, video_path, test_videos, clf_params=clf_params, verbose=1)