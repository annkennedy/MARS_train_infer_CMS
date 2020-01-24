import os
import MARS_train_test as mars
import sys


# The input to run_training is the name of the behavior we want to train a classifier for (right now I'm assuming
# you're going to train one classifier at a time.) However, before we call the training code, we have to change this
# string into a dictionary called "behs". This is because our annotators sometimes use different names or spellings
# when labeling a behavior. The purpose of this dictionary is to define what labels we want to lump together as positive
# examples for our classifier to learn from. (It could eventually be packaged into a function but oh well.)

if sys.argv[1]=='sniff_face':
    behs = {'sniff_face':    ['sniffface', 'snifface', 'sniff-face', 'sniff_face', 'head-investigation','facesniffing']}

elif sys.argv[1]=='sniff_genital':
    behs = {'sniff_genital': ['sniffurogenital','sniffgenitals','sniff_genitals','sniff-genital','sniff_genital',
                              'anogen-investigation']}

elif sys.argv[1]=='sniff_body':
    behs = {'sniff_body':    ['sniff_body','sniffbody','bodysniffing','body-investigation','socialgrooming',
                          'sniff-body','closeinvestigate','closeinvestigation','investigation']}

elif sys.argv[1]=='sniff':
    behs = {'sniff': ['sniffface', 'snifface', 'sniff-face', 'sniff_face', 'head-investigation','facesniffing',
                      'sniffurogenital','sniffgenitals','sniff_genitals','sniff-genital','sniff_genital',
                      'anogen-investigation','sniff_body', 'sniffbody', 'bodysniffing', 'body-investigation',
                      'socialgrooming','sniff-body', 'closeinvestigate', 'closeinvestigation', 'investigation']}

elif sys.argv[1]=='mount':
    behs = {'mount':         ['mount','aggressivemount','intromission','dom_mount']}

elif sys.argv[1]=='attack':
    behs = {'attack':        ['attack']}
else:
    print('I didn''t recognize that behavior, aborting')


# this tells the script where our training and test sets are located- you shouldn't need to change anything here.
video_path = '/groups/Andersonlab/CMS273/'
train_videos = [os.path.join('TRAIN_lite',v) for v in os.listdir(video_path+'TRAIN_lite')]
test_videos = [os.path.join('TEST_lite',v) for v in os.listdir(video_path+'TEST_lite')]

# this is where the trained classifier will be dumped.
save_path = '~/test_output/'

# these are the parameters that define our classifier.
clf_params = dict(clf_type='xgb', n_trees=1500, feat_type='top', do_cwt=False, do_wnd=False)

if (sys.argv[2]=='train') or (sys.argv[2]=='both'):
    mars.train_classifier(behs, video_path, train_videos, clf_params, verbose=1)

if (sys.argv[2]=='test') or (sys.argv[2]=='both'):
    mars.test_classifier(behs, video_path, test_videos, clf_params, verbose=1)
    mars.run_classifier(behs, video_path, test_videos, save_path=save_path, clf_params=clf_params, verbose=1)

if (sys.argv[2]=='runontest'):
    mars.run_classifier(behs, video_path, test_videos, save_path=save_path, clf_params=clf_params, verbose=1)