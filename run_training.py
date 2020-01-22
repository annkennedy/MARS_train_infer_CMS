import os
import MARS_train_test as mars
import sys


# sometimes people use different names or spellings when annotating for a behavior. This chunk of code makes sure we get
# them all when we build our training/test sets.
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
    train_videos = train_videos + [os.path.join('old_cable_data','train_cable',i) for i in tr_cable.keys()]
    train_annot = train_annot + [tr_cable[i] for i in tr_cable.keys()]

elif sys.argv[1]=='attack':
    behs = {'attack':        ['attack']}
    train_videos = train_videos + [os.path.join('old_cable_data', 'train_cable', i) for i in tr_cable.keys()]
    train_annot = train_annot + [tr_cable[i] for i in tr_cable.keys()]
else:
    print('I didn''t recognize that behavior, aborting')


video_path = '/groups/Andersonlab/CMS273/'
train_videos = os.listdir(video_path+'TRAIN')
test_videos = os.listdir(video_path+'TEST')


save_path = '/home/kennedya/output_annot/'
clf_params = dict(clf_type='xgb', n_trees=1500, feat_type='top', do_cwt=True, do_wnd=False)
ver = [7,8]

if (sys.argv[2]=='train') or (sys.argv[2]=='both'):
    mars.train_classifier(behs, video_path, train_videos, clf_params, ver=ver, verbose=1)

if (sys.argv[2]=='test') or (sys.argv[2]=='both'):
    mars.test_classifier(behs, video_path, test_videos, clf_params, ver=ver, verbose=1)
    mars.run_classifier(behs, video_path, test_videos, save_path=save_path, clf_params=clf_params,
                            ver=ver, verbose=1)

if (sys.argv[2]=='runontest'):
    mars.run_classifier(behs, video_path, test_videos, save_path=save_path, clf_params=clf_params,
                            ver=ver, verbose=1)