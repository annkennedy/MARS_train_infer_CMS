from __future__ import division
import os,sys,fnmatch
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import binarize
import dill
import time
from sklearn.ensemble import BaggingClassifier
from hmmlearn import hmm
from scipy import signal
import copy
from sklearn.preprocessing import StandardScaler
# from xgboost import XGBClassifier
import MARS_annotation_parsers as map
import MARS_ts_util as mts
from MARS_clf_helpers import *
# from seqIo import *

# warnings.filterwarnings("ignore")
# plt.ioff()


def load_default_parameters():
    default_params = {'clf_type': 'xgb',
                      'feat_type': 'top',  # keep this to just top for now
                      'downsample_rate': 5,  # temporal downsampling applied to training data
                      'smk_kn': np.array([0.5, 0.25, 0.5]),
                      'blur': 4,
                      'shift': 4,
                      'do_wnd': False,
                      'do_cwt': False
                      }

    # in addition to these parameters, you can also store classifier-specific parameters in clf_params.
    # default values for those are defined below in choose_classifier.

    return default_params


def choose_classifier(clf_type='xgb', clf_params=dict()):

    MLPdefaults = {'hidden_layer_sizes': (256, 512),
                   'learning_rate_init': 0.001,
                   'learning_rate': 'adaptive',
                   'max_iter': 100000,
                   'alpha': 0.0001}

    XGBdefaults = {'n_estimators': 2000}

    # insert defaults for other classifier types here!

    if clf_type.lower() == 'mlp':
        for k in MLPdefaults.keys():
            if not k in clf_params.keys():
                clf_params[k] = MLPdefaults[k]

        hidden_layer_sizes = clf_params['hidden_layer_sizes']
        learning_rate_init = clf_params['learning_rate_init']
        learning_rate = clf_params['learning_rate']
        max_iter = clf_params['max_iter']
        alpha = clf_params['alpha']

        mlp = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, random_state=1,
                            learning_rate=learning_rate, max_iter=max_iter,
                            learning_rate_init=learning_rate_init, verbose=0)
        clf = BaggingClassifier(mlp, max_samples=.1, n_jobs=3, random_state=7, verbose=0)

    else:
        if not clf_type.lower() == 'xgb':
            print('Unrecognized classifier type %s, defaulting to XGBoost!' % clf_type)

        for k in XGBdefaults.keys():
            if not k in clf_params.keys():
                clf_params[k] = XGBdefaults[k]

        n_estimators = clf_params['n_estimators']
        # there are other XGB parameters but we haven't explored them yet.
        clf = XGBClassifier(n_estimators = n_estimators, random_state = 1, n_jobs = -1)

    return clf

def y_dict_to_array(y, key_order):
    new_y = []
    for key in key_order:
        new_y.append(y[key])
    z = np.array(new_y).T
    x = 1 - np.sum(z,axis=1)
    return np.hstack((z, x[:,None]))

def load_data(video_path, video_list, keepLabels, ver=[7, 8], feat_type='top', verbose=0, do_wnd=False, do_cwt=False):
    data = []
    labels = []
    Ybig = []

    for v in video_list:
        vbase = os.path.basename(v)
        vid = []
        seq = []

        for file in os.listdir(os.path.join(video_path, v)):
            if fnmatch.fnmatch(file, '*.txt') or fnmatch.fnmatch(file, '*.annot'):
                ann = file
            elif fnmatch.fnmatch(file, '*.seq'):
                seq = file

        # we load exact frame timestamps for *.annot files to make sure we get the time->frame conversion correct
        if fnmatch.fnmatch(ann, '*.annot') and seq:
            sr = seqIo_reader(seq)
            timestamps = sr.getTs()
        else:
            timestamps = []

        for version in ver:
            fstr = os.path.join(video_path, v, vbase + '_raw_feat_%s_v1_%d.npz' % (feat_type, version))
            if os.path.isfile(fstr):
                if verbose:
                    print('loaded file: ' + os.path.basename(fstr))
                vid = np.load(open(fstr, 'rb'))

        if not vid:
            print('Feature file not found for %s' % vbase)
        else:
            names = vid['features']
            if 'data_smooth' in vid.keys():
                d = vid['data_smooth']
                d = mts.clean_data(d)
                n_feat = d.shape[2]

                # we remove some features that have the same value for both mice (hardcoded for now, shaaame)
                featToKeep = list(flatten([range(39), range(49, 58), 59, 61, 62, 63, range(113, n_feat)]))
                d = np.hstack((d[0, :, :], d[1, :, featToKeep].transpose()))

                # for this project, we also remove raw pixel-based features to keep things simple
                d = mts.remove_pixel_data(d, 'top')
            else: # this is for features created with MARS_feature_extractor (which currently doesn't build data_smooth)
                d = vid['data']
            d = mts.clean_data(d)

            if do_wnd:
                d = mts.apply_windowing(d)
            elif do_cwt:
                d = mts.apply_wavelet_transform(d)
            data.append(np.array(d))

            beh = map.parse_annotations(os.path.join(video_path, v, ann), timestamps=timestamps)
            # labels += beh['behs_frame']
            labels.append(np.array(beh['behs_frame']))

            if len(beh['behs_frame']) != d.shape[0]:
                print('Length mismatch: %s %d %d' % (v, len(beh['behs_frame']), d.shape[0]))

            y = {}
            for label_name in keepLabels.keys():
                y_temp = np.array([]).astype(int)
                for i in beh['behs_frame']: y_temp = np.append(y_temp,1) if i in keepLabels[label_name] else np.append(y_temp,0)
                y[label_name] = y_temp
            Ybig.append(y)

    if not data:
        print('No feature files found')
        return [], [], [], []
    if (verbose):
        print('all files loaded')

    # we only really need this for training the classifier, oh well
    if(verbose):
        print('fitting preprocessing parameters...')

    print('done!\n')
    
    key_order = Ybig[0].keys()
    y_final = []
    for video in Ybig:
        y_final.append(y_dict_to_array(video, key_order))

    key_order += 'None'
    return data, y_final, key_order, names


def assign_labels(all_predicted_probabilities, behaviors_used):
    ''' Assigns labels based on the provided probabilities.'''
    labels = []
    labels_num =[]
    num_frames = all_predicted_probabilities.shape[0]
    # Looping over frames, determine which annotation label to take.
    for i in range(num_frames):
        # Get the [Nx2] matrix of current prediction probabilities.
        current_prediction_probabilities = all_predicted_probabilities[i]

        # Get the positive/negative labels for each behavior, by taking the argmax along the pos/neg axis.
        onehot_class_predictions = np.argmax(current_prediction_probabilities, axis=1)

        # Get the actual probabilities of those predictions.
        predicted_class_probabilities = np.max(current_prediction_probabilities, axis=1)

        # If every behavioral predictor agrees that the current_
        if np.all(onehot_class_predictions == 0):
            # The index here is one past any positive behavior --this is how we code for "other".
            beh_frame = 0
            # How do we get the probability of it being "other?" Since everyone's predicting it, we just take the mean.
            proba_frame = np.mean(predicted_class_probabilities)
            labels += ['other']
        else:
            # If we have positive predictions, we find the probabilities of the positive labels and take the argmax.
            pos = np.where(onehot_class_predictions)[0]
            max_prob = np.argmax(predicted_class_probabilities[pos])

            # This argmax is, by construction, the id for this behavior.
            beh_frame = pos[max_prob]
            proba_frame = predicted_class_probabilities[beh_frame]
            labels += [behaviors_used[beh_frame]]
            beh_frame += 1
        labels_num.append(beh_frame)

    return labels_num


def do_train(beh_classifier, X_tr, y_tr, savedir, verbose=0):

    beh_name = beh_classifier['beh_name']
    clf = beh_classifier['clf']
    clf_params = beh_classifier['params']

    # set some parameters for post-classification smoothing:
    kn = clf_params['smk_kn']
    blur_steps = clf_params['blur'] ** 2
    shift = clf_params['shift']

    # get the labels for the current behavior
    t = time.time()
    y_tr_beh = y_tr[beh_name]

    # shuffle data
    X_tr, idx_tr = shuffle_fwd(X_tr)
    y_tr_beh = y_tr_beh[idx_tr]

    #downsample for classifier fitting
    X_tr_ds = X_tr[::clf_params['downsample_rate'], :]
    y_tr_ds = y_tr_beh[::clf_params['downsample_rate']]

    # fit the classifier!
    if (verbose):
        print('fitting the classifier...')
    clf.fit(X_tr_ds, y_tr_ds)

    # shuffle back
    X_tr = shuffle_back(X_tr, idx_tr)
    y_tr_beh = shuffle_back(y_tr_beh, idx_tr).astype(int)

    # evaluate on training set
    if (verbose):
        print('evaluating on the training set...')
    y_pred_proba = np.zeros((len(y_tr_beh), 2))
    gen = Batch(range(len(y_tr_beh)), lambda x: x % 1e5 == 0, 1e5)
    for i in gen:
        inds = list(i)
        pd_proba_tmp = (clf.predict_proba(X_tr[inds]))
        y_pred_proba[inds] = pd_proba_tmp
    y_pred_class = np.argmax(y_pred_proba, axis=1)

    # do hmm
    if (verbose):
        print('fitting HMM smoother...')
    hmm_bin = hmm.MultinomialHMM(n_components=2, algorithm="viterbi", random_state=42, params="", init_params="")
    hmm_bin.startprob_ = np.array([np.sum(y_tr_beh == i) / float(len(y_tr_beh)) for i in range(2)])
    hmm_bin.transmat_ = mts.ansmat(y_tr_beh, 2)
    hmm_bin.emissionprob_ = mts.get_emissionmat(y_tr_beh, y_pred_class, 2)
    y_proba_hmm = hmm_bin.predict_proba(y_pred_class.reshape((-1, 1)))
    y_pred_hmm = np.argmax(y_proba_hmm, axis=1)

    # forward-backward smoothing with classes
    if (verbose):
        print('fitting forward-backward smoother...')
    len_y = len(y_tr_beh)
    z = np.zeros((3, len_y))
    y_fbs = np.r_[y_pred_hmm[range(shift, -1, -1)], y_pred_hmm, y_pred_hmm[range(len_y - 1, len_y - 1 - shift, -1)]]
    for s in range(blur_steps): y_fbs = signal.convolve(np.r_[y_fbs[0], y_fbs, y_fbs[-1]], kn / kn.sum(), 'valid')
    z[0, :] = y_fbs[2 * shift + 1:]
    z[1, :] = y_fbs[:-2 * shift - 1]
    z[2, :] = y_fbs[shift + 1:-shift]
    z_mean = np.mean(z, axis=0)
    y_pred_fbs = binarize(z_mean.reshape((-1, 1)), .5).astype(int).reshape((1, -1))[0]
    hmm_fbs = copy.deepcopy(hmm_bin)
    hmm_fbs.emissionprob_ = mts.get_emissionmat(y_tr_beh, y_pred_fbs, 2)
    y_proba_fbs_hmm = hmm_fbs.predict_proba(y_pred_fbs.reshape((-1, 1)))
    y_pred_fbs_hmm = np.argmax(y_proba_fbs_hmm, axis=1)

    # print the results of training
    dt = (time.time() - t) / 60.
    print('training took %.2f mins' % dt)
    _, _, _ = prf_metrics(y_tr_beh, y_pred_class, beh_name)
    _, _, _ = prf_metrics(y_tr_beh, y_pred_hmm, beh_name)
    precision, recall, f_measure = prf_metrics(y_tr_beh, y_pred_fbs_hmm, beh_name)

    beh_classifier.update({'clf': clf,
                           'precision': precision,
                           'recall': recall,
                           'f_measure': f_measure,
                           'hmm_bin': hmm_bin,
                           'hmm_fbs': hmm_fbs})

    dill.dump(beh_classifier, open(savedir + 'classifier_' + beh_name, 'wb'))


def do_test(name_classifier, X_te, y_te, verbose=0):

    with open(name_classifier, 'rb') as fp:
        classifier = dill.load(fp)

    # unpack the classifier
    beh_name = classifier['beh_name']
    scaler = classifier['scaler']
    clf = classifier['clf']

    # unpack the smoothers
    hmm_bin = classifier['hmm_bin']
    hmm_fbs = classifier['hmm_fbs']

    # unpack the smoothing parameters
    clf_params = classifier['params']
    kn = clf_params['smk_kn']
    blur_steps = clf_params['blur'] ** 2
    shift = clf_params['shift']

    X_te = scaler.transform(X_te)

    t = time.time()
    len_y = len(y_te[beh_name])
    y_te_beh = y_te[beh_name]
    gt = y_te_beh

    # predict probabilities:
    if (verbose):
        print('predicting behavior probability')
    y_pred_proba = clf.predict_proba(X_te)
    proba = y_pred_proba
    y_pred_class = np.argmax(y_pred_proba, axis=1)
    preds = y_pred_class

    # HMM smoothing:
    if (verbose):
        print('HMM smoothing')
    y_proba_hmm = hmm_bin.predict_proba(y_pred_class.reshape((-1, 1)))
    y_pred_hmm = np.argmax(y_proba_hmm, axis=1)
    proba_hmm = y_proba_hmm
    preds_hmm = y_pred_hmm

    # forward-backward smoothing:
    if (verbose):
        print('forward-backward smoothing')
    z = np.zeros((3, len_y))
    y_fbs = np.r_[
        y_pred_class[range(shift, -1, -1)], y_pred_class, y_pred_class[range(len_y - 1, len_y - 1 - shift, -1)]]
    for s in range(blur_steps): y_fbs = signal.convolve(np.r_[y_fbs[0], y_fbs, y_fbs[-1]], kn / kn.sum(), 'valid')
    z[0, :] = y_fbs[2 * shift + 1:]
    z[1, :] = y_fbs[:-2 * shift - 1]
    z[2, :] = y_fbs[shift + 1:-shift]
    z_mean = np.mean(z, axis=0)
    y_pred_fbs = binarize(z_mean.reshape((-1, 1)), .5).astype(int).reshape((1, -1))[0]

    y_proba_fbs_hmm = hmm_fbs.predict_proba(y_pred_fbs.reshape((-1, 1)))
    y_pred_fbs_hmm = np.argmax(y_proba_fbs_hmm, axis=1)
    preds_fbs_hmm = y_pred_fbs_hmm
    proba_fbs_hmm = y_proba_fbs_hmm
    dt = time.time() - t
    print('inference took %.2f sec' % dt)

    print('########## pd ##########')
    prf_metrics(y_te[beh_name], preds, beh_name)
    print('########## hmm ##########')
    prf_metrics(y_te[beh_name], preds_hmm, beh_name)
    print('########## fbs hmm ##########')
    prf_metrics(y_te[beh_name], preds_fbs_hmm, beh_name)

    return gt, proba, preds, preds_hmm, proba_hmm, preds_fbs_hmm, proba_fbs_hmm


def train_classifier(behs, video_path, train_videos, clf_params={}, ver=[7, 8], verbose=0):

    # unpack user-provided classification parameters, and use default values for those not provided.
    default_params = load_default_parameters()
    for k in default_params.keys():
        if k not in clf_params.keys():
            clf_params[k] = default_params[k]

    # determine which classifier type we're training, which features we're using, and what windowing to use:
    clf_type = clf_params['clf_type']
    feat_type = clf_params['feat_type']
    do_wnd = clf_params['do_wnd']
    do_cwt = clf_params['do_cwt']

    if not (clf_params['downsample_rate']==int(clf_params['downsample_rate'])):
        print('Training set downsampling rate must be an integer; reverting to default value.')
        clf_params['downsample_rate'] = default_params['downsample_rate']

    # now create the classifier and give it an informative name:
    classifier = choose_classifier(clf_type, clf_params)
    suff = str(clf_params['n_trees']) if 'n_trees' in clf_params.keys() else  ''
    suff = suff + '_wnd/' if do_wnd else suff + '_cwt/' if do_cwt else suff + '/'
    classifier_name = feat_type + '_' + clf_type + suff
    folder = 'mars_v1_' + str(ver[-1])
    savedir = os.path.join('trained_classifiers',folder, classifier_name)
    if not os.path.exists(savedir): os.makedirs(savedir)
    print('Training classifier: ' + classifier_name.upper())

    f = open(savedir + '/log_selection.txt', 'w')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)

    print('loading training data')
    X_tr, y_tr, scaler, features = load_data(video_path, train_videos, behs,
                                             ver=ver, feat_type=feat_type, verbose=verbose, do_wnd=do_wnd, do_cwt=do_cwt)
    dill.dump(scaler, open(savedir + 'scaler.dill', 'wb'))
    print('loaded training data: %d X %d - %s ' % (X_tr.shape[0], X_tr.shape[1], list(y_tr.keys())))

    # train each classifier in a loop:
    for b,beh_name in enumerate(behs.keys()):
        print('######################### %s #########################' % beh_name)
        beh_classifier = {'beh_name': beh_name,
                          'beh_id': b +1,
                          'clf': classifier,
                          'scaler': scaler,
                          'params': clf_params}
        do_train(beh_classifier, X_tr, y_tr, savedir, verbose)

    print('done training!')


def test_classifier(behs, video_path, test_videos, clf_params={}, ver=[7,8], verbose=0):

    default_params = load_default_parameters()
    for k in default_params.keys():
        if k not in clf_params.keys():
            clf_params[k] = default_params[k]

    clf_type = clf_params['clf_type']
    feat_type = clf_params['feat_type']
    do_wnd = clf_params['do_wnd']
    do_cwt = clf_params['do_cwt']

    suff = str(clf_params['n_trees']) if 'n_trees' in clf_params.keys() else ''
    suff = suff + '_wnd/' if do_wnd else suff + '_cwt/' if do_cwt else suff + '/'

    classifier_name = feat_type + '_' + clf_type + suff
    savedir = os.path.join('trained_classifiers',classifier_name)

    print('loading test data...')
    X_te_0, y_te, _, _ = load_data(video_path, test_videos, behs,
                                  ver=ver, feat_type=feat_type, verbose=verbose, do_wnd=do_wnd, do_cwt=do_cwt)
    print('loaded test data: %d X %d - %s ' % (X_te_0.shape[0], X_te_0.shape[1], list(set(y_te))))

    T = len(list(y_te.values())[0])
    n_classes = len(behs.keys())
    gt = np.zeros((T, n_classes)).astype(int)
    proba = np.zeros((T, n_classes, 2))
    preds = np.zeros((T, n_classes)).astype(int)
    preds_hmm = np.zeros((T, n_classes)).astype(int)
    proba_hmm = np.zeros((T, n_classes, 2))
    preds_fbs_hmm = np.zeros((T, n_classes)).astype(int)
    proba_fbs_hmm = np.zeros((T, n_classes, 2))
    beh_list = list()

    for b, beh_name in enumerate(behs.keys()):
        print('predicting behavior %s...' % beh_name)
        beh_list.append(beh_name)
        name_classifier = savedir + 'classifier_' + beh_name

        gt[:,b], proba[:, b, :], preds[:, b], preds_hmm[:, b], \
        proba_hmm[:, b, :], preds_fbs_hmm[:, b], proba_fbs_hmm[:, b, :] = \
            do_test(name_classifier, X_te_0, y_te, verbose)

    all_pred = assign_labels(proba, beh_list)
    all_pred_hmm = assign_labels(proba_hmm, beh_list)
    all_pred_fbs_hmm = assign_labels(proba_fbs_hmm, beh_list)

    print('Raw predictions:')
    score_info(gt, all_pred)
    print('Predictions after HMM smoothing:')
    score_info(gt, all_pred_hmm)
    print('Predictions after HMM and forward-backward smoothing:')
    score_info(gt, all_pred_fbs_hmm)
    P = {'0_G': gt,
         '0_Gc': y_te,
         '1_pd': preds,
         '2_pd_hmm': preds_hmm,
         '3_pd_fbs_hmm': preds_fbs_hmm,
         '4_proba_pd': proba,
         '5_proba_pd_hmm': proba_hmm,
         '6_proba_pd_hmm_fbs': proba_fbs_hmm,
         '7_pred_ass': all_pred,
         '8_pred_hmm_ass': all_pred_hmm,
         '9_pred_fbs_hmm_ass': all_pred_fbs_hmm
         }
    dill.dump(P, open(savedir + 'results.dill', 'wb'))


def run_classifier(behs, video_path, test_videos, test_annot, save_path=[], ver=[7,8], verbose=0):
    # this code actually saves *.annot files containing the raw predictions of the trained classifier,
    # instead of just giving you the precision and recall. You can load these *.annot files in Bento
    # along with the *.seq movies to inspect behavior labels by eye.
    #
    # Unlike test_classifier, this function runs classification on each video separately.

    default_params = load_default_parameters()
    for k in default_params.keys():
        if k not in clf_params.keys():
            clf_params[k] = default_params[k]

    clf_type = clf_params['clf_type']
    feat_type = clf_params['feat_type']
    do_wnd = clf_params['do_wnd']
    do_cwt = clf_params['do_cwt']

    suff = str(clf_params['n_trees']) if 'n_trees' in clf_params.keys() else ''
    suff = suff + '_wnd/' if do_wnd else suff + '_cwt/' if do_cwt else suff + '/'

    classifier_name = feat_type + '_' + clf_type + suff
    savedir = os.path.join('trained_classifiers', classifier_name)

    for vid in test_videos:
        print('processing %s...' % vid)
        X_te_0, y_te, _, _ = load_data(video_path, [vid], behs,
                                       ver=ver, feat_type=feat_type, verbose=verbose, do_wnd=do_wnd, do_cwt=do_cwt)

        if not y_te:
            print('skipping this video...\n\n')
            continue

        T = len(list(y_te.values())[0])
        n_classes = len(behs.keys())
        gt = np.zeros((T, n_classes)).astype(int)
        proba = np.zeros((T, n_classes, 2))
        preds = np.zeros((T, n_classes)).astype(int)
        preds_hmm = np.zeros((T, n_classes)).astype(int)
        proba_hmm = np.zeros((T, n_classes, 2))
        preds_fbs_hmm = np.zeros((T, n_classes)).astype(int)
        proba_fbs_hmm = np.zeros((T, n_classes, 2))
        beh_list = list()

        for b, beh_name in enumerate(behs.keys()):
            print('predicting behavior %s...' % beh_name)
            beh_list.append(beh_name)
            name_classifier = savedir + 'classifier_' + beh_name

            gt[:, b], proba[:, b, :], preds[:, b], preds_hmm[:, b], \
            proba_hmm[:, b, :], preds_fbs_hmm[:, b], proba_fbs_hmm[:, b, :] = \
                do_test(name_classifier, X_te_0, y_te, verbose)

        all_pred = assign_labels(proba, beh_list)
        all_pred_hmm = assign_labels(proba_hmm, beh_list)
        all_pred_fbs_hmm = assign_labels(proba_fbs_hmm, beh_list)
        all_gt = assign_labels(gt, beh_list) if b>1 else np.squeeze(gt)

        vname,_ = os.path.splitext(os.path.basename(vid))
        if not save_path:
            save_path = video_path
        map.dump_labels_bento(all_pred, os.path.join(save_path,'predictions_'+vname+'.annot'),
                              moviename=vid, framerate=30, beh_list=beh_list, gt=all_gt)
        map.dump_labels_bento(all_pred_hmm, os.path.join(save_path, 'predictions_hmm_' + vname + '.annot'),
                              moviename=vid, framerate=30, beh_list=beh_list, gt=all_gt)
        map.dump_labels_bento(all_pred_fbs_hmm, os.path.join(save_path, 'predictions_fbs_hmm_' + vname + '.annot'),
                              moviename=vid, framerate=30, beh_list=beh_list, gt=all_gt)
        print('\n\n')

