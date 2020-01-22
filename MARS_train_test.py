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
from xgboost import XGBClassifier
import MARS_annotation_parsers as map
import MARS_ts_util as mts
from MARS_clf_helpers import *
from seqIo import *

# warnings.filterwarnings("ignore")
# plt.ioff()


def choose_classifier(clf_type='xgb', clf_params=dict()):

    MLPdefaults = {'hidden_layer_sizes': (256, 512),
                   'learning_rate_init': 0.001,
                   'learning_rate': 'adaptive',
                   'max_iter': 100000,
                   'alpha': 0.0001}

    XGBdefaults = {'n_estimators': 2000}

    if clf_type.lower() == 'mlp':
        hidden_layer_sizes = clf_params['hidden_layer_sizes'] if 'hidden_layer_sizes' in clf_params.keys() else MLPdefaults['hidden_layer_sizes']
        learning_rate_init = clf_params['learning_rate_init'] if 'learning_rate_init' in clf_params.keys() else MLPdefaults['learning_rate_init']
        learning_rate = clf_params['learning_rate'] if 'learning_rate' in clf_params.keys() else MLPdefaults['learning_rate']
        max_iter = clf_params['max_iter'] if 'max_iter' in clf_params.keys() else MLPdefaults['max_iter']
        alpha = clf_params['alpha'] if 'alpha' in clf_params.keys() else MLPdefaults['alpha']

        mlp = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, random_state=1,
                            learning_rate=learning_rate, max_iter=max_iter,
                            learning_rate_init=learning_rate_init, verbose=0)
        clf = BaggingClassifier(mlp, max_samples=.1, n_jobs=3, random_state=7, verbose=0)

    else:
        if not clf_type.lower() == 'xgb':
            print('Unrecognized classifier type %s, defaulting to XGBoost!' % clf_type)
        n_estimators = clf_params['n_estimators'] if 'n_estimators' in clf_params.keys() else XGBdefaults['n_estimators']
        clf = XGBClassifier(n_estimators=n_estimators, random_state=1, n_jobs=-1)

    return clf


def check_smoothing_type(clf_params):
    do_wnd = clf_params['do_wnd'] if 'do_wnd' in clf_params.keys() else False
    do_cwt = clf_params['do_cwt'] if 'do_cwt' in clf_params.keys() else True
    return do_wnd, do_cwt


def load_data(video_path, video_list, keepLabels,
              ver=[7, 8], feat_type='top', verbose=0, do_wnd=False, do_cwt=False):
    data = []
    labels = []

    for v in video_list:
        vbase = os.path.basename(v)
        vid = []

        for file in os.listdir(v):
            if fnmatch.fnmatch(file, '*.txt') or fnmatch.fnmatch(file, '*.annot'):
                ann = file
            elif fnmatch.fnmatch(file, '*.seq'):
                seq = file

        # we load exact frame timestamps for *.annot files to make sure we get the time->frame conversion correct
        if fnmatch.fnmatch(ann, '*.annot'):
            sr = seqIo_reader(seq)
            timestamps = sr.getTs()
        else:
            timestamps = []

        for version in ver:
            fstr = os.path.join(video_path, v, 'output_v1_%d' % version, vbase,
                                vbase + '_raw_feat_%s_v1_%d.npz' % (feat_type, version))
            if os.path.isfile(fstr):
                if verbose:
                    print('loaded file: ' + os.path.basename(fstr))
                vid = np.load(open(fstr, 'rb'))

        if not vid:
            print('Feature file not found for %s' % vbase)
        else:
            d = vid['data_smooth']
            names = vid['features']
            d = mts.clean_data(d)
            n_feat = d.shape[2]

            # we remove some features that have the same value for both mice (hardcoded for now, shaaame)
            featToKeep = list(flatten([range(39), range(42, 58), 59, 61, 62, 63, range(113, n_feat)]))
            d = np.hstack((d[0, :, :], d[1, :, featToKeep].transpose()))

            # for this project, we also remove raw pixel-based features to keep things simple
            d = mts.remove_pixel_data(d, 'top')
            d = mts.clean_data(d)

            if do_wnd:
                d = mts.apply_windowing(d)
            elif do_cwt:
                d = mts.apply_wavelet_transform(d)
            data.append(d)

            beh = map.parse_annotations(os.path.join(video_path, v, ann), timestamps=timestamps)
            labels += beh['behs_frame']

            if len(beh['behs_frame']) != d.shape[0]:
                print('Length mismatch: %s %d %d' % (v, len(beh['behs_frame']), d.shape[0]))
    if not data:
        print('No feature files found')
        return [], []
    if (verbose):
        print('all test files loaded')

    y = {}
    for label_name in keepLabels.keys():
        y_temp = np.array([]).astype(int)
        for i in labels: y_temp = np.append(y_temp,1) if i in keepLabels[label_name] else np.append(y_temp,0)
        y[label_name] = y_temp

    data = np.concatenate(data, axis=0)
    data = clean_data(data)

    # we only really need this for training the classifier, oh well
    if(verbose):
        print('fitting preprocessing parameters...')
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    print('done!\n')

    return data, y, scaler, names


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


def run_training(beh_classifier, X_tr, y_tr, savedir, downsample_rate = 1, verbose=0):

    beh_name   = beh_classifier['beh_name']
    clf        = beh_classifier['clf']
    kn         = beh_classifier['k']
    blur_steps = beh_classifier['blur_steps']
    shift      = beh_classifier['shift']

    # get the labels for the current behavior
    t = time.time()
    y_tr_beh = y_tr[beh_name]

    # shuffle data
    X_tr, idx_tr = shuffle_fwd(X_tr)
    y_tr_beh = y_tr_beh[idx_tr]

    #downsample for classifier fitting
    X_tr_ds = X_tr[::downsample_rate, :]
    y_tr_ds = y_tr_beh[::downsample_rate]

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
    hmm_bin.transmat_ = get_transmat(y_tr_beh, 2)
    hmm_bin.emissionprob_ = get_emissionmat(y_tr_beh, y_pred_class, 2)
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
    hmm_fbs.emissionprob_ = get_emissionmat(y_tr_beh, y_pred_fbs, 2)
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


def run_test(name_classifier,X_te,y_te,verbose=0):

    with open(name_classifier, 'rb') as fp:
        classifier = dill.load(fp)
    beh_name   = classifier['beh_name']
    scaler     = classifier['scaler']
    clf        = classifier['clf']
    hmm_bin    = classifier['hmm_bin']
    hmm_fbs    = classifier['hmm_fbs']
    kn         = classifier['k']
    blur_steps = classifier['blur_steps']
    shift      = classifier['shift']

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


def train_classifier(behs, video_path, train_videos, train_annot, clf_params={}, ver=8, verbose=0):

    clfDefault = {'clf_type': 'xgb',
                  'feat_type': 'top',
                  'training_ds': 5}
    smkDefault = {'kn': np.array([0.5, 0.25, 0.5]),
                  'blur': 4,
                  'shift': 4}

    # determine which classifier type we're training, which features we're using, and what windowing to use:
    clf_type = clf_params['clf_type'] if 'clf_type' in clf_params.keys() else clfDefault['clf_type']
    feat_type = clf_params['feat_type'] if 'feat_type' in clf_params.keys() else clfDefault['feat_type']
    do_wnd, do_cwt = check_smoothing_type(clf_params)
    downsample_rate = clf_params['training_ds'] if 'training_ds' in clf_params.keys() else clfDefault['training_ds']
    if not (downsample_rate==int(downsample_rate)):
        print('Training set downsampling rate must be an integer; reverting to default value.')
        downsample_rate = clfDefault['training_ds']

    # determine some parameters for post-classification smoothing:
    kn = clf_params['smk_kn'] if 'smk_kn' in clf_params.keys() else smkDefault['kn']
    blur = clf_params['blur'] if 'blur' in clf_params.keys() else smkDefault['blur']
    shift = clf_params['shift'] if 'shift' in clf_params.keys() else smkDefault['shift']
    blur_steps = blur ** 2

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
    X_tr, y_tr, scaler, features = load_data(video_path, train_videos, train_annot, behs,
                                             ver=ver, feat_type=feat_type, verbose=verbose, do_wnd=do_wnd, do_cwt=do_cwt)
    dill.dump(scaler, open(savedir + 'scaler.dill', 'wb'))
    print('loaded training data: %d X %d - %s ' % (X_tr.shape[0], X_tr.shape[1], list(y_tr.keys())))

    # train each classifier in a loop:
    for b,beh_name in enumerate(behs.keys()):
        print('######################### %s #########################' % beh_name)
        beh_classifier = {'beh_name': beh_name, 'beh_id': b + 1, 'clf': classifier, 'scaler': scaler,
                           'k': kn, 'blur': blur, 'blur_steps': blur_steps, 'shift': shift}
        run_training(beh_classifier, X_tr, y_tr, savedir, downsample_rate, verbose)

    print('done training!')


def test_classifier(behs, video_path, test_videos, test_annot, clf_params={}, ver=8, verbose=0):

    clf_type = clf_params['clf_type'] if 'clf_type' in clf_params.keys() else 'xgb'
    feat_type = clf_params['feat_type'] if 'feat_type' in clf_params.keys() else 'top'
    do_wnd, do_cwt = check_smoothing_type(clf_params)
    suff = str(clf_params['n_trees']) if 'n_trees' in clf_params.keys() else ''
    suff = suff + '_wnd/' if do_wnd else suff + '_cwt/' if do_cwt else suff + '/'

    classifier_name = feat_type + '_' + clf_type + suff
    folder = 'mars_v1_' + str(ver[-1])
    savedir = os.path.join('trained_classifiers',folder, classifier_name)

    print('loading test data...')
    X_te_0, y_te, _, _ = load_data(video_path, test_videos, test_annot, behs,
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
            run_test(name_classifier, X_te_0, y_te, verbose)

    all_pred = assign_labels(proba, beh_list)
    all_pred_hmm = assign_labels(proba_hmm, beh_list)
    all_pred_fbs_hmm = assign_labels(proba_fbs_hmm, beh_list)

    print('all pred')
    print(len(gt))
    print(len(all_pred))
    print(' ')
    score_info(gt, all_pred)
    print('all pred hmm')
    score_info(gt, all_pred_hmm)
    print('all pred fbs hmm')
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


def run_classifier(behs, video_path, test_videos, test_annot, save_path=[], clf_params={}, ver=8, verbose=0):
    # hijacking the code of test_classifier to output predictions on the test set

    clf_type = clf_params['clf_type'] if 'clf_type' in clf_params.keys() else 'xgb'
    feat_type = clf_params['feat_type'] if 'feat_type' in clf_params.keys() else 'top'
    do_wnd, do_cwt = check_smoothing_type(clf_params)
    suff = str(clf_params['n_trees']) if 'n_trees' in clf_params.keys() else ''
    suff = suff + '_wnd/' if do_wnd else suff + '_cwt/' if do_cwt else suff + '/'

    classifier_name = feat_type + '_' + clf_type + suff
    folder = 'mars_v1_' + str(ver[-1])
    savedir = os.path.join('trained_classifiers',folder, classifier_name)

    for vid,annot in zip(test_videos, test_annot):
        print('processing %s...' % vid)
        X_te_0, y_te, _, _ = load_data(video_path, [vid], [annot], behs,
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
                run_test(name_classifier, X_te_0, y_te, verbose)

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