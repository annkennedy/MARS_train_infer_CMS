from __future__ import division
import os,sys,glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import binarize
from sklearn.preprocessing import label_binarize
from matplotlib import cm as cmp
import matplotlib.colors as col
import random
import dill
import time
from copy import deepcopy
import itertools
from collections import Counter
import warnings
from sklearn.ensemble import BaggingClassifier
from matplotlib.colors import ListedColormap
import shutil
import scipy.io as sp
import math as mh
import xlwt
from hmmlearn import hmm
from scipy import signal
from matplotlib.ticker import FuncFormatter
import copy
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler,normalize
import pdb
from sklearn import preprocessing
import json
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from MARS_clf_helpers_orig import *
import MARS_ts_util as mts
import pdb

warnings.filterwarnings("ignore")
plt.ioff()
behs = ['sniff','mount','attack']
label2id = {'other': 0,
            'cable_fix': 0,
            'intruder_introduction': 0,
            'corner': 0,
            'ignore': 0,
            'groom': 0,
            'groom_genital': 0,
            'grooming': 0,
            'tailrattle': 0,
            'tail_rattle': 0,
            'tailrattling': 0,
            'intruder_attacks': 0,
            'approach': 0,

            'sniffface': 1,
            'snifface': 1,
            'sniff-face': 1,
            'sniff_face': 1,
            'head-investigation': 1,
            'facesniffing': 1,

            'sniffurogenital': 1,
            'sniffgenitals': 1,
            'sniff_genitals': 1,
            'sniff-genital': 1,
            'anogen-investigation': 1,

            'sniff_body': 1,
            'sniffbody': 1,
            'bodysniffing': 1,
            'body-investigation': 1,
            'socialgrooming': 1,
            'sniff-body': 1,
            'closeinvestigate': 1,
            'closeinvestigation': 1,
            'investigation': 1,

            'mount': 2,
            'aggressivemount': 2,
            'mount_attempt': 2,
            'attempted_mount': 2,
            'intromission': 2,

            'attack': 3,

            'attempt_attack': 3,
            'attempted_attack': 3,
            'agg_investigation': 3,
            }

flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))


def choose_classfier(classifier_type):
    if classifier_type == 0:
        mlp = MLPClassifier(solver='adam', alpha=0.0001, hidden_layer_sizes=(256, 512), random_state=1,
                            learning_rate='adaptive', max_iter=100000,
                            learning_rate_init=.001, verbose=0)
        clf = BaggingClassifier(mlp, max_samples=.1, n_jobs=3, random_state=7, verbose=0)

        clf_name = 'mlp'

    if classifier_type == 1:
        clf = XGBClassifier(n_estimators=2000, random_state=1, n_jobs=-1)
        clf_name = 'xgb'

    return clf, clf_name


def get_transmat(gt, n_states):
    # The hidden transitions are the transitions between states in the training set
    # Count transitions
    cs = [Counter() for _ in range(n_states)]
    prev = gt[0]
    for row in gt[1:]:
        cs[prev][row] += 1
        prev = row

    # Convert to probabilities
    transitions = np.zeros((n_states, n_states))
    for x in range(n_states):
        for y in range(n_states):
            transitions[x, y] = float(cs[x][y]) / float(sum(cs[x].values()))
    return transitions

def get_emissionmat(gt,pred,n_states):

    # The emmissions are the transitions from ground truth to predicted
    # Count emissions
    cs = [Counter() for _ in range(n_states)]
    for i, row in enumerate(gt):
        cs[row][pred[i]] += 1

    # Compute probabilities
    emissions = np.zeros((n_states, n_states))
    for x in range(n_states):
        for y in range(n_states):
            emissions[x, y] = float(cs[x][y]) / float(sum(cs[x].values()))
    return emissions

def score_info(y, y_pred):
    precision, recall, fscore, _ = score(y, y_pred)
    print('#Precision: {}'.format(np.round(precision, 3)))
    print('#Recall:    {}'.format(np.round(recall, 3)))
    print('#F1score:   {}'.format(np.round(fscore, 3)))
    return precision,recall,fscore

def shuffle_fwd(L):
    idx = range(L.shape[0])
    random.shuffle(idx)
    return L[idx], idx

def shuffle_back(L,idx):
    L_out = np.zeros(L.shape)
    for i,j in enumerate(idx):
        L_out[j] = L[i]
    return L_out


def load_data_train(video_list, video_path, train_annot, ver):
    # print "loading train data"
    data = []
    labels = []

    for v,ann in zip(video_list,train_annot):
        vbase = os.path.basename(v)
        vbase2 = '_'.join(vbase.split('_')[:-1])
        for version in ver:
            fstr = os.path.join(video_path, v, vbase + '_raw_feat_top_v1_%d.npz' % version)
            fstr2 = os.path.join(video_path, v, vbase2 + '_raw_feat_top_v1_%d.npz' % version)
            if os.path.isfile(fstr):
                vid = dict(np.load(open(fstr, 'rb')))
            elif os.path.isfile(fstr2):
                vid = dict(np.load(open(fstr2, 'rb')))
                
        names=vid['features']
        if 'data_smooth' in vid.keys():
            d = vid['data_smooth']
            d = mts.clean_data(d)
            n_feat = d.shape[2]

            # we remove some features that have the same value for both mice (hardcoded for now, shaaame)
            featToKeep = list(flatten([range(39), range(49, 58), 59, 61, 62, 63, range(113, n_feat)]))
            d = np.hstack((d[0, :, :], d[1, :, featToKeep].transpose()))

        else: # this is for features created with MARS_feature_extractor (which currently doesn't build data_smooth)
            d = vid['data']
        
        d = mts.apply_windowing(d)
        data.append(d)
        timestamps = []
        if('time' in vid.keys()):
            timestamps = vid['time']

        beh = parse_annotations(os.path.join(video_path, v, ann), timestamps=timestamps)
        labels += beh['behs_frame']

        if len(beh['behs_frame'])!=d.shape[0]:
            print('%s %d %d' % (v, len(beh['behs_frame']), d.shape[0]))
    
    y = np.array([]).astype(int)
    for i in labels: y=np.append(y,label2id[i]) if i in label2id else np.append(y,0)

    data = np.concatenate(data,axis=0)

    # correct bad values
    idx = np.where(np.isnan(data) | np.isinf(data))
    if idx:
        for j in range(len(idx[0])):
            if idx[0][j] == 0:
                data[idx[0][j], idx[1][j]] = 0.
            else:
                data[idx[0][j], idx[1][j]] = data[idx[0][j] - 1, idx[1][j]]

    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    return data, y, scaler, names


def load_data_test(video_list, video_path, test_annot, scaler,ver):
    # print "loading test data"
    data = []
    labels = []

    for v, ann in zip(video_list, test_annot):
        vbase = os.path.basename(v)
        vbase2 = '_'.join(vbase.split('_')[:-1])
        for version in ver:
            fstr = os.path.join(video_path, v, vbase + '_raw_feat_top_v1_%d.npz' % version)
            fstr2 = os.path.join(video_path, v, vbase2 + '_raw_feat_top_v1_%d.npz' % version)
            if os.path.isfile(fstr):
                vid = dict(np.load(open(fstr, 'rb')))
            elif os.path.isfile(fstr2):
                vid = dict(np.load(open(fstr2, 'rb')))
        
        if 'data_smooth' in vid.keys():
            d = vid['data_smooth']
            d = mts.clean_data(d)
            n_feat = d.shape[2]

            # we remove some features that have the same value for both mice (hardcoded for now, shaaame)
            featToKeep = list(flatten([range(39), range(49, 58), 59, 61, 62, 63, range(113, n_feat)]))
            d = np.hstack((d[0, :, :], d[1, :, featToKeep].transpose()))

        else: # this is for features created with MARS_feature_extractor (which currently doesn't build data_smooth)
            d = vid['data']
        
        d = mts.apply_windowing(d)
        data.append(d)

        beh = parse_annotations(os.path.join(video_path, v, ann))
        labels += beh['behs_frame']

        if len(beh['behs_frame']) != d.shape[0]:
            print(v)

    y = np.array([]).astype(int)
    for i in labels: y = np.append(y, label2id[i]) if i in label2id else np.append(y, 0)

    data = np.concatenate(data, axis=0)

    # correct bad values
    idx = np.where(np.isnan(data) | np.isinf(data))
    if idx:
        for j in range(len(idx[0])):
            if idx[0][j] == 0:
                data[idx[0][j], idx[1][j]] = 0.
            else:
                data[idx[0][j], idx[1][j]] = data[idx[0][j] - 1, idx[1][j]]
    data = scaler.transform(data)

    return data, y


class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

def prf_metrics(y_tr_beh,pd_class,beh):
    eps=np.spacing(1)
    pred_pos = np.where(pd_class == 1)[0]
    true_pred = np.where(y_tr_beh[pred_pos] == 1)[0]
    true_pos = np.where(y_tr_beh == 1)[0]
    pred_true = np.where(pd_class[true_pos] == 1)[0]

    n_pred_pos = len(pred_pos)
    n_true_pred = len(true_pred)
    n_true_pos = len(true_pos)
    n_pred_true = len(pred_true)

    precision = n_true_pred / (n_pred_pos+eps)
    recall = n_pred_true / (n_true_pos+eps)
    f_measure = 2 * precision * recall / (precision + recall+np.spacing(1))
    print('P: %5.4f, R: %5.4f, F1: %5.4f    %s' % (precision, recall, f_measure,beh))
    return precision, recall, f_measure

class Batch:
    def __init__(self, iterable, condition=(lambda x: True), limit=None):
        self.iterator = iter(iterable)
        self.condition = condition
        self.limit = limit
        try:
            self.current = next(self.iterator)
        except StopIteration:
            self.on_going = False
        else:
            self.on_going = True

    def group(self):
        yield self.current
        # start enumerate at 1 because we already yielded the last saved item
        for num, item in enumerate(self.iterator, 1):
            self.current = item
            if num == self.limit or self.condition(item):
                break
            yield item
        else:
            self.on_going = False

    def __iter__(self):
        while self.on_going:
            yield self.group()

def do_fbs(y_pred_class, kn, blur, blur_steps, shift):
    """Does forward-backward smoothing."""
    len_y = len(y_pred_class)

    # fbs with classes
    z = np.zeros((3, len_y))  # Make a matrix to hold the shifted predictions --one row for each shift.

    # Create mirrored start and end indices for extending the length of our prediction vector.
    mirrored_start = range(shift, -1, -1)  # Creates indices that go (shift, shift-1, ..., 0)
    mirrored_end = range(len_y - 1, len_y - 1 - shift, -1)  # Creates indices that go (-1, -2, ..., -shift)

    # Now we extend the predictions to have a mirrored portion on the front and back.
    extended_predictions = np.r_[
        y_pred_class[mirrored_start],
        y_pred_class,
        y_pred_class[mirrored_end]
    ]

    # Do our blurring.
    for s in range(blur_steps):
        extended_predictions = signal.convolve(np.r_[extended_predictions[0],
                                                     extended_predictions,
                                                     extended_predictions[-1]],
                                               kn / kn.sum(),  # The kernel we are convolving.
                                               'valid')  # Only use valid conformations of the filter.
        # Note: this will leave us with 2 fewer items in our signal each iteration, so we append on both sides.

    z[0, :] = extended_predictions[2 * shift + 1:]
    z[1, :] = extended_predictions[:-2 * shift - 1]
    z[2, :] = extended_predictions[shift + 1:-shift]

    z_mean = np.mean(z, axis=0)  # Average the blurred and shifted signals together.

    y_pred_fbs = binarize(z_mean.reshape((-1, 1)), .5).astype(int).reshape((1, -1))[0]  # Anything that has a signal strength over 0.5, is taken to be positive.

    return y_pred_fbs

def do_hmm(gt,pd):
    hmm_bin = hmm.MultinomialHMM(n_components=2, algorithm="viterbi", random_state=42, params="", init_params="")
    hmm_bin.startprob_ = np.array([np.sum(gt == i) / float(len(gt)) for i in range(2)])
    hmm_bin.transmat_ = get_transmat(gt, 2)
    hmm_bin.emissionprob_ = get_emissionmat(gt, pd, 2)
    return hmm_bin

def assign_labels(all_predicted_probabilities, behaviors_used):
    ''' Assigns labels based on the provided probabilities.'''
    labels = []
    labels_num =[]
    num_frames = all_predicted_probabilities.shape[0]
    # Looping over frames, determine which annotation label to take.
    for i in xrange(num_frames):
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
            # print pos
            # pdb.set_trace()
            max_prob = np.argmax(predicted_class_probabilities[pos])

            # This argmax is, by construction, the id for this behavior.
            beh_frame = pos[max_prob]
            # We also want to save that probability,maybe.
            proba_frame = predicted_class_probabilities[beh_frame]
            labels += [behaviors_used[beh_frame]]
            beh_frame += 1
        labels_num.append(beh_frame)

    return labels_num



for clf_n in [1]:

    video_path = '/groups/Andersonlab/CMS273/'
    train_videos = [os.path.join('TRAIN',v) for v in os.listdir(video_path+'TRAIN')]
    test_videos = [os.path.join('TEST',v) for v in os.listdir(video_path+'TEST')]
    train_annot = [os.path.join(r,v) for r in train_videos for v in [item for sublist in [glob.glob(video_path + r + e) for e in ['/*.annot', '/*.txt']] for item in sublist]]
    test_annot = [os.path.join(r,v) for r in test_videos for v in [item for sublist in [glob.glob(video_path + r + e) for e in ['/*.annot', '/*.txt']] for item in sublist]]	
    dataset = 'tm'
    ver=[7,8]
    
    n_trees=2000
    suff = str(n_trees) if clf_n==1 else ''
    classifier, clf_name = choose_classfier(clf_n)
    annotator='top_pcf_' + dataset + '_' + clf_name + suff  + '_wnd/'
    folder = 'mars_v1_8'
    savedir = os.path.join(folder, annotator)
    if not os.path.exists(savedir):os.makedirs(savedir)
    print(annotator.upper())
    
    n_classes = len(behs)
    kn = np.array([.5, .25, .5])
    blur = 4;shift = 4;blur_steps = blur ** 2
    f = open(savedir + '/log_selection.txt', 'w')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)
    
    print('loading data')
    X_tr, y_tr, scaler, features = load_data_train(train_videos, video_path, train_annot, ver)
    y_tr_bin = label_binarize(y_tr, range(n_classes+1))
    dill.dump(scaler, open(savedir + 'scaler.dill', 'wb'))
    
    X_te, y_te = load_data_test(test_videos, video_path, test_annot, scaler, ver)
    y_te_bin = label_binarize(y_te, range(n_classes + 1))
    print('data loaded')
    print ('train data %d X %d - %s ' % (X_tr.shape[0], X_tr.shape[1], list(set(y_tr))))
    print ('test data %d X %d - %s ' % (X_te.shape[0], X_te.shape[1], list(set(y_te))))
    counter_train = Counter(y_tr)
    print(counter_train)
    counter_test = Counter(y_te)
    print(counter_test)
    
    gt = np.zeros((len(y_te), n_classes)).astype(int)
    proba = np.zeros((len(y_te), n_classes, 2))
    preds = np.zeros((len(y_te), n_classes)).astype(int)
    preds_hmm = np.zeros((len(y_te), n_classes)).astype(int)
    proba_hmm = np.zeros((len(y_te), n_classes, 2))
    preds_fbs_hmm = np.zeros((len(y_te), n_classes)).astype(int)
    proba_fbs_hmm = np.zeros((len(y_te), n_classes, 2))
    
    
    for b in range(n_classes):
    	print('######################### %s ###############' % behs[b])
    	#1. train
    	clf=deepcopy(classifier)
    	t = time.time()
    	y_tr_beh = y_tr_bin[:, b + 1]
    	X_tr, idx_tr = shuffle_fwd(X_tr)
    	y_tr_beh = y_tr_beh[idx_tr]
    	clf.fit(X_tr, y_tr_beh)
    	X_tr = shuffle_back(X_tr, idx_tr)
    	y_tr_beh = shuffle_back(y_tr_beh, idx_tr).astype(int)
    
    	# 2. test on all of the data
    	y_pred_proba = np.zeros((len(y_tr_beh), 2))
    	gen = Batch(range(len(y_tr_beh)), lambda x: x % 1e5 == 0, 1e5)
    	for i in gen:
    		inds = list(i)
    		pd_proba_tmp = (clf.predict_proba(X_tr[inds]))
    		y_pred_proba[inds] = pd_proba_tmp
    	y_pred_class = np.argmax(y_pred_proba, axis=1)
    
    	# do hmm
    	hmm_bin = hmm.MultinomialHMM(n_components=2, algorithm="viterbi", random_state=42, params="", init_params="")
    	hmm_bin.startprob_ = np.array([np.sum(y_tr_beh == i) / float(len(y_tr_beh)) for i in range(2)])
    	hmm_bin.transmat_ = get_transmat(y_tr_beh, 2)
    	hmm_bin.emissionprob_ = get_emissionmat(y_tr_beh,y_pred_class, 2)
    	y_proba_hmm = hmm_bin.predict_proba(y_pred_class.reshape((-1, 1)))
    	y_pred_hmm = np.argmax(y_proba_hmm, axis=1)
    
    	# fbs with classes
    	len_y = len(y_tr_bin)
    	z = np.zeros((3, len_y))
    
    	#pdb.set_trace()
    
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
    
    	dt = (time.time() - t)/60.
    	print('training took %.2f mins' % dt)
    	_,_,_ = prf_metrics(y_tr_beh, y_pred_class, behs[b])
    	_,_,_ = prf_metrics(y_tr_beh, y_pred_hmm, behs[b])
    	precision, recall, f_measure = prf_metrics(y_tr_beh, y_pred_fbs_hmm, behs[b])
    
    	beh_classifier = {'beh_name': behs[b],
    					  'beh_id': b + 1,
    					  'scaler': scaler,
    					  'features': features,
    					  'clf': clf,
    					  'precision': precision,
    					  'recall': recall,
    					  'f_measure': f_measure,
    					  'k': kn,
    					  'blur': blur,
    					  'blur_steps': blur_steps,
    					  'shift': shift,
    					  'hmm':hmm_bin,
    					  'hmm_fbs':hmm_fbs
    					  }
    	dill.dump(beh_classifier, open(savedir + 'classifier_' + behs[b], 'wb'))
    
    	t=time.time()
    	len_y = len(y_te_bin)
    	y_te_beh = y_te_bin[:, b + 1]
    	gt[:, b] = y_te_beh
    
    	y_pred_proba = clf.predict_proba(X_te)
    	proba[:, b, :] = y_pred_proba
    	y_pred_class = np.argmax(y_pred_proba, axis=1)
    	preds[:, b] = y_pred_class
    
    	y_proba_hmm = hmm_bin.predict_proba(y_pred_class.reshape((-1, 1)))
    	y_pred_hmm = np.argmax(y_proba_hmm, axis=1)
    	proba_hmm[:, b, :] = y_proba_hmm
    	preds_hmm[:, b] = y_pred_hmm
    
    	z = np.zeros((3, len_y))
    	y_fbs = np.r_[y_pred_class[range(shift, -1, -1)], y_pred_class, y_pred_class[range(len_y - 1, len_y - 1 - shift, -1)]]
    	for s in range(blur_steps): y_fbs = signal.convolve(np.r_[y_fbs[0], y_fbs, y_fbs[-1]], kn / kn.sum(), 'valid')
    	z[0, :] = y_fbs[2 * shift + 1:]
    	z[1, :] = y_fbs[:-2 * shift - 1]
    	z[2, :] = y_fbs[shift + 1:-shift]
    	z_mean = np.mean(z, axis=0)
    	y_pred_fbs = binarize(z_mean.reshape((-1, 1)), .5).astype(int).reshape((1, -1))[0]
    
    	y_proba_fbs_hmm = hmm_fbs.predict_proba(y_pred_fbs.reshape((-1, 1)))
    	y_pred_fbs_hmm = np.argmax(y_proba_fbs_hmm, axis=1)
    	preds_fbs_hmm[:, b] = y_pred_fbs_hmm
    	proba_fbs_hmm[:, b, :] = y_proba_fbs_hmm
    	dt=time.time()-t
    	print('inference took %.2f sec' % dt)
    
    	print('########## pd #########')
    	prf_metrics(y_te_bin[:,b+1], preds[:,b], behs[b])
    	print('########## hmm #########')
    	prf_metrics(y_te_bin[:,b+1], preds_hmm[:,b], behs[b])
    	print('########## fbs hmm #########')
    	prf_metrics(y_te_bin[:,b+1], preds_fbs_hmm[:,b], behs[b])
    
    
    all_pred = assign_labels(proba,behs)
    all_pred_hmm = assign_labels(proba_hmm,behs)
    all_pred_fbs_hmm = assign_labels(proba_fbs_hmm,behs)
    
    print('all pred')
    score_info(y_te, all_pred)
    print('all pred hmm')
    score_info(y_te, all_pred_hmm)
    print('all pred fbs hmm')
    score_info(y_te, all_pred_fbs_hmm)
    
    
    P = {'0_G':gt,
    	 '0_Gc':y_te,
    	 '1_pd':preds,
    	 '2_pd_hmm':preds_hmm,
    	 '3_pd_fbs_hmm':preds_fbs_hmm,
    	 '4_proba_pd':proba,
    	 '5_proba_pd_hmm':proba_hmm,
    	 '6_proba_pd_hmm_fbs':proba_fbs_hmm,
    	 '7_pred_ass':all_pred,
    	 '8_pred_hmm_ass':all_pred_hmm,
    	 '9_pred_fbs_hmm_ass':all_pred_fbs_hmm
    	 }
    dill.dump(P, open(savedir + 'results.dill', 'wb'))