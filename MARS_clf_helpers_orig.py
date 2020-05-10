from __future__ import division
import os,sys
import matplotlib.pyplot as plt
import numpy as np
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


def parse_annotations(fid,useChannel=['Ch1'],timestamps=[]):

    if(fid.endswith('.txt')):
        ann_dict = parse_txt(fid)
        return ann_dict

    elif(fid.endswith('.annot')):
        ann_dict = parse_annot(fid,useChannel,timestamps)
        return ann_dict


def parse_txt(f_ann):

    header='Caltech Behavior Annotator - Annotation File'
    conf = 'Configuration file:'
    fid = open(f_ann)
    ann = fid.read().splitlines()
    fid.close()
    NFrames = []
    #check the header
    assert ann[0].rstrip()==header
    assert ann[1].rstrip()==''
    assert ann[2].rstrip()== conf
    #parse action list
    l=3
    names=[None] *1000
    keys=[None] *1000
    types =[]
    bnds =[]
    k=-1

    #get config keys and names
    while True:
        ann[l] = ann[l].rstrip()
        if not isinstance(ann[l], str) or not ann[l]:
            l+=1
            break
        values = ann[l].split()
        k += 1
        names[k] = values[0]
        keys[k] = values[1]
        l+=1
    names = names[:k+1]
    keys = keys[:k+1]

    #read in each stream in turn until end of file
    bnds0 =[None]*10000
    types0 = [None]*10000
    actions0 = [None]*10000
    nStrm1 = 0
    while True:
        ann[l]=ann[l].rstrip()
        nStrm1 +=1
        t = ann[l].split(":")
        l += 1
        ann[l] = ann[l].rstrip()
        assert int(t[0][1])==nStrm1
        assert ann[l] == '-----------------------------'
        l+=1
        bnds1 =np.ones((10000,2),dtype=int)
        types1=np.ones(10000,dtype=int)*-1
        actions1 = [None] *10000
        k=0
        # start the annotations
        while True:
            ann[l] = ann[l].rstrip()
            t = ann[l]
            if not isinstance(t, str) or not t:
                l+=1
                break
            t = ann[l].split()
            type = [i for i in range(len(names)) if  t[2]== names[i]]
            type = type[0]
            if type==None:
                print('undefined behavior' + t[2])
            if bnds1[k-1,1]!= int(t[0])-1 and k>0:
                print('%d ~= %d' % (bnds1[k,1], int(t[0]) - 1))
            bnds1[k,:]=[int(t[0]),int(t[1])]
            types1[k] = type
            actions1[k] = names[type]
            k+=1
            l+=1
            if l == len(ann):
                break
        if nStrm1==1:
            nFrames = bnds1[k-1,1]
        assert nFrames == bnds1[k-1,1]
        bnds0[nStrm1-1] = bnds1[:k]
        types0[nStrm1-1] = types1[:k]
        actions0[nStrm1-1] = actions1[:k]
        if l==len(ann):
            break
        while not ann[l]:
            l+=1

    bnds = bnds0[:nStrm1]
    types = types0[:nStrm1]
    actions = actions0[:nStrm1]

    idx = 0
    if len(actions[0])< len(actions[1]):
        idx = 1
    type_frame = []
    action_frame = []
    len_bnd = []

    for i in range(len(bnds[idx])):
        numf  = bnds[idx][i,1] - bnds[idx][i,0]+1
        len_bnd.append(numf)
        action_frame.extend([actions[idx][i]] * numf)
        type_frame.extend([types[idx][i]] * numf)

    ann_dict = {
        'keys': keys,
        'behs':names,
        'nstrm':nStrm1,
        'nFrames': nFrames,
        'behs_se': bnds,
        'behs_dur': len_bnd,
        'behs_bout': actions,
        'behs_frame':action_frame
    }

    return ann_dict



def parse_annot(filename,useChannel=['Ch1'],timestamps=[]):
    """ Takes as input a path to a .annot file and returns the frame-wise behavioral labels."""
    if not filename:
        print("No filename provided")
        return -1

    behaviors = []
    channel_names = []
    keys = []

    channel_dict = {}
    with open(filename, 'r') as annot_file:
        line = annot_file.readline().rstrip()
        # Parse the movie files
        while line != '':
            line = annot_file.readline().rstrip()
            # Get movie files if you want

        # Parse the stim name and other stuff
        start_frame = 0
        end_frame = 0
        framerate = 30
        stim_name = ''

        line = annot_file.readline().rstrip()
        split_line = line.split()
        stim_name = split_line[-1]

        line = annot_file.readline().rstrip()
        split_line = line.split()
        start_frame = int(split_line[-1])

        line = annot_file.readline().rstrip()
        split_line = line.split()
        end_frame = int(split_line[-1])

        line = annot_file.readline().rstrip()
        if(not(line=='')): # newer annot files have a framerate line added
            split_line = line.split()
            framerate = float(split_line[-1])
            line = annot_file.readline().rstrip()
        assert (line == '')

        # Just pass through whitespace
        while line == '':
            line = annot_file.readline().rstrip()

        # pdb.set_trace()
        # At the beginning of list of channels
        assert 'channels' in line
        line = annot_file.readline().rstrip()
        while line != '':
            key = line
            keys.append(key)
            line = annot_file.readline().\
                rstrip()

        # pdb.set_trace()
        # At beginning of list of annotations.
        line = annot_file.readline().rstrip()
        assert 'annotations' in line
        line = annot_file.readline().rstrip()
        while line != '':
            behavior = line
            behaviors.append(behavior)
            line = annot_file.readline().rstrip()

        # At the start of the sequence of channels
        line = annot_file.readline()
        while line != '':
            # Strip the whitespace.
            line = line.rstrip()

            assert ('----------' in line)
            channel_name = line.rstrip('-')
            channel_name = channel_name[:3] # sloppy fix for now, to get simplified channel name-----------------------
            channel_names.append(channel_name)

            behaviors_framewise = [''] * end_frame
            line = annot_file.readline().rstrip()
            while '---' not in line:

                # If we've reached EOF (end-of-file) break out of this loop.
                if line == '':
                    break

                # Now get rid of newlines and trailing spaces.
                line = line.rstrip()

                # If this is a blank
                if line == '':
                    line = annot_file.readline()
                    continue

                # Now we're parsing the behaviors
                if '>' in line:
                    # print(line)
                    curr_behavior = line[1:]
                    # Skip table headers.
                    annot_file.readline()
                    line = annot_file.readline().rstrip()

                # Split it into the relevant numbers
                start_stop_duration = line.split()

                # Collect the bout info.
                if all('.' not in s for s in start_stop_duration):
                    bout_start = int(start_stop_duration[0])
                    bout_end = int(start_stop_duration[1])
                    bout_duration = int(start_stop_duration[2])
                elif len(timestamps)!=0:
                    bout_start = np.where(timestamps==start_stop_duration[0])[0][0]
                    bout_end =  np.where(timestamps==start_stop_duration[1])[0][0]
                    bout_duration = bout_end-bout_start
                else:
                    bout_start = int(round(float(start_stop_duration[0])*framerate))
                    bout_end = int(round(float(start_stop_duration[1])*framerate))
                    bout_duration = bout_end-bout_start

                # Store it in the appropriate place.
                behaviors_framewise[(bout_start-1):bout_end] = [curr_behavior] * (bout_duration+1)

                line = annot_file.readline()

                # end of channel
            channel_dict[channel_name] = behaviors_framewise

        # for now, we'll just merge kept channels together, in order listed. this can cause behaviors happening in
        # earlier channels to be masked by other behaviors in later channels, so down the line we should change this to
        # do a smart-merge based on what behaviors we're looking for
        behFlag = 0
        changed_behavior_list = ['other'] * end_frame
        for ch in useChannel:
            if (ch in channel_dict):
                chosen_behavior_list = channel_dict[ch]
                if not(behFlag):
                    changed_behavior_list = [annotated_behavior if annotated_behavior != '' else 'other' for annotated_behavior in
                                             chosen_behavior_list]
                    behFlag=1
                else:
                    changed_behavior_list = [anno[0] if anno[1] == '' else anno[1] for anno in zip(changed_behavior_list,chosen_behavior_list)]

        ann_dict = {
            'keys': keys,
            'behs': behaviors,
            'nstrm': len(channel_names),
            'nFrames': end_frame,
            'behs_frame': changed_behavior_list
        }
        return ann_dict
