# MARS-train-infer
This is a suite of scripts for training new classifiers in MARS. Contents:

- [MARS_annotation_parsers.py](##MARS_annotation_parsers.py)
- [MARS_train_test.py](##MARS_train_test.py)
- [MARS_clf_helpers.py](##MARS_clf_helpers.py)
- [MARS_ts_util.py](##MARS_ts_util.py)


##MARS_annotation_parsers.py
Contains scripts for reading and writing behavior annotations. The relevant functions are:

###`parse_annotations(fid, use_channels=[], timestamps=[])`
Reads annotations from a `*.txt` or `*.annot` file and returns them in a dictionary.
#####Inputs:
* `fid` **(string)**: path to an annotation file (`*.txt` or `*.annot`).
* `use_channels` **(list of strings)**: for `*.annot` file, names of channels to load annotations from (default: merges annotations from all channels).
* `timestamps` **(list)**: for `*.annot` files, exact timestamps of frames in the annotated `*.seq` movie, to ensure accurate conversion from annotated times to frame numbers (if not provided, will perform conversion using framerate specified in the `*.annot` file header).

#####Output:
 A **dict** with the following keys:
* `behs` **(string array)**: list of behaviors annotated for in the file (some may not actually appear).
* `nstrm` **(int)**: number of channels annotated (typically 1 or 2).
* `nFrames` **(int)**: number of frames in the annotated video.
* `behs_frame` **(string array)**: a string array of length `nFrames`. Each entry is a single behavior from `behs`, or "other" if no actions were annotated on that frame. 
* (Plus a few other keys that you can ignore)


##MARS_train_test.py
The core set of functions for creating, training, and testing behavior classifiers.

### `choose_classifier(clf_type='xgb', clf_params=dict())`
Initializes a classifier of type `clf_type` with additional parameter values set by `clf_params`. You can extend this code to include support for other kinds of classifiers, so long as they have the usual `scikit-learn` structure.

#####Inputs:
* `clf_type` **(string)**: current options are 'xgb' (XGBoost) or 'mlp' (multi-layer Perceptron)
* `clf_params` **(dict)**: optional, allows user specification of classifier parameters, otherwise a set of default values are provided by the function.

#####Output:
* `clf` **(Classifier)** an initialized classifier, currently either `XGBClassifier` or `BaggingClassifier` (bag of multi-layer Perceptrons).



### `load_data(video_path, video_list, keepLabels)`

#####Inputs:
* `video_path` **(string)**: Path to directory containing all videos.
* `video_list` **(string array)**: Which folders in that directory to load.
* `keepLabels` **(string array)**: Which behaviors to load annotations of.
* `do_wnd = (False)|True` **(bool)**: Apply temporal windowing to features before returning.
* `do_cwt = (False)|True` **(bool)**: Apply wavelet transform to features before returning.
<!-- * `ver = ([7, 8])` **(list)**: Version of MARS pose estimate to use
* `feat_type = ('top')|'top_pcf'|'front'` **(string)**: Version of MARS features to use (keep this set to 'top')
* `verbose = (0)|1` **(int)** -->


#####Outputs:
* `data` **(2D numpy array)**: a (time x features) array concatenated over all loaded videos.
* `y` **(dict of 1D numpy arrays)**: keys are the behaviors in `keepLabels`; for each key, contains a binary array of length (time) indicating the presence or absence of that behavior on each frame, concatenated over all loaded videos.
* `scaler` **(StandardScaler)**: operator that whitens data (based on statistics of the training set) prior to classification.
* `names` **(string array)**: names of the loaded features.


##MARS_clf_helpers.py

####`prf_metrics(y_tr_beh, pd_class, beh)`
Computes precision and recall for classifier output, given numpy arrays of 0's and 1's (representing presence of a behavior).

#####Inputs:
* `y_tr_beh` **(1D binary numpy array)**: ground truth annotations
* `pd_class` **(1D binary numpy array)**: predicted labels
* `beh` **(string)**: name of the behavior being detected

#####Output:

Prints precision (P), recall (R), and F1 score (F1) to the command line.

##MARS_ts_util.py
A handful of functions for cleaning up or transforming pose features. Most of these are for a step in existing MARS classifiers where annotations are cleaned up using forward/backward smoothing and/or an HMM. However this file also contains the following:

###`clean_data(data)`
Used to eliminate NaN/Inf values in features extracted from mouse pose.

#####Input:
* `data` **(2D numpy array)**: a (time x features) array of pose features.

#####Output:
* `data_clean` **(2D numpy array)**: same array, with NaN and Inf values replaced by the last value that was neither.
 
 
 ###`apply_windowing(starter_features)`
 Applies MARS-style temporal windowing to feature values: for each feature, on each frame, computes the mean, standard deviation, min, and max within a window of 3, 11, and 21 frames of the current frame (resulting in a 12-fold increase in feature count).
 #####Input:
 * `starter_features` **(2D numpy array)**: a (time x features) array of pose features.
 
 #####Output:
 * `windowed_features` **(2D numpy array)**: now a (time x 12*features) array of windowed pose features.
 
 ###`apply_wavelet_transform(starter_features)`
 A simple script for convolving pose features with a set of wavelets. Frequency range of wavelets (`scales`) is currently hard-coded in the function but this could easily be modified.
 
 #####Input:
 * `starter_features` **(3D numpy array)**: a (mouse ID x time x features) array of pose features.
 
 #####Output:
 * `transformed_features` **(2D numpy array)**: a new array, now with dimensions (time x features * (1 + len(`scales`)) * 2 * 2): each feature has been convolved with each of two wavelets that have been scaled by each value in `scales`. Feature values for the two mice are now provided in columns of `transformed_features` ie `transformed_features = [[all convolved mouse 1 features] [all convolved mouse 2 features]]`.