import numpy as np

def precision(predicted, actual):
    '''Returns precision of categorical predictions per category
    Precision is defined as the ratio of predicted true values that are
    actually true in the ground truth. The formula is:
    True Positive / (True Positive + False Positive)

    Arguments
        - predicted: numpy array of predicted values, with each row
        being a sample and each column being a one hot encoded feature
        - actual: numpy array of ground truth, matching the dimensions of
        our predicted array

    Returns
        - Precision: a list of the precisions per feature, indexes match
        with those of features passed on with predicted/actual arrays
    ''''
    precision = []
    for i in range(predicted.shape[1]):
        is_i = i == np.argmax(predicted, axis=1)
        is_correct = i == np.argmax(actual, axis=1)
        num = np.sum(is_correct & is_i)
        denom = np.sum(is_i)
        precision.append(float(num)/float(denom))
    return(precision)

def recall(predicted, actual):
    '''Returns recall of categorical predictions per category
    Precision is defined as the ratio of ground truth true values that are
    predicted correctly. The formula is:
    True Positive / (True Positive + False Negative)

    Arguments
        - predicted: numpy array of predicted values, with each row
        being a sample and each column being a one hot encoded feature
        - actual: numpy array of ground truth, matching the dimensions of
        our predicted array

    Returns
        - Precision: a list of the recalls per feature, indexes match
        with those of features passed on with predicted/actual arrays
    ''''

    recall = []
    for i in range(predicted.shape[1]):
        is_i = i == np.argmax(actual, axis=1)
        is_correct = np.argmax(predicted, axis=1) == np.argmax(actual, axis=1)
        num = np.sum(is_i & is_correct)
        denom = np.sum(is_i)

        recall.append(float(num)/float(denom))
    return recall

def stats_of(X):
    '''Returns some simple statistics on our training data per feature

    Arguments
        - X: list of training data, with each value being an numpy arrays
        of data from one sample (will take what is returned fron load_data)

    Returns
        - stats: a dictionary containing the max, min, mean, and standard
        deviation statisticts. each key is the label of a statistic (with
        standard deviation shortened to sd) and each value associated with
        a key is an numpy array of the value of the statistic per feature
    '''
    X_merged = np.concatenate(X, axis = 0)
    stats = {}
    stats["max"] = np.amax(X_merged, axis = 0)
    stats["min"] = np.amin(X_merged, axis = 0)
    stats["mean"] = np.mean(X_merged, axis = 0)
    stats["sd"] = np.std(X_merged, axis = 0)
    return stats

def normalize(X, stats):
    '''Normalizes our training data to the range [0, 1]

    Arguments
        - X: list of training data, with each value being an numpy arrays
        of data from one sample (will take what is returned fron load_data)
        - stats: Dictionary of stats as returned by stats_of(X)

    Returns
        - X: normalized X as specified
    '''
    span = stats["max"] - stats["min"]
    for i in range(len(X)):
        X[i] = (stats["max"] - X[i]) / span
    return X
