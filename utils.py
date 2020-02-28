import numpy as np

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def precision(predicted, actual, is_one_hot = False):
    '''Returns precision of categorical predictions per category
    Precision is defined as the ratio of predicted true values that are
    actually true in the ground truth. The formula is:
    True Positive / (True Positive + False Positive)

    Arguments
        - predicted: array of predicted values
        - actual: array of ground truth
        - is_one_hot: Bool indicating if your predictions and actuals are
        one hot encoded (numpy array of predicted values, with each row
        being a sample and each column being a one hot encoded feature)

    Returns
        - Precision: a list of the precisions per feature
    '''
    if is_one_hot:
        predicted = np.argmax(predicted, axis=1)
        actual = np.argmax(actual, axis=1)
    precision = []
    for i in range(np.max(actual) + 1):
        predicted_i = i == predicted
        actual_i = i == actual
        num = np.sum(predicted_i & actual_i)
        denom = np.sum(predicted_i)
        if (denom == 0):
            precision.append(np.nan)
        else:
            precision.append(float(num)/float(denom))
    return(precision)

def recall(predicted, actual, is_one_hot = False):
    '''Returns recall of categorical predictions per category
    Precision is defined as the ratio of ground truth true values that are
    predicted correctly. The formula is:
    True Positive / (True Positive + False Negative)

    Arguments
        - predicted: array of predicted values
        - actual: array of ground truth
        - is_one_hot: Bool indicating if your predictions and actuals are
        one hot encoded (numpy array of predicted values, with each row
        being a sample and each column being a one hot encoded feature)

    Returns
        - Precision: a list of the recalls per feature
    '''
    if is_one_hot:
        predicted = np.argmax(predicted, axis=1)
        actual = np.argmax(actual, axis=1)

    recall = []
    for i in range(np.max(actual) + 1):
        predicted_i = i == predicted
        actual_i = i == actual
        num = np.sum(predicted_i & actual_i)
        denom = np.sum(actual_i)
        if (denom == 0):
            recall.append(np.nan)
        else:
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
