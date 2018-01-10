import numpy as np
from scipy.stats import spearmanr

def mean_score(scores):
    """ calculate mean score for AVA dataset
    :param scores:
    :return: row wise mean score if scores contains multiple rows, else
             a single mean score
    """
    si = np.arange(1, 11, 1).reshape(1,10)
    mean = np.sum(scores * si, axis=1)
    if mean.shape==(1,):
        mean = mean[0]
    return mean


def std_score(scores):
    """ calculate standard deviation of scores for AVA dataset
    :param scores:
    :return: row wise standard deviations if scores contains multiple rows,
             else a single standard deviation
    """
    si = np.arange(1, 11, 1).reshape(1,10)
    mean = mean_score(scores).reshape(-1,1)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores, axis=1))
    if std.shape==(1,):
        std = std[0]
    return std


def srcc(y_test, y_pred):
    """ calculate spearman's rank correlation coefficient
    :param y_test: the human ratings (width 10)
    :param y_pred: the predicted ratings (width 10)
    :return:
    """
    rho, pValue = spearmanr(mean_score(y_test), mean_score(y_pred))
    return rho
