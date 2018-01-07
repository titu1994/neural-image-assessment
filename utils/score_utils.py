import numpy as np

def mean_score(scores):
    si = np.arange(1, 11, 1)
    mean = np.sum(scores * si)
    return mean

def std_score(scores):
    si = np.arange(1, 11, 1)
    mean = mean_score(scores)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return std