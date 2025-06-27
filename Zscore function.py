import numpy as np

def zscore_normalize_features(x):
    """
    computes x's zscore normalized by column

    :param x: data, m examples with n features
    :return:
        x_norm: data normalized by column
        mu: mean of each feature
        sigma: standard deviation of each feature
    """

    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x_norm = (x - mu) / sigma

    return x_norm, mu, sigma
