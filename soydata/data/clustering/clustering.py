import numpy as np

def make_rectangular_clusters(n_clusters=8, dim=2, min_size=10, max_size=15, volume=0.2, seed=None):
    """
    Arguments
    ---------
    n_clusters : int
        Number of clusters
    dim : int
        Dimension of data
    min_size : int
        Minimum number of samples in a cluster
    max_size : int
        Maximum number of samples in a cluster
    volume : float
        The volume of a cluster
    seed : int or None
        Random seed

    Returns
    -------
    X : numpy.ndarray
        Samples = (n_samples, dim)
    labels : numpy.ndarray
        The integer labels (0 or 1) for class membership of each sample.
        Shape = (n_samples,)

    Usage
    -----
        >>> X, labels = make_clusters(n_clusters=8, min_size=10, max_size=15, volume=0.2, seed=0)
    """
    np.random.seed(seed)
    X = []
    labels = []
    for label in range(n_clusters):
        size = np.random.randint(min_size, max_size+1, 1)[0]
        center = np.random.random_sample((1, dim))
        samples = center + volume * (np.random.random_sample((size, dim)) - 0.5)
        X.append(samples)
        labels += [label] * size
    X = np.vstack(X)
    labels = np.asarray(labels)
    return X, labels
