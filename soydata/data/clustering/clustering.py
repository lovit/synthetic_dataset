import numpy as np
from ..base import check_range
from ..base import make_circle


def make_circular_clusters(n_clusters=8, size_min=100, size_max=120, r_min=0.1, r_max=0.2,
    equal_density=False, noise=0.0, centers=None, seed=None):

    """
    Arguments
    ---------
    n_clusters : int
        Number of clusters
    size_min : int
        Minimum number of samples in a cluster
    size_max : int
        Maximum number of samples in a cluster
    r_min : float
        Minimum radius of clusters
    r_max : float
        Maximum radius of clusters
    equal_density : Boolean
        If True, the density in the cluster is equal.
        Else, the closer to center, the denser.
    noise : float
        Proportion of noise and cluster data
    centers : list of tuple or None
        User-specified center of clusters
    seed : int or None
        Random seed

    Returns
    -------
    X : numpy.ndarray
        Samples = (n_samples, dim)
    labels : numpy.ndarray
        The integer labels for class membership of each sample.
        Shape = (n_samples,)

    Usage
    -----
        >>> from soydata.data.clustering import make_circular_clusters
        >>> from soydata.visualize import scatterplot

        >>> X, labels = X, labels = make_circular_clusters(n_clusters=10,
                r_min=0.05, r_max=0.1, equal_density=True, noise=0.05, seed=0)
        >>> scatterplot(X, labels=labels)
    """

    np.random.seed(seed)
    r_min, r_max = check_range(r_min, r_max)

    if hasattr(r_max, '__len__'):
        if len(r_max) < n_clsuters:
            raise ValueError('Set `r_max` with number or same length of list')
        radius = r_max[:n_clusters]
    else:
        radius = np.random.random_sample(n_clusters) * (r_max - r_min) + r_min

    if hasattr(centers, '__len__'):
        if len(r_max) < n_clsuters:
            raise ValueError('Set `centers` with None or same length of list')
        centers = np.asarray(centers[:n_clusters])
    else:
        centers = np.random.random_sample((n_clusters, 2))
    if centers.shape[1] != 2:
        raise ValueError('The dimension of center must be 2')

    sizes = np.random.randint(low=size_min, high=size_max, size=(n_clusters,))

    X = []
    for size, center, r in zip(sizes, centers, radius):
        Xi = make_circle(size, center[0], center[1], r, equal_density)
        X.append(Xi)
    labels = [i for i, Xi in enumerate(X) for _ in range(Xi.shape[0])]

    X = np.vstack(X)
    labels = np.asarray(labels, dtype=np.int)

    if noise > 0:
        n_noise = int(X.shape[0] * noise)
        factor_x = 1.1 * (X[:,0].max() - X[:,0].min())
        factor_y = 1.1 * (X[:,1].max() - X[:,1].min())
        noise = np.random.random_sample((n_noise, 2))
        noise[:,0] = noise[:,0] * factor_x
        noise[:,1] = noise[:,1] * factor_y
        X = np.vstack([X, noise])
        labels = np.concatenate([labels, -np.ones(n_noise, dtype=np.int)])
    return X, labels

def make_rectangular_clusters(n_clusters=8, dim=2, size_min=10, size_max=15, volume=0.2, seed=None):
    """
    Arguments
    ---------
    n_clusters : int
        Number of clusters
    dim : int
        Dimension of data
    size_min : int
        Minimum number of samples in a cluster
    size_max : int
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
        The integer labels for class membership of each sample.
        Shape = (n_samples,)

    Usage
    -----
        >>> X, labels = make_rectangular_clusters(n_clusters=8, min_size=10, max_size=15, volume=0.2, seed=0)
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
    labels = np.asarray(labels, dtype=np.int)
    return X, labels
