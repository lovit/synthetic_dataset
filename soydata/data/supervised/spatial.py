import numpy as np
from ..base import make_radial


def make_two_layer_radial(n_samples_per_clusters=100, n_classes=2,
    n_clusters_per_class=3, gap=0.0, equal_proportion=True, seed=None):

    """
    Arguments
    ----------
    n_samples_per_class : int, optional (default=100)
        The number of points of a class.
    n_classes : int, optional (default=2)
        The number of spiral
    n_clusters_per_class : int, optional (default=3)
        The number of clusters of each class
    gap : float, optional (default=0.0)
        The gap between adjacent clusters
        It should be bounded in [0, 1)
    equal_proportion : Boolean, optional (default=True)
        Equal maximum radius for each section
    seed : int or None
        Random seed

    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    color : array of shape [n_samples, n_classes]
        The integer labels for class membership of each sample.

    Usage
    -----
        >>> from soydata.data.supervised import make_two_layer_radial
        >>> from soydata.visualize import scatterplot

        >>> X, labels = make_two_layer_radial()
        >>> scatterplot(X, labels=labels)
    """
    np.random.seed(seed)

    X_0, labels_0 = make_radial(
        n_samples_per_clusters, n_classes, n_clusters_per_class,
        gap, equal_proportion, radius_min=0.1, radius_scale=1)
    X_1, labels_1 = make_radial(
        n_samples_per_clusters, n_classes, n_clusters_per_class,
        gap, equal_proportion, radius_min=1 * (1 + gap), radius_scale=1)

    labels_1[:-n_samples_per_clusters] = labels_1[n_samples_per_clusters:]
    labels_1[-n_samples_per_clusters:] = 0

    X = np.concatenate((X_0, X_1))
    labels = np.concatenate((labels_0, labels_1))
    return X, labels
