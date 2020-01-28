import numpy as np
from ..base import make_radial
from ..base import generate_range


def make_two_layer_radial(n_samples_per_cluster=100, n_classes=2,
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
    X : numpy.ndarray
        The generated samples, shape = (n_samples, 2)
    labels : numpy.ndarray
        The integer labels [0, 1, ..., n_classes) for class membership of each sample.
        Shape = (n_samples,)

    Usage
    -----
        >>> from soydata.data.classification import make_two_layer_radial
        >>> from soydata.visualize import scatterplot

        >>> X, labels = make_two_layer_radial()
        >>> scatterplot(X, labels=labels)
    """
    np.random.seed(seed)

    X_0, labels_0 = make_radial(
        n_samples_per_cluster, n_classes, n_clusters_per_class,
        gap, equal_proportion, radius_min=0.1, radius_scale=1)
    X_1, labels_1 = make_radial(
        n_samples_per_cluster, n_classes, n_clusters_per_class,
        gap, equal_proportion, radius_min=1 * (1 + gap), radius_scale=1)

    labels_1[:-n_samples_per_cluster] = labels_1[n_samples_per_cluster:]
    labels_1[-n_samples_per_cluster:] = 0

    X = np.concatenate((X_0, X_1))
    labels = np.concatenate((labels_0, labels_1))
    return X, labels

def make_complex_rectangulars(n_samples=3000, n_classes=2,
    volume=0.5, n_rectangulars=10, seed=None):

    """
    Arguments
    ---------
    n_samples : int
        The number of generated sample data in 2D [(0, 1), (0, 1)]
    n_classes : int
        The number of rectangular classes
    volume : float
        The volume of randomly generated rectangulars
    n_rectangulars : int
        The number of randomly generated rectangulars
    seed : int or None
        Random seed

    Returns
    -------
    X : numpy.ndarray
        The generated samples, shape = (n_samples, 2)
    labels : numpy.ndarray
        The integer labels for class membership of each sample.
        Shape = (n_samples,)

    Usage
    -----
        >>> from soydata.data.classification import make_complex_rectangulars
        >>> from soydata.visualize import scatterplot

        >>> X, labels = make_complex_rectangulars(n_samples=5000, n_rectangulars=20, volume=0.5, seed=0)
        >>> p = scatterplot(X, labels=labels)
    """

    np.random.seed(seed)
    X = np.random.random_sample((n_samples, 2))
    labels = np.zeros(n_samples)
    for label in range(n_rectangulars):
        x_min, x_max = generate_range(volume)
        y_min, y_max = generate_range(volume)
        indices = np.where(
            (x_min <= X[:,0]) & (X[:,0] <= x_max) &
            (y_min <= X[:,1]) & (X[:,1] <= y_max)
        )[0]
        label = label % n_classes
        labels[indices] = label
    return X, labels
