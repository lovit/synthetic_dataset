import numpy as np
from ..base import make_radial
from ..base import make_rectangular
from ..base import generate_range
from ..base import rotate


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

def make_multilayer_rectangulars(rec_size=100, n_layers=2,
    n_classes=2, random_label=False, rotate_radian=0, seed=None):
    """
    Arguments
    ---------
    rec_size : int
        The number of samples in a rectangular
    n_layers : int
        The number of layers. The number of rectauglars is (2 * `n_layers`)^2
    n_classes : int
        The number of classes. It is used only when `random_label` is True
    random_label : Boolean
        If True, it permutate labels
    rotate_radian : float
        If rotate_radian != 0, it rotates X
    seed : int or None
        Random seed

    Returns
    -------
    X : numpy.ndarray
        The generated samples, shape = (n_samples, 2)
    labels : numpy.ndarray
        The integer labels [0, 0, ..., 1, 1, ... 0, 0, ...]

    Usage
    -----
    Import functions

        >>> from soydata.data.classification import make_multilayer_rectangulars
        >>> from soydata.visualize import scatterplot

    To generate regular patterned data

        >>> X, labels = make_multilayer_rectangulars(rec_size=100, n_layers=2)
        >>> p = scatterplot(X, labels=labels, title='Multilayer rectangulars')

    To generate random labeled data

        >>> X, labels = make_multilayer_rectangulars(
                n_layers=5, random_label=True, n_classes=5)
        >>> p = scatterplot(X, labels=labels, title='Random-labeled multilayer rectangulars')
    """
    np.random.seed(seed)
    n_rectangulars = (2*n_layers) ** 2
    X, labels = [], []
    for y in range(-n_layers, n_layers, 1):
        for x in range(-n_layers, n_layers, 1):
            X.append(make_rectangular(n_samples=rec_size,
                x_min=x, x_max=x+1, y_min=y, y_max=y+1))
            if random_label:
                label = np.random.randint(0, n_classes)
            else:
                label = abs(y % 2 + x) % 2
            labels += [label] * rec_size
    X = np.vstack(X)
    if abs(rotate_radian) > 0:
        X = rotate(X, rotate_radian)
    return X, labels
