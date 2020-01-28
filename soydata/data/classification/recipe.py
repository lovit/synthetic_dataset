import numpy as np

from ..base import make_rectangular
from ..base import make_triangular


def make_predefined_data(name='decision-tree-1', n_samples=1000):
    """
    Arguments
    ---------
    name : str
        Dataset name, Valid values are following

            names = ['decision-tree-1', 'decision-tree-2']

    n_samples : int
        Number of generated samples

    Returns
    -------
    X : numpy.ndarray
        The generated samples, shape = (n_samples, 2)
    labels : numpy.ndarray
        The integer labels for class membership of each sample.
        Shape = (n_samples,)

    Usage
    -----
        >>> from soydata.data.classification import make_decision_tree_data_from_recipe
        >>> from soydata.visualize import scatterplot

        >>> X, labels = make_predefined_data('decision-tree-1', n_samples=5000)
        >>> X, labels = make_predefined_data('decision-tree-2', n_samples=5000)
        >>> p = scatterplot(X, labels=labels)
    """
    if name == 'decision-tree-1':
        recipe = [
            # (type, volume, label, x_min, x_max, y_min, y_max)
            ('rec', 25, 1, 0, 5, 0, 5),
            ('rec', 15, 0, 5, 10, 0, 3),
            ('rec', 3, 0, 5, 7, 3, 4.5),
            ('rec', 10, 1, 7, 10, 3, 6.5),
            ('rec', 1, 1, 5, 7, 4.5, 5),
            ('rec', 10, 1, 2, 7, 5, 7),
            ('rec', 4, 0, 0, 2, 5, 7),
            ('rec', 13.5, 0, 0, 4.5, 7, 10),
            ('rec', 8.5, 1, 4.5, 7, 7, 10),
            ('rec', 10, 0, 7, 10, 6.5, 10),
        ]
        return make_decision_tree_data_from_recipe(n_samples, recipe)
    elif name == 'decision-tree-2':
        # (type, volume, label, x_min, x_max, y_min, y_max)
        recipe = [
            ('rec', 32, 1, 0, 7, 0, 4),
            ('rec', 9.5, 0, 7, 10, 0, 3.5),
            ('rec', 4.5, 1, 7, 10, 3.5, 5),
            ('rec', 0.5, 1, 7.5, 8, 4, 5),
            ('rec', 7.5, 0, 7.5, 10, 5, 8),
            ('rec', 4.5, 1, 8, 10, 8, 10),
            ('rec', 1, 0, 7.5, 8, 8, 10),
            ('rec', 16.5, 0, 2, 7.5, 7, 10),
            ('rec', 12, 0, 0, 2, 4, 10),
            ('upper', 8.25, 1, 2, 7.5, 4, 7),
            ('lower', 8.25, 0, 2, 7.5, 4, 7),
        ]
        return make_decision_tree_data_from_recipe(n_samples, recipe)
    else:
        raise ValueError('Unknown datataset')

def make_decision_tree_data_from_recipe(n_samples, recipe):
    """
    Arguments
    ---------
    n_samples : int
        Number of generated samples
    recipe : list of tuple
        (type, volume, label, x_min, x_max, y_min, y_max) form
        Available types are following

            types = ['rec', 'upper', 'lower']

    Returns
    -------
    X : numpy.ndarray
        The generated samples, shape = (n_samples, 2)
    labels : numpy.ndarray
        The integer labels for class membership of each sample.
        Shape = (n_samples,)

    Usage
    -----
        >>> recipe = [
                # (type, volume, label, x_min, x_max, y_min, y_max)
                ('rec', 25, 1, 0, 5, 0, 5),
                ('rec', 15, 0, 5, 10, 0, 3),
                ('rec', 3, 0, 5, 7, 3, 4.5),
                ('rec', 10, 1, 7, 10, 3, 6.5),
                ('rec', 1, 1, 5, 7, 4.5, 5),
                ('rec', 10, 1, 2, 7, 5, 7),
                ('rec', 4, 0, 0, 2, 5, 7),
                ('rec', 13.5, 0, 0, 4.5, 7, 10),
                ('rec', 8.5, 1, 4.5, 7, 7, 10),
                ('rec', 10, 0, 7, 10, 6.5, 10),
            ]
        >>> n_samples = 1000
        >>> make_decision_tree_data_from_recipe(n_samples, recipe)
    """
    labels = []
    X = []
    for type, volume, label, x_min, x_max, y_min, y_max in recipe:
        ns = int(n_samples * volume / 100)
        if type == 'rec':
            Xs = make_rectangular(ns, x_min, x_max, y_min, y_max)
        elif type == 'upper':
            Xs = make_triangular(ns, upper=True, x_min=x_min,
                x_max=x_max, y_min=y_min, y_max=y_max)
        elif type == 'lower':
            Xs = make_triangular(ns, upper=False, x_min=x_min,
                x_max=x_max, y_min=y_min, y_max=y_max)
        else:
            raise ValueError('Profile type error. Type={}'.format(p[0]))
        X.append(Xs)
        labels += [label] * ns
    X = np.vstack(X)[:n_samples]
    labels = np.asarray(labels, dtype=np.int)[:n_samples]
    return X, labels
