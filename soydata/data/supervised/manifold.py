import numpy as np
from sklearn.utils import check_random_state


def make_moons(n_samples=100, xy_ratio=1.0, x_gap=0.0, y_gap=0.0, noise=None, seed=None):
    """
    Arguments
    ----------
    n_samples : int (default=100)
        The total number of points generated.
    xy_ratio : float (default=1.0)
        ratio of y range over x range. It should be positive.
    x_gap : float (default=0.0)
        Gap between y-axis center of two moons.
        It should be larger than -0.3
    y_gap : float (default=0.0)
        Gap between x-axis center of two moons.
        It should be larger than -0.3
    noise : float or None (default=None)
        Standard deviation of Gaussian noise added to the data.
    seed : int or None
        Random seed

    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels (0 or 1) for class membership of each sample.

    Usage
    -----
        >>> from soydata.data.supervised import make_moons
        >>> from soydata.visualize import scatterplot

        >>> X, labels = make_moons(n_samples=1000, noise=0.1)
        >>> scatterplot(X, labels=labels)

    References
    ----------
    .. [1] scikit-learn sklearn.dataset.samples_generator.make_moons
    """

    assert xy_ratio > 0
    assert -0.3 <= x_gap
    assert -0.3 <= y_gap

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    generator = check_random_state(seed)

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out)) - x_gap
    outer_circ_y = xy_ratio * np.sin(np.linspace(0, np.pi, n_samples_out)) + y_gap
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in)) + x_gap
    inner_circ_y = xy_ratio * (1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - (.5 + y_gap))

    X = np.vstack(
        (np.append(outer_circ_x, inner_circ_x),
         np.append(outer_circ_y, inner_circ_y))
    ).T
    labels = np.hstack(
        [np.zeros(n_samples_out, dtype=np.intp),
         np.ones(n_samples_in, dtype=np.intp)]
    )

    if noise is not None:
        noise = generator.normal(scale=noise, size=X.shape)
        noise[:,1] = noise[:,1] * xy_ratio
        X += noise

    return X, labels
