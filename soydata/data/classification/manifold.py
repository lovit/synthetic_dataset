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
    X : numpy.ndarray        
        The generated samples, shape = (n_samples, 2)
    labels : numpy.ndarray
        The integer labels (0 or 1) for class membership of each sample.
        Shape = (n_samples,)

    Usage
    -----
        >>> from soydata.data.classification import make_moons
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

def make_spiral(n_samples_per_class=300, n_classes=2, n_rotations=3, gap_between_spiral=0.0,
    gap_between_start_point=0.0, equal_interval=True, noise=None, seed=None):

    """
    Arguments
    ---------
    n_samples_per_class : int (default=300)
        The number of points of a class.
    n_classes : int (default=2)
        The number of spiral
    n_rotations : int (default=3)
        The rotation number of spiral
    gap_between_spiral : float (default=0.0)
        The gap between two parallel lines
    gap_betweein_start_point : float (default=0.0)
        The gap between spiral origin points
    equal_interval : Boolean (default=True)
        Equal interval on a spiral line if True.
        Else their intervals are proportional to their radius.
    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.
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
        >>> from soydata.data.classification import make_spiral
        >>> from soydata.visualize import scatterplot

        >>> X, labels = make_spiral(noise=0.5, n_rotations=2)
        >>> scatterplot(X, labels=labels)
    """

    assert 1 <= n_classes and type(n_classes) == int

    generator = check_random_state(None)

    X = []
    theta = 2 * np.pi * np.linspace(0, 1, n_classes + 1)[:n_classes]

    for c in range(n_classes):

        t_shift = theta[c]
        x_shift = gap_between_start_point * np.cos(t_shift)
        y_shift = gap_between_start_point * np.sin(t_shift)

        power = 0.5 if equal_interval else 1.0
        t = n_rotations * np.pi * (2 * generator.rand(1, n_samples_per_class) ** (power))
        x = (1 + gap_between_spiral) * t * np.cos(t + t_shift) + x_shift
        y = (1 + gap_between_spiral) * t * np.sin(t + t_shift) + y_shift
        Xc = np.concatenate((x, y))

        if noise is not None:
            Xc += generator.normal(scale=noise, size=Xc.shape)

        Xc = Xc.T
        X.append(Xc)

    X = np.concatenate(X)
    labels = np.asarray([c for c in range(n_classes) for _ in range(n_samples_per_class)])

    return X, labels
