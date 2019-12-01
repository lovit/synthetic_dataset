import numpy as np
from sklearn.utils import check_random_state


def make_swiss_roll(n_samples=100, n_rotations=1.5, gap=0,
    thickness=0.0, width=10.0, discretize=True, seed=None):

    """
    Arguments
    ---------
    n_samples : int (default=100)
        The number of samples in swiss roll
    n_rotations : float (default=1.5)
        The number of turns
    gap : float (default=1.0)
        The gap between adjacent roll-planes
    thickness : float (default=0.0)
        The thickness of roll plane
    width : float (default=0.0)
        The scale of y axis
    descretize : Boolean
        If True, it returns color as integer which is bounded in [0, 256)

    Returns
    -------
    X : numpy.ndarray
        Shape = (n_samples, 3)
    color : numpy.ndarray
        Shape = (n_samples,)
        The normalized univariate position of the sample according to the main
        dimension of the points in the manifold. Its scale is bounded in [0, 1]

    References
    ----------
    [1] scikit-learn sklearn.dataset.samples_generator.make_swiss_roll
    """
    generator = check_random_state(seed)

    t = n_rotations * np.pi * (1 + 2 * generator.rand(1, n_samples))
    x = (1 + gap) * t * np.cos(t)
    y = width * (generator.rand(1, n_samples) - 0.5)
    z = (1 + gap) * t * np.sin(t)

    X = np.concatenate((x, y, z))
    X += thickness * generator.randn(3, n_samples)
    X = X.T
    t = np.squeeze(t)
    color = (t - t.min()) / (t.max() - t.min())

    if discretize:
        color = np.asarray([int(256 * c) for c in color], dtype=np.int)

    return X, color
