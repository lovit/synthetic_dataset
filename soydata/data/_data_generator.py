import numpy as np
from sklearn.utils import check_random_state


def make_moons(n_samples=100, 
        xy_ratio=1.0, x_gap=0.0, y_gap=0.0, noise=None):

    """Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points generated.
    xy_ratio : float, optional (default=1.0)
        ratio of y range over x range. It should be positive
    x_gap : float, optional (default=0.0)
        Gap between y-axis center of two moons. 
        It should be larger than -0.3
    y_gap : float, optional (default=0.0)
        Gap between x-axis center of two moons. 
        It should be larger than -0.3
    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels (0 or 1) for class membership of each sample.
    References
    ----------
    .. [1] scikit-learn sklearn.dataset.samples_generator.make_moons
    """

    assert xy_ratio > 0
    assert -0.3 <= x_gap
    assert -0.3 <= y_gap

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    generator = check_random_state(None)

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out)) - x_gap
    outer_circ_y = xy_ratio * np.sin(np.linspace(0, np.pi, n_samples_out)) + y_gap
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in)) + x_gap
    inner_circ_y = xy_ratio * (1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - (.5 + y_gap))

    X = np.vstack((np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y))).T
    y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
                   np.ones(n_samples_in, dtype=np.intp)])

    if noise is not None:
        noise = generator.normal(scale=noise, size=X.shape)
        noise[:,1] = noise[:,1] * xy_ratio
        X += noise

    return X, y

def make_spiral(n_samples_per_class=100, n_classes=2,
        n_rotations=3, gap_between_spiral=0.0, 
        gap_between_start_point=0.0, equal_interval=True,                
        noise=None):

    """Parameters
    ----------
    n_samples_per_class : int, optional (default=100)
        The number of points of a class.
    n_classes : int, optional (default=2)
        The number of spiral
    n_rotations : int, optional (default=3)
        The rotation number of spiral
    gap_between_spiral : float, optional (default=0.0)
        The gap between two parallel lines
    gap_betweein_start_point : float, optional (default=0.0)
        The gap between spiral origin points
    equal_interval : Boolean, optional (default=True)
        Equal interval on a spiral line if True.
        Else their intervals are proportional to their radius.
    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    color : array of shape [n_samples, n_classes]
        The integer labels for class membership of each sample.
    """

    assert 1 <= n_classes and type(n_classes) == int

    generator = check_random_state(None)

    X_array = []
    theta = 2 * np.pi * np.linspace(0, 1, n_classes + 1)[:n_classes]

    for c in range(n_classes):

        t_shift = theta[c]
        x_shift = gap_between_start_point * np.cos(t_shift)
        y_shift = gap_between_start_point * np.sin(t_shift)

        if equal_interval:
            t = n_rotations * np.pi * (2 * generator.rand(1, n_samples_per_class) ** (1/2))
        else:
            t = n_rotations * np.pi * (2 * generator.rand(1, n_samples_per_class))

        x = (1 + gap_between_spiral) * t * np.cos(t + t_shift) + x_shift
        y = (1 + gap_between_spiral) * t * np.sin(t + t_shift) + y_shift

        X = np.concatenate((x, y))

        if noise is not None:
            X += generator.normal(scale=noise, size=X.shape)
        
        X = X.T
        X_array.append(X)

    X = np.concatenate(X_array)
    color = np.asarray([c for c in range(n_classes) for _ in range(n_samples_per_class)])

    return X, color

def make_swiss_roll(n_samples=100, n_rotations=1.5, 
        gap=0, thickness=0.0, width=10.0):

    """Generate a swiss roll dataset.
    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples in swiss roll
    n_rotations : float, optional (default=1.5)
        The number of turns
    gap : float, optional (default=1.0)
        The gap between two roll planes
    thickness : float, optional (default=0.0)
        The thickness of roll plane
    
    noise : float, optional (default=0.0)
        The standard deviation of the gaussian noise.
    Returns
    -------
    X : array of shape [n_samples, 3]
        The points.
    color : array of shape [n_samples]
        The normalized univariate position of the sample according to the main 
        dimension of the points in the manifold. Its scale is bounded in [0, 1]

    References
    ----------
    .. [1] scikit-learn sklearn.dataset.samples_generator.make_swiss_roll
    """
    generator = check_random_state(None)

    t = n_rotations * np.pi * (1 + 2 * generator.rand(1, n_samples))
    x = (1 + gap) * t * np.cos(t)
    y = width * (generator.rand(1, n_samples) - 0.5)
    z = (1 + gap) * t * np.sin(t)

    X = np.concatenate((x, y, z))
    X += thickness * generator.randn(3, n_samples)
    X = X.T
    t = np.squeeze(t)
    color = (t - t.min()) / (t.max() - t.min())

    return X, color

def make_radial(n_samples_per_sections=100, n_classes=2, 
        n_sections_per_class=3, gap=0.0, equal_proportion=True,
        radius_min=0.1, radius_base=1.0, radius_variance=0.0):

    """Parameters
    ----------
    n_samples_per_class : int, optional (default=100)
        The number of points of a class.
    n_classes : int, optional (default=2)
        The number of spiral
    n_sections_per_class : int, optional (default=3)
        The number of sections of each class
    gap : float, optional (default=0.0)
        The gap between adjacent sections
        It should be bounded in [0, 1)
    equal_proportion : Boolean, optional (default=True)
        Equal maximum radius for each section
    radius_min : float, optional (default=0.1)
        Minimum radius of a point
    radius_base : float, optional (default=1.0)
        Average radius of points in a section
    radius_variance : float, optional (default=0.0)
        Variance in maximum radius of sections
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    color : array of shape [n_samples, n_classes]
        The integer labels for class membership of each sample.
    """

    assert 0 <= gap < 1

    if equal_proportion:
        theta = 2 * np.pi * np.linspace(
            0, 1, n_classes * n_sections_per_class + 1)
    else:
        theta = np.cumsum(np.linspace(
            0, 1, n_classes * n_sections_per_class + 1))
        theta = 2 * np.pi * (theta - theta[0]) / (theta[-1] - theta[0])

    radius = radius_base * (1 + radius_variance * np.random.rand(
        n_sections_per_class * n_classes).reshape(-1))

    X_array = []
    color_array = []

    # for each section
    for s in range(n_sections_per_class * n_classes):
        t_begin = theta[s]
        t_end = theta[s+1]
        if gap > 0:
            t_begin += (t_end - t_begin) * gap
            t_end -= (t_end - t_begin) * gap
        
        t = t_begin + (t_end - t_begin) * np.random.rand(
            1, n_samples_per_sections)
        r = np.diag(radius_min + radius[s] * (np.random.rand(
            1, n_samples_per_sections) ** (1/2))[0])
        x = np.cos(t)
        y = np.sin(t)
        X = np.concatenate((x, y))
        X = X.dot(r)
        X = X.T
        
        color = np.asarray([s % n_classes]
                           * n_samples_per_sections)
        
        X_array.append(X)
        color_array.append(color)
    
    X = np.concatenate(X_array)
    color = np.concatenate(color_array)
    return X, color

def make_two_layer_radial(n_samples_per_sections=100, n_classes=2, 
        n_sections_per_class=3, gap=0.0, equal_proportion=True):

    """Parameters
    ----------
    n_samples_per_class : int, optional (default=100)
        The number of points of a class.
    n_classes : int, optional (default=2)
        The number of spiral
    n_sections_per_class : int, optional (default=3)
        The number of sections of each class
    gap : float, optional (default=0.0)
        The gap between adjacent sections
        It should be bounded in [0, 1)
    equal_proportion : Boolean, optional (default=True)
        Equal maximum radius for each section
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    color : array of shape [n_samples, n_classes]
        The integer labels for class membership of each sample.
    """

    X_0, color_0 = make_radial(
        n_samples_per_sections, n_classes, n_sections_per_class,
        gap, equal_proportion, radius_min=0.1, radius_base=1)
    X_1, color_1 = make_radial(
        n_samples_per_sections, n_classes, n_sections_per_class,
        gap, equal_proportion, radius_min=1 * (1 + gap), radius_base=1)

    color_1[:-n_samples_per_sections] = color_1[n_samples_per_sections:]
    color_1[-n_samples_per_sections:] = 0

    X = np.concatenate((X_0, X_1))
    color = np.concatenate((color_0, color_1))
    return X, color
