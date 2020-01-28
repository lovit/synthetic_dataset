import numpy as np


def check_range(value_1, value_2):
    return min(value_1, value_2), max(value_1, value_2)

def make_circle(n_samples=100, center_x=0.0, center_y=0.0, r_max=0.5, equal_density=False, seed=None):
    """
    Arguments
    ---------
    n_samples : int
    center_x : float
        X coordinate of center
    center_y : float
        Y coordinate of center
    r_max : float
        Maximum radius
    equal_density : Boolean
        If True, the density in the cluster is equal.
        Else, the closer to center, the denser.
    seed : int or None
        Random seed

    Returns
    -------
    X : numpy.ndarray
        The generated samples, shape = (n_samples, 2)
    """

    np.random.seed(seed)
    t = np.random.random_sample(n_samples) * 2 * np.pi
    r = np.random.random_sample(n_samples)
    if equal_density:
        r = np.sqrt(r)
    r *= r_max
    x = np.cos(t) * r + center_x
    y = np.sin(t) * r + center_y
    X = np.vstack([x, y]).T
    return X

def make_rectangular(n_samples=100, x_min=0, x_max=1, y_min=0, y_max=1, seed=None):
    """
    Arguments
    ---------
    n_samples : int
    x_min : float
        Available minimum value of x in rectangular
    x_max : float
        Available maximum value of x in rectangular
    y_min : float
        Available minimum value of y in rectangular
    y_max : float
        Available maximum value of y in rectangular
    seed : int or None
        Random seed

    Returns
    -------
    X : numpy.ndarray
        The generated samples, shape = (n_samples, 2)
    """

    np.random.seed(None)
    x_min, x_max = check_range(x_min, x_max)
    y_min, y_max = check_range(y_min, y_max)

    X = np.random.random_sample((n_samples, 2))
    X[:,0] = x_min + X[:,0] * (x_max - x_min)
    X[:,1] = y_min + X[:,1] * (y_max - y_min)
    return X

def make_triangular(n_samples=100, upper=True, positive_direction=True,
    x_min=0, x_max=2, y_min=0, y_max=1, seed=None):
    """
    Arguments
    ---------
    n_samples : int
    upper : Boolean
        If True, it generated points located in the upper triangle.
    positive_direction : Boolean
        If True, the slope of triangular is (y_max - y_min) / (x_max - x_min)
        Else, the slope is (y_min - y_max) / (x_max - x_min)
    x_min : float
        Available minimum value of x in triangular
    x_max : float
        Available maximum value of x in triangular
    y_min : float
        Available minimum value of y in triangular
    y_max : float
        Available maximum value of y in triangular
    seed : int or None
        Random seed

    Returns
    -------
    X : numpy.ndarray
        The generated samples, shape = (n_samples, 2)
    """

    np.random.seed(None)
    grad = (y_max - y_min) / (x_max - x_min)
    if positive_direction:
        is_upper = lambda q: q[1] - y_min > grad * (q[0] - x_min)
    else:
        is_upper = lambda q: (q[1] - y_min) < -grad * (q[0] - x_max)

    X = make_rectangular(int(2.5 * n_samples), x_min, x_max, y_min, y_max, seed)

    indices = np.where(np.apply_along_axis(is_upper, 1, X) == upper)
    return X[indices][:n_samples]

def make_radial(n_samples_per_cluster=100, n_classes=2, n_clusters_per_class=3,
    gap=0.0, equal_proportion=True, radius_min=0.1, radius_scale=1.0,
    radius_variance=0.0, seed=None):

    """
    Arguments
    ---------
    n_samples_per_cluster : int (default=100)
        The number of points of a class.
    n_classes : int (default=2)
        The number of spiral
    n_clusters_per_class : int (default=3)
        The number of clusters of each class
    gap : float (default=0.0)
        The gap between adjacent clusters
        It should be bounded in [0, 1)
    equal_proportion : Boolean (default=True)
        Equal maximum radius for each cluster
    radius_min : float (default=0.1)
        Minimum radius of a point
    radius_scale : float (default=1.0)
        Average radius of points in a cluster
    radius_variance : float (default=0.0)
        Variance in maximum radius of clusters
    seed : int or None
        Random seed

    Returns
    -------
    X : numpy.ndarray
        The generated samples, shape = (n_samples, 2)
    labels : numpy.ndarray
        The integer labels for class membership of each sample.
        Shape = (n_samples,)
    """

    np.random.seed(seed)

    assert 0 <= gap < 1

    if equal_proportion:
        theta = 2 * np.pi * np.linspace(0, 1, n_classes * n_clusters_per_class + 1)
    else:
        theta = np.cumsum(np.linspace(0, 1, n_classes * n_clusters_per_class + 1))
        theta = 2 * np.pi * (theta - theta[0]) / (theta[-1] - theta[0])

    radius = radius_scale * (1 + radius_variance * np.random.rand(
        n_clusters_per_class * n_classes).reshape(-1))

    X = []
    labels = []

    # for each cluster
    for s in range(n_clusters_per_class * n_classes):
        t_begin = theta[s]
        t_end = theta[s+1]
        if gap > 0:
            t_begin += (t_end - t_begin) * gap
            t_end -= (t_end - t_begin) * gap
        t = t_begin + (t_end - t_begin) * np.random.rand(1, n_samples_per_cluster)
        r = np.diag(radius_min + radius[s] * (np.random.rand(1, n_samples_per_cluster) ** (1/2))[0])
        x = np.cos(t)
        y = np.sin(t)
        Xs = np.concatenate((x, y))
        Xs = Xs.dot(r)
        Xs = Xs.T

        label = np.asarray([s % n_classes] * n_samples_per_cluster)
        X.append(Xs)
        labels.append(label)

    X = np.concatenate(X)
    labels = np.concatenate(labels)

    return X, labels

def generate_range(scale):
    base = np.random.random_sample(1) * 0.5
    ranges = np.random.random_sample(3) * scale
    ranges.sort()
    min = ranges[0] + base
    max = ranges[-1] + base
    return min, max

def rotate(xy, radians):
    """
    Arguments
    ---------
    xy : numpy.ndarray
        Shape = (n_data, 2)
    radians : float
        Radian to rotate

    Returns
    -------
    xy : numpy.ndarray
        Rotated 2d array, shape = (n_data, 2)
    """
    x = xy[:,0]
    y = xy[:,1]
    xx = x * np.cos(radians) + y * np.sin(radians)
    yy = -x * np.sin(radians) + y * np.cos(radians)
    xy = np.vstack([xx, yy]).T
    return xy
