import numpy as np

def check_range(value_1, value_2):
    return min(value_1, value_2), max(value_1, value_2)

def make_rectangular(n_samples=100, x_min=0, x_max=1, y_min=0, y_max=1, seed=None):
    np.random.seed(None)
    x_min, x_max = check_range(x_min, x_max)
    y_min, y_max = check_range(y_min, y_max)

    X = np.random.random_sample((n_samples, 2))
    X[:,0] = x_min + X[:,0] * (x_max - x_min)
    X[:,1] = y_min + X[:,1] * (y_max - y_min)
    return X

def make_triangular(n_samples=100, upper=True, positive_direction=True,
    x_min=0, x_max=2, y_min=0, y_max=1, seed=None):

    np.random.seed(None)
    grad = (y_max - y_min) / (x_max - x_min)
    print(grad)
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
    X : array of shape [n_samples, 2]
        The generated samples.
    labels : array of shape [n_samples, n_classes]
        The integer labels for class membership of each sample.
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
