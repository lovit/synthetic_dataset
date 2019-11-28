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
