import numpy as np


def make_linear_regression_data(n_data=100, a=1.0, b=1.0,
    noise=1.0, x_range=(-10.0, 10.0), seed=None):
    """
    It generates artificial data for linear regression

    Arguments
    ---------
    n_data : int
        Number of generated data
    a : float
        Regression coefficient a in 'y = ax + b'
    b : float
        Interpret coefficient b in 'y = ax + b'
    noise : float
        Range of residual, e = y - (ax + b)
    x_range : tuple
        size = (float, float)
    seed : int or None
        Random seed

    Returns
    -------
    x : numpy.ndarray
        Input data sequence, shape = (n_data,)
    y : numpy.ndarray
        Output data sequence with random noise, shape = (n_data,)
    y_true : numpy.ndarray
        Truen output data sequence (regressed output), shape = (n_data,)

    Usage
    -----
        >>> x, y, _ = make_linear_regression_data()
        >>> x, y, y_true = make_linear_regression_data(
            n_data=100, a=1.0, b=1.0, noise=1.0, x_range=(-10.0, 10.0))
    """

    assert (len(x_range) == 2) and (x_range[0] < x_range[1])

    if isinstance(seed, int):
        np.random.seed(seed)

    x_scale = x_range[1] - x_range[0]
    x = np.random.random_sample(n_data) * x_scale + x_range[0]
    residual = (np.random.random_sample(n_data) - 0.5) * noise
    y = a * x + b + residual
    y_true = a * x + b

    return x, y, y_true

def make_polynomial_regression_data(n_data=100, degree=2, coefficients=None,
    coefficient_scale=3.0, noise=0.1, x_range=(-1.0, 1.0), seed=None):
    """
    It generates artificial data for linear regression

    Arguments
    ---------
    n_data : int
        Number of generated data
    degree : int
        Degree of polynomial
    coefficients : list_or_None
        Coefficients bi such that y = b0 + sum_{i=1 to degree} bi x x^i
    coefficient_scale : float
        Range of coefficients bi
    noise : float
        Range of residual, e = y - f(x)
    x_range : tuple
        size = (float, float)
    seed : int or None
        Random seed

    Returns
    -------
    x : numpy.ndarray
        Input data sequence, shape = (n_data,)
    y : numpy.ndarray
        Output data sequence with random noise, shape = (n_data,)
    y_true : numpy.ndarray
        Truen output data sequence (regressed output), shape = (n_data,)

    Usage
    -----
        >>> x, y, _ = make_linear_regression()
        >>> x, y, y_true = make_linear_regression(
            n_data=100, a=1.0, b=1.0, noise=1.0, x_range=(-10.0, 10.0))
    """

    if (not isinstance(degree, int)) or (degree < 0):
        raise ValueError(f'degree must be nonnegative integer, however input is {degree}')

    if isinstance(seed, int):
        np.random.seed(seed)

    if coefficients is None:
        coefficients = coefficient_scale * (np.random.random_sample(degree + 1) - 0.5)

    len_coef = len(coefficients)
    if len_coef != degree + 1:
        raise ValueError('The length of coefficients must be degree'\
            f'However, length is {len_coef} with degree = {degree}')

    x_scale = x_range[1] - x_range[0]
    x = np.random.random_sample(n_data) * x_scale + x_range[0]

    y_true = np.zeros(n_data)
    for p, coef in enumerate(coefficients):
        y_true = y_true + coef * np.power(x, p)
    residual = (np.random.random_sample(n_data) - 0.5) * noise
    y = y_true + residual

    return x, y, y_true