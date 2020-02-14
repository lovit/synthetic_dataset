import numpy as np


def make_linear_regression_data(n_samples=100, a=1.0, b=1.0,
    noise=1.0, x_range=(-10.0, 10.0), seed=None):
    """
    It generates artificial data for linear regression

    Arguments
    ---------
    n_samples : int
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
        >>> from soydata.data.regression import make_linear_regression_data
        >>> from soydata.visualize import lineplot

        >>> x, y, _ = make_linear_regression_data()
        >>> x, y, y_true = make_linear_regression_data(
            n_data=100, a=1.0, b=1.0, noise=1.0, x_range=(-10.0, 10.0))

        >>> p = lineplot(x, y, show_inline=False, line_width=2)

    To overlay true regression line

        >>> p = lineplot(x, y_true, p=p, line_color='red')
    """

    assert (len(x_range) == 2) and (x_range[0] < x_range[1])

    if isinstance(seed, int):
        np.random.seed(seed)

    x_scale = x_range[1] - x_range[0]
    x = np.random.random_sample(n_samples) * x_scale + x_range[0]
    residual = (np.random.random_sample(n_samples) - 0.5) * noise
    y = a * x + b + residual
    y_true = a * x + b

    return x, y, y_true

def make_polynomial_regression_data(n_samples=100, degree=2,
    coefficients=None, noise=0.1, x_range=(-1.0, 1.0), seed=None):
    """
    It generates artificial data for linear regression

    Arguments
    ---------
    n_samples : int
        Number of generated data
    degree : int
        Degree of polynomial
    coefficients : list_or_None
        Coefficients bi such that y = b0 + sum_{i=1 to degree} bi x x^i
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
        >>> from soydata.data.regression import make_linear_regression_data
        >>> from soydata.visualize import lineplot

        >>> x, y, y_true = make_polynomial_regression_data(degree=5, noise=0.2, seed=1)
        >>> p = lineplot(x, y, show_inline=False, line_width=2)

    To overlay true regression line

        >>> p = lineplot(x, y_true, p=p, line_color='red')
    """

    if (not isinstance(degree, int)) or (degree < 0):
        raise ValueError(f'degree must be nonnegative integer, however input is {degree}')

    if isinstance(seed, int):
        np.random.seed(seed)

    if coefficients is None:
        sign = (np.random.randint(2, size=degree + 1) * 2 - 1)
        coefficients = np.random.random_sample(degree + 1) + 0.5
        coefficients *= sign

    len_coef = len(coefficients)
    if len_coef != degree + 1:
        raise ValueError('The length of coefficients must be degree'\
            f'However, length is {len_coef} with degree = {degree}')

    x_scale = x_range[1] - x_range[0]
    x = np.random.random_sample(n_samples) * x_scale + x_range[0]

    y_true = np.zeros(n_samples)
    for p, coef in enumerate(coefficients):
        y_true = y_true + coef * np.power(x, p)
    residual = (np.random.random_sample(n_samples) - 0.5) * noise
    y = y_true + residual

    return x, y, y_true

def make_randomwalk_timeseries_data(n_samples=500, std=1.0, noise=1.0, n_repeats=1, seed=None):
    """
    It generated timeseries formed regression dataset.

        y_t = y_(t-1) + N(0, std)

    Arguments
    ---------
    n_samples : int
        Number of generated data
    std : float
        Standard devation of N(0, std)
    noise : float
        Factor of noise
    n_repeats : int
        Number of samples which have same x
    x_range : tuple
        size = (float, float)
    seed : int or None
        Random seed

    Returns
    -------
        >>> from soydata.data.regression import make_randomwalk_timeseries
        >>> from soydata.visualize import scatterplot

        >>> x, y, y_true = make_randomwalk_timeseries_data(n_repeats=3, noise=0.1, std=10, seed=0)
        >>> scatterplot(x, y, size=3, height=200)
    """

    np.random.seed(seed)
    x_line = np.arange(n_samples)
    y_line = (np.random.randn(n_samples) * std).cumsum()
    x = np.concatenate([x_line for _ in range(n_repeats)])
    add_noise = lambda y: y + np.random.randn(n_samples) * std * noise
    y = np.concatenate([add_noise(y_line)  for _ in range(n_repeats)])
    return x, y, y_line

def make_stepwise_regression_data(n_samples=500, n_steps=5, noise=0.1, x_range=(-1,1), seed=None):
    """
    It generated timeseries formed regression dataset.

        y_t = y_(t-1) + N(0, std)

    Arguments
    ---------
    n_samples : int
        Number of generated data
    n_steps : int
        Number of partially linear regions
    noise : float
        Noise level
    x_range : tuple
        size = (float, float)
    seed : int or None
        Random seed

    Returns
    -------
        >>> from soydata.data.regression import make_stepwise_regression_data
        >>> from soydata.visualize import scatterplot

        >>> x, y, y_true = make_stepwise_regression_data(n_steps=5, noise=0.1, seed=5)
        >>> p = scatterplot(x, y, size=3, height=400, width=800, title='Stepwise regression')
    """

    np.random.seed(seed)
    x = np.linspace(x_range[0], x_range[1], n_samples)
    a = 5*(np.random.random_sample(n_steps) - 0.5)
    size = (n_samples / (n_steps * 3) + np.random.random_sample(n_steps) * n_samples / n_steps)
    size = np.array(n_samples * size/size.sum(), dtype=np.int)
    size = np.concatenate([[0], size]).cumsum()
    size[-1] = n_samples
    y_last = 0
    y_line = []
    for slope, b, e in zip(a, size, size[1:]):
        y_partial = (x[b:e] - x[b]) * slope + y_last
        y_last = y_partial[-1]
        y_line.append(y_partial)
    y_line = np.concatenate(y_line)
    y = y_line + np.random.randn(n_samples) * noise
    return x, y, y_line

def make_step_function_data(n_samples=500, n_steps=5, noise=0.1, x_range=(-1,1), seed=None):
    """
    It generated timeseries formed regression dataset.

        y_t = y_(t-1) + N(0, std)

    Arguments
    ---------
    n_samples : int
        Number of generated data
    n_steps : int
        Number of partially linear regions
    noise : float
        Noise level
    x_range : tuple
        size = (float, float)
    seed : int or None
        Random seed

    Returns
    -------
        >>> from soydata.data.regression import make_step_function_data
        >>> from soydata.visualize import scatterplot

        >>> x, y, y_true = make_step_function_data(n_steps=5, noise=0.1, seed=5)
        >>> p = scatterplot(x, y, size=3, height=400, width=800, title='Step function data')
    """

    np.random.seed(seed)
    x = np.linspace(x_range[0], x_range[1], n_samples)
    y_mean = 5*(np.random.random_sample(n_steps) - 0.5)
    size = (n_samples / (n_steps * 3) + np.random.random_sample(n_steps) * n_samples / n_steps)
    size = np.array(n_samples * size/size.sum(), dtype=np.int)
    size = np.concatenate([[0], size]).cumsum()
    size[-1] = n_samples
    y_line = []
    for mean, b, e in zip(y_mean, size, size[1:]):
        y_line.append([mean] * (e-b))
    y_line = np.concatenate(y_line)
    y = y_line + np.random.randn(n_samples) * noise
    return x, y, y_line
