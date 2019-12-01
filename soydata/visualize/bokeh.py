import numpy as np
from bokeh.plotting import figure, show
from bokeh.palettes import Spectral, Turbo256


def initialize_palette(labels, palette=None):
    uniques = set(labels)
    n_labels = len(uniques)
    if palette is not None:
        palette = palette
    elif n_labels <= 3:
        palette = '#B8D992 #BE83AB #78CCD0'.split()
    elif n_labels <= 11:
        palette = Spectral[max(4, n_labels)]
    else:
        step = int(256 / n_labels)
        palette = [Turbo256[i] for i in range(0, 256, step)][:n_labels]
    n_colors = len(palette)
    label_to_color = {label:palette[i % n_colors] for i, label in enumerate(uniques)}
    color = [label_to_color[label] for label in labels]
    return color

def scatterplot(x, y=None, labels=None, color='#5e4fa2', size=5,
    alpha=0.95, p=None, show_inline=True, **kargs):

    if isinstance(x, np.ndarray) and (len(x.shape) == 2) and (x.shape[1] == 2):
        if (y is not None) and (labels is None):
            labels = y
        x, y = x[:,0], x[:,1]

    if p is None:
        p = initialize_figure(kargs.get('title', None),
            kargs.get('height', 600), kargs.get('width', 600))

    if labels is not None:
        color = initialize_palette(labels, kargs.get('palette', None))

    p.scatter(x, y, color=color, size=size, alpha=alpha)
    if show_inline:
        show(p)

    return p

def lineplot(X, y=None, pairs=None, line_width=0.5, line_dash=(5,3),
    line_color='#2b83ba', p=None, show_inline=True, **kargs):

    if p is None:
        p = initialize_figure(kargs.get('title', None),
            kargs.get('height', 600), kargs.get('width', 600))

    if y is not None:
        X = np.vstack([X, y]).T
    if line_dash is None:
        line_dash = []

    if pairs is not None:
        for f, t in pairs:
            x = [X[f,0], X[t,0]]
            y = [X[f,1], X[t,1]]
            p.line(x, y, line_color=line_color, line_dash=line_dash, line_width=line_width)
    else:
        x = X[:,0]
        y = X[:,1]
        p.line(x, y, line_color=line_color, line_dash=line_dash, line_width=line_width)

    if show_inline:
        show(p)

    return p

def regression_line(x, model_or_y, p=None, n_steps=2, margin=0.025,
    legend=None, line_dash=(4,4), line_color='orange', line_width=2, **kargs):

    """
    :param numpy.ndarray x: x value or range of x
        If x stands for x-range, the length of x must be 2
        If x is instance of numpy.ndarray, the x must be column vector
    :param model_or_numpy.ndarray model_or_y: predicative model or y value
        All functions are possilble to use if it works like

            y = model(x)

        Or numpy.ndarray column vector

    :param bokeh.plotting.figure.Figure p: Figure to overlay line
    :param int n_steps: The number of points in x
        If works only when x is range
    :param float margin: x_ramge margin
    :param str legend: Line legend
    :param tuple line_dash: bokeh.core.properties.DashPattern
    :param str line_color: Color code
    :param int line_width: Width of regression line

    :returns: p
        Bokeh figure which ovelayed regerssion line

    Usgae
    -----
        Draw base scatter plot

        >>> p = scatterplot((x, y), colors='#323232')

        With dummy model

        >>> model = lambda x: 1.0 * x + 1.0
        >>> p = overlay_regression_line(x, model, p, n_steps=5, legend='test')
        >>> show(p)
    """

    if p is None:
        p = initialize_figure(kargs.get('title', None),
            kargs.get('height', 600), kargs.get('width', 600))

    if isinstance(x, np.ndarray):
        if len(x.shape) == 1:
            x = x.reshape(-1,1)
        elif len(x.shape) > 2:
            raise ValueError(f'x must be vector not tensor, however the shape of input is {x.shape}')
        if x.shape[1] > 1:
            raise ValueError(f'x must be 1D data, however the shape of input is {x.shape}')
        x_ = x.copy()
        sorting_indices = x_.argsort(axis=0).reshape(-1)
        x_ = x_[sorting_indices]
    elif len(x) != 2:
        raise ValueError(f'x must be numpy.ndarray column vector or range, however the length of x is {len(x)}')
    else:
        x_min, x_max = x
        x_min, x_max = x_min - margin, x_max + margin
        x_ = np.linspace(x_min, x_max, n_steps).reshape(-1,1)

    if isinstance(model_or_y, np.ndarray):
        y_pred = model_or_y.copy()
        if not isinstance(x, np.ndarray):
            raise ValueError(f'x should be numpy.ndarray when y is numpy.ndarray instance')
        y_pred = y_pred[sorting_indices]
    else:
        # (n_data, 1) -> (n_data, 1)
        y_pred = model_or_y(x_)

    # as column vector
    x_ = x_.reshape(-1)
    y_pred = y_pred.reshape(-1)

    if legend is None:
        p.line(x_, y_pred, line_dash=line_dash, line_color=line_color, line_width=line_width)
    else:
        p.line(x_, y_pred, line_dash=line_dash, line_color=line_color, line_width=line_width, legend_label=legend)
    return p

def initialize_figure(title=None, height=600, width=600):
    return figure(title=title, height=height, width=width)
