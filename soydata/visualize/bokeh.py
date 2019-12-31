import numpy as np
from bokeh.plotting import figure, show
from bokeh.palettes import Spectral, Turbo256


def scatterplot(x, y=None, labels=None, color='#5e4fa2', size=5,
    alpha=0.95, p=None, show_inline=True, marker='circle', **kargs):

    if isinstance(x, np.ndarray) and (len(x.shape) == 2) and (x.shape[1] == 2):
        if (y is not None) and (labels is None):
            labels = y
        x, y = x[:,0], x[:,1]

    if p is None:
        p = initialize_figure(
            title = kargs.get('title', None),
            height = kargs.get('height', 600),
            width = kargs.get('width', 600),
            toolbar_location = kargs.get('toolbar_location', 'right'),
            tools = kargs.get('tools', 'default')
        )

    if labels is not None:
        color = initialize_palette(
            labels = labels,
            palette = kargs.get('palette', None),
            n_labels = kargs.get('n_labels', -1)
        )[:x.shape[0]]

    p.scatter(x, y, color=color, size=size, alpha=alpha, marker=marker)
    if show_inline:
        show(p)

    return p

def lineplot(X, y=None, pairs=None, line_width=0.5, line_dash=(5,3),
    line_color='#2b83ba', p=None, show_inline=True, **kargs):

    if p is None:
        p = initialize_figure(
            title = kargs.get('title', None),
            height = kargs.get('height', 600),
            width = kargs.get('width', 600),
            toolbar_location = kargs.get('toolbar_location', 'right'),
            tools = kargs.get('tools', 'default')
        )

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
        X.sort(axis=0)
        x = X[:,0]
        y = X[:,1]
        p.line(x, y, line_color=line_color, line_dash=line_dash, line_width=line_width)

    if show_inline:
        show(p)

    return p

def initialize_figure(title=None, height=600, width=600, toolbar_location='right', tools='default'):
    if tools == 'default':
        tools = 'pan wheel_zoom box_zoom save reset'.split()
    return figure(title=title, height=height, width=width, toolbar_location=toolbar_location, tools=tools)

def initialize_palette(labels, palette=None, n_labels=-1):
    uniques = np.unique(labels)
    uniques = uniques[np.where(uniques >= 0)[0]]
    if n_labels < 1:
        n_labels = uniques.max()+1
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
    label_to_color[-1] = 'lightgrey'
    color = [label_to_color[label] for label in labels]
    return color
