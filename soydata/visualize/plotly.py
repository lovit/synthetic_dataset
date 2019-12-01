import numpy as np
from plotly.offline import plot, iplot
import plotly.graph_objs as go


def scatterplot3d(X, color, text=None, title=None, width=600,
    height=600, marker_size=3, colorscale='Jet', show_inline=True):

    """
    Arguments
    ---------
    X : numpy.ndarray
        Sample data, Shape = (n_samples, 3)
    color : numpy.ndarray
        Shape = (n_samples,)
        The normalized univariate position of the sample according to the main
        dimension of the points in the manifold.
        Its scale is bounded in [0, 1] with float or [0, 256) with integer.
    text : str or None
        Description of each point
    title : str or None
        Title of the figure
    width : int
        Figure width, default is 600
    height : int
        Figure height, default is 600
    marker_size : int
        Size of markers
    colorscale : str
        Predefined colorscales in Plotly express, for example `Jet`, `Magma`
    show_inline : Boolean
        If True, it shows the figure

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        Plotly figure

    Usage
    -----
    To draw 3D scatterplot of swiss-roll

        >>> from soydata.data.unsupervised import make_swiss_roll
        >>> from soydata.visualize import scatterplot3d

        >>> X, color = make_swiss_roll(n_samples=1000, thickness=1.5, gap=2)
        >>> fig = scatterplot3d(X, color)

    To save figure

        >>> from plotly.offline import plot
        >>> plot(fig, filename='plotly-3d-scatter-small.html', auto_open=False)

    """
    data = go.Scatter3d(
        x=X[:,0],
        y=X[:,1],
        z=X[:,2],
        text = text if text else ['point #{}'.format(i) for i in range(X.shape[0])],
        mode='markers',
        marker=dict(
            size=marker_size,
            color=color,
            colorscale=colorscale,
            line=dict(
                #color='rgba(217, 217, 217, 0.14)',
                #color='rgb(217, 217, 217)',
                width=0.0
            ),
            opacity=0.8
        )
    )

    layout = go.Layout(
        title = title if title else '',
        autosize=False,
        width=width,
        height=height,
        margin=go.Margin(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        #paper_bgcolor='#7f7f7f',
        #plot_bgcolor='#c7c7c7'
    )

    fig = go.Figure(data=[data], layout=layout)
    if show_inline:
        iplot(fig)

    return fig
