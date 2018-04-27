import numpy as np
from plotly.offline import plot, iplot
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

def ipython_3d_scatter(X, color, text=None, 
        width=600, height=600, marker_size=3, colorscale='Jet'):

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
    iplot(fig)

def ipython_2d_scatter(X, color, text=None,
        width=600, height=600, marker_size=3, colorscale='Jet'):

    colorset = np.unique(color)
    if colorset.shape[0] == 1:
        colormap = np.asarray([0])
    else:
        colormap = (colorset - colorset[0]) / (colorset[-1] - colorset[0])

    text = text if text else ['point #{}'.format(i) for i in range(X.shape[0])]

    data = []
    for label in colorset:
        indices = np.where(color == label)[0]
        Xsub = X[indices]
        trace = go.Scatter(
            x = Xsub[:,0],
            y = Xsub[:,1],
            text = [text[i] for i in indices],
            mode = 'markers',
            marker = dict(
                size=marker_size,
                color=colormap[label],
                colorscale=colorscale
            ),
            opacity=0.8
        )
        data.append(trace)
    
    layout = go.Layout(
        autosize=False,
        width=width,
        height=height,
        margin=go.Margin(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        )
    )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig)