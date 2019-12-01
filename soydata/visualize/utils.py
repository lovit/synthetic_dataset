from bokeh.plotting import output_notebook
from plotly.offline import init_notebook_mode

def use_notebook():
    init_notebook_mode(connected=True)
    output_notebook()
