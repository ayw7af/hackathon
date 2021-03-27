# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import ui 
import data_analysis as da

app = dash.Dash(__name__, external_stylesheets=ui.external_stylesheets)

fig = px.bar(da.df_sub, x="age", y="amount", color="category", barmode="group")

app.layout = html.Div(children=[
    html.H1(children=
    'Fraud Classification Dashboard'
    ),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)