import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import ui 
import data_analysis as da

app = dash.Dash(__name__, external_stylesheets= ui.external_stylesheets)
app.title = 'Fraud Classification'
server = app.server

fig = px.histogram(da.df, x="amount")

app.layout = html.Div(children=[
    html.Header(
        className="header",
        children=[
            html.H1('Fraud Classification', className="title")
        ]
    ),
    html.P('This application showcases a classification of fraudulent charges based on a set of features.', className="description"), 
    html.Div(
        className="graphs-container", 
        children=[
            dcc.Graph(
                id='graph1',
                figure=fig
            ),
            dcc.Graph(
                id='graph2',
                figure=fig
            ),
            dcc.Graph(
                id='graph3',
                figure=fig
            ),
            dcc.Graph(
                id='graph4',
                figure=fig
            ),
        ]
    ),
    html.P('Created by Amy Wang and Chloe Tran, 2020.', className="footer-tag"),
    html.P('Powered with Dash.', className="footer-tag")
])

if __name__ == '__main__':
    app.run_server()