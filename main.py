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

data_url = 'https://raw.githubusercontent.com/ayw7af/hackathon/main/bs140513_032310.csv'
    #df = getData (data_url)
df = da.getColHist(data_url)
arrays = da.is_category (df, 15)

fig1 = px.histogram(arrays[0], title="Transportation Costs Histogram")
fig2 = px.histogram(arrays[3], title="Food Costs Histogram")
fig3 = px.histogram(arrays[6], title="Technology Costs Histogram")

app.layout = html.Div(
    children=[
        html.Header(
            className="header",
            children=[
                html.H1('Credit Card Transaction Visualizations', className="title")
            ]
    ),
    html.P('This application showcases visualizations of transaction categories.', className="description"), 
    html.Div(
        className="graphs-container", 
        children=[
            dcc.Graph(
                id='graph1',
                figure=fig1
            ),
            dcc.Graph(
                id='graph2',
                figure=fig2
            ),
            dcc.Graph(
                id='graph3',
                figure=fig3
            )
        ]
    ),
    html.P('Created by Amy Wang and Chloe Tran, 2021.', className="footer-tag"),
    html.P('Powered with Dash.', className="footer-tag")
])

if __name__ == '__main__':
    app.run_server()