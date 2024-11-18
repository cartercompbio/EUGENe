import os
import logging
import json
import pickle

from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

from .utils import (
    merge_parameters,
)

logger = logging.getLogger("eugene")

default_params = {
}


def main(
    path_params,
    path_out,
    overwrite=False,
):
    
    # Merge with default parameters
    params = merge_parameters(path_params, default_params)
    command = params["command"]
    # Create Dash app
    app = Dash(
        __name__, 
        use_pages=False, 
        external_stylesheets=[dbc.themes.LUX],  # Updated theme to LUX for a modern look
        suppress_callback_exceptions=True
    )
    app.title = f"EUGENe Report: {command}"
    

    app.layout = dbc.Container([
            
            # Title Row
            dbc.Row([
                dbc.Col(html.H1(f"EUGENe Report: {command}",
                                style={'fontSize': '3rem', 'textAlign': 'center', 
                                    'color': '#2c3e50', 'marginBottom': '20px', 'fontWeight': 'bold'}))
            ]),

            # Tabs for different sections: Run Parameters, SeqData, Sequence Statistics, K-mer Analysis, Motif Analysis
            dcc.Tabs([
            
                # Run Parameters
                dcc.Tab(label='Run Parameters', children=[
                    html.Div([
                        html.H2("Run Parameters"),
                        html.Pre(json.dumps(params, indent=4))
                    ])
                ]),

                # SeqData
                dcc.Tab(label='SeqData', children=[
                    html.Div([
                        html.H2("SeqData"),
                        html.Pre("SeqData")
                    ])
                ]),

                # Sequence Statistics
                dcc.Tab(label='Sequence Statistics', children=[
                    html.Div([
                        html.H2("Sequence Statistics"),
                        html.Pre("Sequence Statistics")
                    ])
                ]),

                # K-mer Analysis
                dcc.Tab(label='K-mer Analysis', children=[
                    html.Div([
                        html.H2("K-mer Analysis"),
                        html.Pre("K-mer Analysis")
                    ])
                ]),

                # Motif Analysis
                dcc.Tab(label='Motif Analysis', children=[
                    html.Div([
                        html.H2("Motif Analysis"),
                        html.Pre("Motif Analysis")
                    ])
                ]),
            ])
        ])

    app.run_server(debug=True, host='0.0.0.0', port=8050)
