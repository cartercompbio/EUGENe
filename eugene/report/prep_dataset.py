import os
import logging
import json
import pickle
import numpy as np
import pandas as pd

from dash import Dash, html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from dash import DiskcacheManager
import dash_dangerously_set_inner_html
import diskcache
from uuid import uuid4

import seqdata as sd

from ..utils import (
    merge_parameters,
    infer_covariate_types
)
from .plot import (
    generate_plot,
    trackplot
)
from .static_report import generate_html_report

logger = logging.getLogger("eugene")

default_params = {
}


results = None

def main(
    path_params,
    path_out,
    overwrite=False,
):
    
    global results
    
    # Merge with default parameters
    params = merge_parameters(path_params, default_params)
    command = params["command"]

    # Setup cache
    launch_uid = uuid4()
    cache = diskcache.Cache("./.cache")
    background_callback_manager = DiskcacheManager(
        cache, cache_by=[lambda: launch_uid], expire=60
    )

    # Cache the data loading and parsing function
    @cache.memoize()
    def load_and_parse_data(path_sdata):
        sdata = sd.open_zarr(path_sdata)
        logger.info(f"Loaded SeqData with {sdata.dims['_sequence']} sequences")

        # Parse data
        seq_df = sdata[["chrom", "chromStart", "chromEnd", "type", "length", "gc_percent", "total_counts"]].to_dataframe()
        char_df = sdata["alphabet_cnt"].to_dataframe().reset_index()
        char_df["_alphabet"]=char_df["_alphabet"].astype(str)
        char_df = char_df.pivot(index="_sequence", columns="_alphabet", values="alphabet_cnt")
        char_df["non_alphabet_cnt"] = sdata["non_alphabet_cnt"].values
        char_df = char_df.div(sdata["length"].values, axis=0)
        assert np.allclose(char_df.sum(axis=1), 1)
        seq_df = pd.concat([seq_df, char_df], axis=1)
        
        # Results
        results = {
            "seq_df": seq_df,
            "sdata": sdata,
        }
        return results
    
    # Load and parse data
    results = load_and_parse_data(params["path_sdata"])
    cache.set('results', results)
    results = cache.get("results")

    # Infer covariate types
    seq_df = results["seq_df"]
    covariate_types = infer_covariate_types(seq_df)
    covariates = list(covariate_types.keys())

    # Get all regions
    regions = results["sdata"]._sequence.values

    # Create Dash app
    app = Dash(
        __name__, 
        use_pages=False, 
        external_stylesheets=[dbc.themes.LUX],
        background_callback_manager=background_callback_manager, 
        suppress_callback_exceptions=True
    )
    app.title = f"Interactive {command} Report"
    

    app.layout = dbc.Container([
            
            # Title Row
            dbc.Row([
                dbc.Col(html.H1(f"Interactive {command} Report",
                                style={'fontSize': '3rem', 'textAlign': 'center', 
                                    'color': '#2c3e50', 'marginBottom': '20px', 'fontWeight': 'bold'}))
            ]),

            # Tabs for different sections: SeqData, Sequence Statistics, K-mer Analysis, Motif Analysis, Run Parameters
            dcc.Tabs([

                # SeqData
                dcc.Tab(label='SeqData', children=[
                    html.Div([
                        html.H2("SeqData"),
                        html.Div(  # Use a Div for rendering HTML content
                            dash_dangerously_set_inner_html.DangerouslySetInnerHTML(results["sdata"]._repr_html_()),
                        )
                    ]),
                    dbc.Row([
                        # Drop down for region to plot
                        dbc.Col([
                            html.Label("Select Region"),
                            dcc.Dropdown(
                                id='region',
                                options=[{'label': i, 'value': i} for i in regions],
                                value=regions[0]
                            )
                        ]),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='tracks')
                        ])
                    ])
                ]),

                # Sequence Statistics
                dcc.Tab(label='Sequence Statistics', children=[
                    html.Div([
                        html.H2("Sequence Statistics"),
                        html.P("This tab provides..."),
                        dbc.Row([
                
                            # Drop down for covariate 1
                            dbc.Col([
                                html.Label("Select Covariate 1"),
                                dcc.Dropdown(
                                    id='cov1',
                                    options=[{'label': i, 'value': i} for i in covariates],
                                    value=None
                                )
                            ]),

                            # Drop down for covariate 2
                            dbc.Col([
                                html.Label("Select Covariate 2"),
                                dcc.Dropdown(
                                    id='cov2',
                                    options=[{'label': i, 'value': i} for i in covariates],
                                    value=None
                                )
                            ]),
                        ]),

                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='sequence_stats')
                            ])
                        ])

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

                # Run Parameters
                dcc.Tab(label='Run Parameters', children=[
                    html.Div([
                        html.H2("Run Parameters"),
                        html.Pre(json.dumps(params, indent=4))
                    ])
                ]),
            ])
        ])

    app.run_server(debug=True, host='0.0.0.0', port=8050)
    
    # delete the cache once the app is closed
    cache.clear()

    # TODO: Generate a static report
    path_report = generate_html_report(seq_df, path_out)
    

# Callback for sequence statistics
@callback(
    Output('sequence_stats', 'figure'),
    [Input('cov1', 'value'),
     Input('cov2', 'value')]
)
def update_sequence_stats(
    cov1, 
    cov2,
    debug=False
):

    seq_df = results["seq_df"]
    if debug:
        print(f"cov1: {cov1}, cov2: {cov2}")
        print(f"seq_df: {seq_df.head()}")
    return generate_plot(seq_df, x=cov1, y=cov2, plot_type="auto")


# Callback for tracks
@callback(
    Output('tracks', 'figure'),
    [Input('region', 'value')]
)
def update_tracks(
    region,
    debug=False
):
    sdata = results["sdata"]
    if debug:
        print(f"region: {region}")
        print(f"sdata: {sdata}")
    example = sdata.sel(_sequence=region)
    trues = example["cov"].values.squeeze()
    chrom = example["chrom"].values
    chromStart = example["chromStart"].values
    chromEnd = example["chromEnd"].values
    interval = dict(chrom=chrom, start=chromStart, end=chromEnd)
    tracks = {
        "Signal": trues,
    }
    colors = {
        "Signal": "lightcoral",
    }
    fig = trackplot(tracks, interval, colors=colors, height=2)
    return fig
