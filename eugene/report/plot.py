import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..utils import infer_covariate_types


def countplot(data: pd.DataFrame, column: str, title: str):
    """
    Generate a countplot for a categorical or binary variable.

    Parameters:
    - data: DataFrame containing the data.
    - column: The column to plot.
    - title: Title of the plot.

    Returns:
    - A Plotly Figure.
    """
    fig = px.bar(
        data[column].value_counts().reset_index(),
        x=column,
        y="count",
        labels={column: column, "count": "Count"},
        title=title,
        template="simple_white",
    )
    fig.update_layout(
        font=dict(size=14),
        xaxis=dict(showline=True, linewidth=1, linecolor="black", showgrid=False),
        yaxis=dict(showline=True, linewidth=1, linecolor="black", showgrid=False),
        plot_bgcolor="white",
    )
    return fig



def histplot(
    data: pd.DataFrame,
    column: str,
    hue: str = None,
    title: str = "",
    cumulative: bool = False,
    colors: list = None,
):
    """
    Generate a histogram, optionally split by a hue (categorical variable).

    Parameters:
    - data: DataFrame containing the data.
    - column: The continuous variable to plot.
    - hue: Optional categorical variable to split the histogram by.
    - title: Title of the plot.
    - cumulative: Whether to plot cumulative histograms.
    - colors: List of colors for each category in the hue.

    Returns:
    - A Plotly Figure.
    """
    fig = go.Figure()

    if hue:
        categories = data[hue].dropna().unique()
        colors = colors or px.colors.qualitative.Set1[: len(categories)]
        for i, category in enumerate(categories):
            subset = data[data[hue] == category]
            fig.add_trace(
                go.Histogram(
                    x=subset[column],
                    cumulative=dict(enabled=cumulative),
                    name=str(category),
                    marker=dict(color=colors[i]),
                    opacity=0.7,
                )
            )
    else:
        fig.add_trace(
            go.Histogram(
                x=data[column],
                cumulative=dict(enabled=cumulative),
                marker=dict(color=colors[0] if colors else "blue"),
            )
        )

    fig.update_layout(
        barmode="overlay",
        title=title,
        xaxis=dict(title=column, showline=True, linewidth=1, linecolor="black"),
        yaxis=dict(title="Count", showline=True, linewidth=1, linecolor="black"),
        template="simple_white",
        plot_bgcolor="white",
        legend=dict(title=hue if hue else None),
    )
    return fig


def violinplot(data: pd.DataFrame, x_column: str, y_column: str, title: str):
    """
    Generate a violin plot for a continuous variable split by a categorical variable.

    Parameters:
    - data: DataFrame containing the data.
    - x_column: The categorical variable.
    - y_column: The continuous variable.
    - title: Title of the plot.

    Returns:
    - A Plotly Figure.
    """
    fig = px.violin(
        data, x=x_column, y=y_column, box=True, title=title
    )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    return fig


def heatmap(data: pd.DataFrame, x_column: str, y_column: str, title: str):
    """
    Generate a heatmap for the crosstab of two categorical variables.

    Parameters:
    - data: DataFrame containing the data.
    - x_column: The first categorical variable.
    - y_column: The second categorical variable.
    - title: Title of the plot.

    Returns:
    - A Plotly Figure.
    """
    crosstab = pd.crosstab(data[x_column], data[y_column])
    fig = px.imshow(crosstab, text_auto=True, title=title)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def scatterplot(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    sorted: bool = False,
    logx: bool = False,
    logy: bool = False,
    x_axis_title: str = None,
    y_axis_title: str = None,
    cumulative: bool = False,
    show_xaxis_labels: bool = False,
    colors: list = None,  # New parameter for optional colors
    size: int = 1
):
    """Create a scatter plot layout in Dash using Plotly.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data for the plot.
    x_column : str
        The column to use for the x-axis.
    y_column : str
        The column to use for the y-axis.
    title : str
        Title of the plot.
    x_axis_title : str, optional
        Title for the x-axis.
    y_axis_title : str, optional
        Title for the y-axis.
    cumulative : bool, optional
        Whether to plot cumulative values.
    show_xaxis_labels : bool, optional
        Whether to show labels on the x-axis.
    colors : list, optional
        List of colors corresponding to each point in the plot.

    Returns
    -------
    go.Figure
        A Plotly Figure containing the scatter plot.
    """
    # Compute cumulative values
    if cumulative:
        data[y_column] = data[y_column].cumsum()

    # Sort
    x_data = data.sort_values(y_column, ascending=cumulative)[x_column] if sorted else data[x_column]
    y_data = data.sort_values(y_column, ascending=cumulative)[y_column] if sorted else data[y_column]

    # Plot
    fig = go.Figure(
        data=go.Scattergl(
            x=x_data,
            y=y_data,
            mode='markers',
            marker=dict(
                color=colors if colors is not None else 'blue',  # Use passed colors or default to blue
                size=size
            )
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title if x_axis_title else x_column,
        yaxis_title=y_axis_title if y_axis_title else y_column,
        xaxis=dict(showticklabels=show_xaxis_labels),
        template='simple_white',
        plot_bgcolor='white'
    )

    if logx:
        fig.update_layout(xaxis_type='log')
    if logy:
        fig.update_layout(yaxis_type='log')
    
    return fig


def generate_plot(
    data: pd.DataFrame, 
    x: str = None, 
    y: str = None, 
    plot_type: str = "auto", 
    **kwargs
):
    """
    Generate an appropriate plot based on selected covariates and desired plot type.

    Parameters:
    - data: DataFrame containing the data.
    - x: The first selected covariate.
    - y: The second selected covariate (optional).
    - plot_type: Manual override for the plot type ("auto", "violin", "histogram").
    - kwargs: Additional Plotly customization options.

    Returns:
    - A Plotly Figure.
    """
    covariate_types = infer_covariate_types(data)

    if not x and not y:
        fig = go.Figure(layout=dict(template='plotly'))
        fig.update_layout(title="Select covariates to generate a plot.")
        return fig
    
    if x and not y:
        if covariate_types[x] in ["binary", "categorical"]:
            return countplot(data, column=x, title=f"Countplot for {x}", **kwargs)
        elif covariate_types[x] == "continuous":
            return histplot(data, column=x, title=f"Histogram for {x}", **kwargs)

    if x and y:
        x_type, y_type = covariate_types[x], covariate_types[y]

        if plot_type == "histogram" and x_type in ["binary", "categorical"] and y_type == "continuous":
            return histplot(data, column=y, hue=x, title=f"Histogram of {y} by {x}", **kwargs)
        elif x_type in ["binary", "categorical"] and y_type == "continuous":
            return violinplot(data, x_column=x, y_column=y, title=f"Violin Plot of {y} by {x}", **kwargs)
        elif x_type == "continuous" and y_type in ["binary", "categorical"]:
            return violinplot(data, x_column=y, y_column=x, title=f"Violin Plot of {x} by {y}", **kwargs)
        elif x_type in ["binary", "categorical"] and y_type in ["binary", "categorical"]:
            return heatmap(data, x_column=x, y_column=y, title=f"Heatmap of {x} vs {y}", **kwargs)
        else:
            return scatterplot(data, x_column=x, y_column=y, title=f"Scatterplot of {x} vs {y}", **kwargs)

    raise ValueError("Invalid selection or plot type.")

def trackplot(
    tracks, 
    interval, 
    height=1.5, 
    colors=None
):
    # Create a figure object
    fig = go.Figure()
    
    x = np.linspace(interval["start"], interval["end"], num=len(next(iter(tracks.values()))))
    
    # Add each track to the figure
    for title, y in tracks.items():
        color = colors[title] if colors else None
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                fill='tozeroy',
                name=title,
                line=dict(color=color) if color else None
            )
        )
    
    # Set axis labels and layout
    fig.update_layout(
        title="Tracks",
        xaxis_title=f"{interval['chrom']}:{interval['start']}-{interval['end']}",
        yaxis_title="Value",
        height=height * 300,  # Adjust height dynamically
        template="simple_white",
        showlegend=True
    )
    
    return fig