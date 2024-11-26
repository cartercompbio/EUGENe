import os
from os import PathLike
import yaml
import logging
import torch
import pandas as pd


def make_dirs(
    output_dir: PathLike,
    overwrite: bool = False,
):
    """Make a directory if it doesn't exist.

    Parameters
    ----------
    output_dir : PathLike
        The path to the directory to create.
    overwrite : bool, optional
        Whether to overwrite the directory if it already exists, by default False.
    """
    if os.path.exists(output_dir):
        if overwrite:
            logging.info("Overwriting existing directory: {}".format(output_dir))
            os.system("rm -rf {}".format(output_dir))
        else:
            print("Output directory already exists: {}".format(output_dir))
            return
    os.makedirs(output_dir)


def infer_covariate_types(df: pd.DataFrame, categorical_threshold=0.05):
    """
    Infers the type of covariate for each column in a DataFrame.

    Args:
    - df: Input pandas DataFrame.
    - categorical_threshold: The threshold (as a proportion of the total rows) 
      below which a numeric column is considered categorical. Default is 5%.

    Returns:
    - A dictionary where the keys are column names and the values are the inferred types:
      'binary', 'categorical', or 'continuous'.
    """
    covariate_types = {}

    for col in df.columns:
        # Drop NaN values for proper evaluation
        unique_values = df[col].dropna().unique()
        num_unique = len(unique_values)
        total_rows = df[col].dropna().shape[0]

        # Skip columns with all NaNs
        if total_rows == 0:
            continue

        # Check if column is binary (exactly 2 unique values)
        if num_unique == 2:
            covariate_types[col] = 'binary'

        # Check if column is categorical (non-numeric or few unique values)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # For numeric columns, decide based on the number of unique values
            unique_ratio = num_unique / total_rows

            # If unique_ratio is below a threshold, consider it categorical
            if unique_ratio < categorical_threshold:
                covariate_types[col] = 'categorical'
            else:
                covariate_types[col] = 'continuous'
        else:
            # For non-numeric types, it's considered categorical if there are multiple unique values
            covariate_types[col] = 'categorical' if num_unique > 1 else 'binary'

    return covariate_types


def merge_parameters(parameters, default_parameters):
    """Merge the provided parameters with the default parameters.


    Parameters
    ----------
    parameters: str
        Name of the YAML file with the provided parameters

    default_parameters: dict
        The default parameters for the operation.


    Returns
    -------
    params: dict
        The merged set of parameters.
    """

    if isinstance(parameters, str):
        with open(parameters, "r") as infile:
            parameters = yaml.load(infile, Loader=yaml.FullLoader)
        
    unset_parameters = ("ctrl_var", "n_control_tracks")
    for parameter, value in default_parameters.items():
        
        # If the value itself is a dictionary, recursively merge
        if isinstance(value, dict):
            parameters[parameter] = merge_parameters(parameters.get(parameter, {}), value)

        # If the parameter is not in the provided parameters
        elif parameter not in parameters:
            if value is None and parameter not in unset_parameters:
                raise ValueError("Must provide value for '{}'".format(parameter))
            parameters[parameter] = value
    
    return parameters


def log_parameters(params, logger):
    """Log the parameters to the logger.

    Parameters
    ----------
    params: dict
        The parameters to log.
    logger: logging.Logger
        The logger to use for logging.
    """
    logger.info("Parameters:")
    for key, value in params.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for subkey, subvalue in value.items():
                logger.info(f"    {subkey}: {subvalue}")
        else:
            logger.info(f"  {key}: {value}")
    logger.info("")


def check_for_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device