import yaml
import torch
import numpy as np
import pandas as pd


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
        
    unset_parameters = ("ctrl_var", "n_control_tracks", "early_stopping")
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