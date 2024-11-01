"""Single run of prep-dataset, given input arguments."""

import argparse
import logging
import os
import sys
import yaml
import traceback
from datetime import datetime
from typing import Dict, Optional, Tuple, Union

import numpy as np
import matplotlib
import pandas as pd
import psutil

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # This needs to be after matplotlib.use('Agg')
import seaborn as sns

from eugene.prep_dataset import consts

logger = logging.getLogger("eugene")


def run_prep_dataset(args: argparse.Namespace):
    """The full script for the command line tool prep-dataset.

    Args:
        args: Inputs from the command line, already parsed using argparse.

    Note: Returns nothing, but writes output to a file(s) specified from command line.

    """
    try:
        # Log the start time.
        logger.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("Running prep-dataset")

        # Get params
        params = args.params_file
        if params is not None:
            logger.info(f"Using parameters from {params}")
        else:
            logger.info("Using default parameters")

        # Get path_out
        path_out = args.path_out
        logger.info(f"Output directory: {path_out}")

        # Get overwrite
        overwrite = args.overwrite

        # Get subcommand
        if args.command == "tabular":
            logger.info("Subcommand 'tabular' detected. Preparing tabular dataset...")
            from eugene.prep_dataset.tabular import main
            main(params, path_out, overwrite)

        elif args.command == "regions":
            logger.info("Subcommand 'regions' detected. Preparing regions dataset...")
            from eugene.prep_dataset.regions import main
            main(params, path_out, overwrite)

        # Log the end time
        logger.info("Completed prep-dataset")
        logger.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S\n"))

    # The exception allows user to end inference prematurely with CTRL-C.
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Terminated without saving\n")
