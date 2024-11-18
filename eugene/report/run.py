"""Single run of report, given input arguments."""

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

from eugene.report import consts

logger = logging.getLogger("eugene")


def run_report(args: argparse.Namespace):
    """The full script for the command line tool report.

    Args:
        args: Inputs from the command line, already parsed using argparse.

    Note: Returns nothing, but writes output to a file(s) specified from command line.

    """
    try:
        # Log the start time.
        logger.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("Running report")

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
        logger.info(f"Overwrite: {overwrite}")

        # Get subcommand
        if args.command == "prep-dataset":
            logger.info("Subcommand 'prep-dataset' selected")
            from eugene.report.prep_dataset import main
            main(params, path_out, overwrite)

        # Log the end time
        logger.info("Completed report")
        logger.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S\n"))

    # The exception allows user to end inference prematurely with CTRL-C.
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Terminated without saving\n")
