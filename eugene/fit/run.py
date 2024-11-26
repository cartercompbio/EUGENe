"""Single run of fit, given input arguments."""

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

from eugene.fit import consts

logger = logging.getLogger("eugene")


def run_fit(args: argparse.Namespace):
    """The full script for the command line tool fit.

    Args:
        args: Inputs from the command line, already parsed using argparse.

    Note: Returns nothing, but writes output to a file(s) specified from command line.

    """
    try:
        # Log the start time.
        logger.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("Running fit")

        # Get params
        params = args.params_file
        if params is not None:
            logger.info(f"Using parameters from {params}")
        else:
            logger.info("Using default parameters")

        # Get path_out
        path_out = args.path_out
        logger.info(f"Output directory: {path_out}")

        # Get report
        report = args.report
        logger.info(f"Report: {report}")

        # Get overwrite
        overwrite = args.overwrite
        logger.info(f"Overwrite: {overwrite}")

        # Get subcommand
        if args.command == "bpnet-lite":
            logger.info("Subcommand 'bpnet-lite' detected. Training bpnet-lite model...")
            from eugene.fit.bpnet_lite import main
            main(params, path_out, report, overwrite)

        elif args.command == "chrombpnet-lite":
            logger.info("Subcommand 'chrombpnet-lite' detected. Training chrombpnet-lite model...")
            from eugene.fit.chrombpnet_lite import main
            main(params, path_out, report, overwrite)

        # Log the end time
        logger.info("Completed fit")
        logger.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S\n"))

    # The exception allows user to end inference prematurely with CTRL-C.
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Terminated without saving\n")
