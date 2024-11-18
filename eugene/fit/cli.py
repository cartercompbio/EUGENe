"""Command-line tool functionality for fit."""

import argparse
import logging
import os
import sys

import eugene
from eugene.base_cli import AbstractCLI, get_version
from eugene.fit.checkpoint import create_workflow_hashcode


class CLI(AbstractCLI):
    """CLI implements AbstractCLI from the eugene package."""

    def __init__(self):
        self.name = "fit"
        self.args = None

    def get_name(self) -> str:
        return self.name

    @staticmethod
    def validate_args(args) -> argparse.Namespace:
        """Validate parsed arguments."""

        # Ensure write access to the save directory.
        if not os.path.exists(args.path_out):
            os.makedirs(args.path_out)
        if args.path_out:
            assert os.access(args.path_out, os.W_OK), (
                f"Cannot write to specified output directory {args.path_out}. "
                f"Ensure the directory exists and is write accessible."
            )

        # Make sure n_threads makes sense.
        if args.n_threads is not None:
            assert args.n_threads > 0, "--cpu-threads must be an integer >= 1"

        # Return the validated arguments.
        return args

    @staticmethod
    def run(args):
        """Run the main tool functionality on parsed arguments."""

        # Run the tool.
        return main(args)


def setup_and_logging(args):
    """Take command-line input, parse arguments, and run tests or tool."""

    # Send logging messages to stdout as well as a log file.
    path_out = args.path_out
    log_file = os.path.join(path_out, "fit.log")
    logger = logging.getLogger("eugene")  # name of the logger
    logger.setLevel(logging.INFO if not args.debug else logging.DEBUG)
    formatter = logging.Formatter("eugene:fit: %(message)s")
    file_handler = logging.FileHandler(filename=log_file, mode="w", encoding="UTF-8")
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)  # set the file format
    console_handler.setFormatter(formatter)  # use the same format for stdout
    logger.addHandler(file_handler)  # log to file
    logger.addHandler(console_handler)  # log to stdout

    # Log the command as typed by user.
    logger.info("Command:\n" + " ".join(["eugene", "fit"] + sys.argv[2:]))
    logger.info("eugene " + get_version())

    # Set up checkpointing by creating a unique workflow hash.
    hashcode = create_workflow_hashcode(
        module_path=os.path.dirname(eugene.__file__),
        args_to_remove=(
            [
                "path_out",
                "debug",
                "cpu_threads",
            ]
        ),
        args=args,
    )
    args.checkpoint_filename = hashcode  # store this in args
    logger.info(f"(Workflow hash {hashcode})")
    return args, file_handler


def main(args):
    """Take command-line input, parse arguments, and run tests or tool."""

    args, file_handler = setup_and_logging(args)

    # Run the tool.
    from eugene.fit.run import run_fit
    run_fit(args)
    file_handler.close()

    return
