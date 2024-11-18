"""Handle the saving and loading of models from checkpoint files."""

import argparse
import glob
import hashlib
import logging
import os
import pickle
import random
import shutil
import tarfile
import tempfile
import traceback
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("eugene")


def save_random_state(filebase: str) -> List[str]:
    """Write states of various random number generators to files.

    NOTE: the pyro.util.get_rng_state() is a compilation of python, numpy, and
        torch random states.  Here, we save the three explicitly ourselves.
        This is useful for potential future usages outside of pyro.
    """
    # https://stackoverflow.com/questions/32808686/storing-a-random-state/32809283

    file_dict = {}

    # Random state information
    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    file_dict.update(
        {
            filebase + "_random.python": python_random_state,
            filebase + "_random.numpy": numpy_random_state,
        }
    )

    # Save it
    for file, state in file_dict.items():
        with open(file, "wb") as f:
            pickle.dump(state, f)

    return list(file_dict.keys())


def load_random_state(filebase: str):
    """Load random states from files and update generators with them."""

    with open(filebase + "_random.python", "rb") as f:
        python_random_state = pickle.load(f)
    random.setstate(python_random_state)

    with open(filebase + "_random.numpy", "rb") as f:
        numpy_random_state = pickle.load(f)
    np.random.set_state(numpy_random_state)


def create_workflow_hashcode(
    module_path: str,
    args: argparse.Namespace,
    args_to_remove: List[str] = ["epochs", "fpr"],
    name: str = "md5",
    verbose: bool = False,
) -> str:
    """Create a hash blob from eugene python code plus input arguments."""

    hasher = hashlib.new(name=name)

    files_safe_to_ignore = [
        "report.py",
        "monitor.py",
        "sparse_utils.py",
    ]

    if not os.path.exists(module_path):
        return ""

    try:
        # files
        for root, _, files in os.walk(module_path):
            for file in files:
                if not file.endswith(".py"):
                    continue
                if file in files_safe_to_ignore:
                    continue
                if "test" in file:
                    continue
                if verbose:
                    print(file)
                with open(os.path.join(root, file), "rb") as f:
                    buf = b"\n".join(f.readlines())
                hasher.update(buf)  # encode python files

        # inputs
        args_dict = vars(args).copy()
        for arg in args_to_remove:
            args_dict.pop(arg, None)
        hasher.update(str(args_dict).encode("utf-8"))  # encode parsed input args

        # input file
        # TODO
        # this is probably not necessary for real data... why would two different
        # files have the same name?
        # but it's useful for development where simulated datasets change

    except Exception:
        return ""

    return hasher.hexdigest()
