import os
import logging
import json
import pickle
import torch
import numpy as np
import pandas as pd
import xarray as xr
import seqdata as sd
import seqpro as sp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from bpnetlite.chrombpnet import BPNet

from .utils import (
    merge_parameters,
    check_for_gpu,
)

logger = logging.getLogger("eugene")

default_params = {
    "seqdata": {
        "ctrl_var": None,
        "overwrite": False,
    },
    "model": {
        "n_control_tracks": None,
    },
}


def main(
    path_params,
    path_out,
    overwrite=False,
):

    #-------------- Load parameters --------------#

    # Merge with default parameters
    params = merge_parameters(path_params, default_params)

    # Log parameters
    message=("--- Parameters ---\n")
    for key, value in params.items():
        message += f"\t{key}: "
        if isinstance(value, dict):
            for key, value in value.items():
                message += f"\n\t\t{key}: {value}"
            message += "\n"
        else:
            message += f"{value}\n"
    message += "\n"
    logger.info(message)
    
    # Grab params
    name = params["name"]
    threads = params["threads"]
    random_state = params["random_state"]

    # Check if prefix.torch exists, if it does and overwrite is False, raise an error
    prefix = os.path.join(path_out, name)
    if not overwrite and os.path.exists(prefix + ".torch"):
        raise ValueError(f"{prefix}.torch already exists. Set overwrite=True to overwrite.")
    
    # 
    device = check_for_gpu()
    if device == torch.device("cpu"):
        raise ValueError("Cannot currently train bpnet-lite models on CPU. Exiting.")

    #-------------- Load SeqData --------------#
    logger.info("--- Loading SeqData ---")
    path_seqdata = params["seqdata"]["path"]
    fold = params["seqdata"]["fold"]
    seq_length = params["seqdata"]["seq_length"]
    target_length = params["seqdata"]["target_length"]
    max_jitter = params["seqdata"]["max_jitter"]

    # Read in the sequence data
    logger.info(f"Reading sequence data from {path_seqdata}\n")
    sdata = sd.open_zarr(path_seqdata)

    # Define the trimming
    logger.info("--- Defining the trimming ---")
    trimming = (seq_length - target_length) // 2
    seqs_start = (sdata.dims["_length"] // 2) - (seq_length // 2)
    counts_start = (sdata.dims["_length"] // 2) - (target_length // 2)
    logger.info(f"Sequence length is {seq_length}")
    logger.info(f"Target length is {target_length}")
    logger.info(f"Trimming is {trimming}\n")

    #-------------- Split the data --------------#
    logger.info("--- Splitting the data ---")
    train_idx = np.where(sdata[fold] == "train")[0]
    valid_idx = np.where(sdata[fold] == "valid")[0]
    test_idx = np.where(sdata[fold] == "test")[0]
    train_data = sdata.isel(_sequence=train_idx).load()
    valid_data = sdata.isel(_sequence=valid_idx).load()
    test_data = sdata.isel(_sequence=test_idx).load()
    logger.info(f"# training seqs: {train_data.dims['_sequence']}")
    logger.info(f"# validation seqs: {valid_data.dims['_sequence']}")
    logger.info(f"# test seqs: {test_data.dims['_sequence']}\n")

    #-------------- Instantiate the model --------------#
    logger.info("--- Instantiating a BPNet model ---")
    n_filters = params["model"]["n_filters"]
    n_layers = params["model"]["n_layers"]
    n_outputs = 1
    n_control_tracks = 0 if params["model"]["n_control_tracks"] is None else params["model"]["n_control_tracks"]
    alpha = params["model"]["alpha"]
    arch = BPNet(
        n_filters=n_filters,
        n_layers=n_layers,
        n_outputs=n_outputs,
        n_control_tracks=n_control_tracks,
        trimming=trimming,
        alpha=alpha,
        name=prefix,
        verbose=True,
    )
    logger.info(arch)

    #-------------- Train the model --------------#
    logger.info("--- Training the model ---")
    learning_rate = params["training"]["learning_rate"]
    batch_size = params["training"]["batch_size"]
    max_epochs = params["training"]["max_epochs"]
    validation_iter = params["training"]["validation_iter"]
    rng = np.random.default_rng(random_state)
    if params["seqdata"]["ctrl_var"] is not None:
        def train_transform(batch):
            batch['seq'], batch['cov'], batch['ctrl'] = sp.jitter(batch['seq'], batch["cov"], batch["ctrl"], max_jitter=max_jitter, length_axis=-1, jitter_axes=0)
            batch['cov'] = batch['cov'][..., trimming:-trimming]
            batch['ctrl'] = batch['ctrl'][..., trimming:-trimming]
            batch['seq'] = sp.DNA.ohe(batch['seq']).transpose(0, 2, 1)
            if rng.choice(2) == 1:
                batch['seq'] = sp.reverse_complement(batch['seq'], alphabet=sp.DNA, length_axis=-1, ohe_axis=1).copy()
                batch['cov'] = np.flip(batch['cov'], axis=-1).copy()
                batch['ctrl'] = np.flip(batch['ctrl'], axis=-1).copy()
            return batch
    else:
        def train_transform(batch):
            batch['seq'], batch['cov'] = sp.jitter(batch['seq'], batch["cov"], max_jitter=max_jitter, length_axis=-1, jitter_axes=0)  # jitter
            batch['cov'] = batch['cov'][..., trimming:-trimming]  # trim
            batch['seq'] = sp.DNA.ohe(batch['seq']).transpose(0, 2, 1)  # one hot encode
            if rng.choice(2) == 1:
                batch['seq'] = sp.reverse_complement(batch['seq'], alphabet=sp.DNA, length_axis=-1, ohe_axis=1).copy()
                batch['cov'] = np.flip(batch['cov'], axis=-1).copy()
            return batch
    
    # Get the train dataloader
    vars = ['seq', 'cov'] if params["seqdata"]["ctrl_var"] is None else ['seq', 'cov', 'ctrl']
    logger.info(f"Creating the train dataloader with variables: {vars}")
    train_data = train_data.load()  # TODO: remove this line oncewe have faster dl
    train_dl = sd.get_torch_dataloader(
        train_data,
        sample_dims=['_sequence'],
        variables=vars,
        prefetch_factor=None,
        transform=train_transform,
        batch_size=batch_size,
        shuffle=True,
        return_tuples=True
    )
    logger.info(f"Dataloader contains {len(train_dl)} batches")
    batch = next(iter(train_dl))
    if params["seqdata"]["ctrl_var"] is None:
        logger.info(f"Batch looks like: {batch[0].shape}, {batch[1].shape}")
        X_ctl_valid = None
    else:
        logger.info(f"Batch looks like: {batch[0].shape}, {batch[1].shape}, {batch[2].shape}")
        X_ctl_valid = torch.tensor(valid_data["ctrl"].values[..., counts_start:counts_start + target_length], dtype=torch.float32)
    X_valid = torch.tensor(sp.ohe(valid_data["seq"].values[:, seqs_start:seqs_start + seq_length], alphabet=sp.DNA).transpose(0, 2, 1), dtype=torch.float32)
    y_valid = torch.tensor(valid_data["cov"].values[..., counts_start:counts_start + target_length], dtype=torch.float32)
    arch.cuda()
    optimizer = torch.optim.Adam(arch.parameters(), lr=learning_rate)

    # Use the models fit_generator method to train the model
    logger.info("Fitting the model")
    arch.fit(
        train_dl,
        optimizer,
        X_valid=X_valid,
        y_valid=y_valid,
        X_ctl_valid=X_ctl_valid,
        max_epochs=max_epochs,
        validation_iter=validation_iter,
    )
