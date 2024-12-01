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

import matplotlib.pyplot as plt
import seaborn as sns

from tangermeme.predict import predict
from bpnetlite.chrombpnet import ChromBPNet
from bpnetlite.bpnet import BPNet, CountWrapper, ProfileWrapper, ControlWrapper, _ProfileLogitScaling
from tangermeme.deep_lift_shap import deep_lift_shap, _nonlinear
import modiscolite
from modiscolite.util import calculate_window_offsets

from .utils import (
    merge_parameters,
    check_for_gpu,
)
from .report import generate_html_report
from ..plot.performance import scatter
from ..plot.bpnet_lite import plot_training_curves

logger = logging.getLogger("eugene")

default_params = {
    "seqdata": {
        "ctrl_var": None,
    },
    "model": {
        "n_control_tracks": None,
    },
    "training": {
        "early_stopping": None,
    },
}


def main(
    path_params,
    path_out,
    report=False,
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

    # Check for GPU
    device = check_for_gpu()
    if device == torch.device("cpu"):
        raise ValueError("Cannot currently train bpnet-lite models on CPU. Exiting.")
    
    #-------------- Load SeqData --------------#
    logger.info("--- Loading SeqData ---")
    path_seqdata = params["seqdata"]["path"]
    seq_var = params["seqdata"]["seq_var"]
    cov_var = params["seqdata"]["cov_var"]
    ctrl_var = params["seqdata"]["ctrl_var"]
    fold = params["seqdata"]["fold"]
    seq_length = params["seqdata"]["seq_length"]
    target_length = params["seqdata"]["target_length"]
    max_jitter = params["seqdata"]["max_jitter"]
    max_counts = params["seqdata"]["max_counts"]
    min_counts = params["seqdata"]["min_counts"]
    outlier_threshold = params["seqdata"]["outlier_threshold"]

    # Load the data
    logger.info(f"Reading sequence data from {path_seqdata}")
    sdata = sd.open_zarr(path_seqdata)

    # Define the trimming
    trimming = (seq_length - target_length) // 2
    seqs_start = (sdata.dims["_length"] // 2) - (seq_length // 2)
    counts_start = (sdata.dims["_length"] // 2) - (target_length // 2)
    logger.info(f"Sequence length is {seq_length}")
    logger.info(f"Target length is {target_length}")
    logger.info(f"Trimming is {trimming}\n")

    # Check if prefix.torch exists, if it does and overwrite is False, raise an error
    prefix = os.path.join(path_out, name)
    generate = True
    if not overwrite and os.path.exists(prefix + ".torch"):
        logger.info(f"{prefix}.torch already exists. Set overwrite=True to overwrite.")
        generate = False
        logger.info("Loading model from file, will only evaluate on test data.")
    if generate:
        #-------------- Split the data --------------#
        logger.info("--- Splitting the data ---")
        training_idx = np.where(sdata[fold].isin(["train", "valid"]))[0]
        training_data = sdata.isel(_sequence=training_idx)
        logger.info(f"# training seqs: {training_data.dims['_sequence']}")

        # Sample neg_sampling_ratio negative sequences (type == "negative")
        neg_sampling_ratio = params["seqdata"]["neg_sampling_ratio"]
        if neg_sampling_ratio is not None:
            logger.info(f"Sampling {neg_sampling_ratio} negative sequences")
            neg_idx = np.where(sdata.type.values == "negative")[0]
            loci_idx = np.where(sdata.type.values == "loci")[0]
            neg_idx = np.random.choice(neg_idx, size=int(neg_sampling_ratio * len(loci_idx)), replace=False)
            training_data = sdata.isel(_sequence=sorted(np.concatenate([loci_idx, neg_idx])))
            logger.info(f"# training seqs after sampling negative sequences: {training_data.dims['_sequence']}")
        
        # -------------- Filter the data --------------#
        logger.info("--- Filtering the data ---")
        logger.info(f"Filtering based on total counts between {min_counts} and {max_counts}")
        training_cov = training_data[cov_var].values
        training_counts = training_cov[..., counts_start:counts_start + target_length].sum(axis=(1,2))
        counts_msk = (training_counts <= max_counts) & (training_counts >= min_counts)
        if outlier_threshold is not None:
            logger.info(f"Further filtering based on outlier threshold of {outlier_threshold}")
            filt_counts = training_counts[counts_msk]
            upper_thresh = np.quantile(filt_counts, outlier_threshold)
            lower_thresh = np.quantile(filt_counts, 1 - outlier_threshold)
            outlier_msk = (training_counts <= upper_thresh) & (training_counts >= lower_thresh)
        else:
            outlier_msk = counts_msk
        training_data = training_data.isel(_sequence=outlier_msk)
        logger.info(f"# training seqs after filtering: {training_data.dims['_sequence']}\n")

        #-------------- Instantiate the model --------------#
        print("--- Instantiating a ChromBPNet model ---")
        n_filters = params["model"]["n_filters"]
        n_layers = params["model"]["n_layers"]
        n_outputs = 1
        n_control_tracks = 0 if params["model"]["n_control_tracks"] is None else params["model"]["n_control_tracks"]
        path_bias = params["model"]["bias_model"]

        # Load the bias model
        bias_model = torch.load(path_bias, map_location=device)

        # Get alpha
        alpha = params["model"]["alpha"]
        if alpha is None:
            print("Computing counts loss weight")
            alpha = np.median(training_counts[outlier_msk]) / 10
            print(f"Counts loss weight is {alpha}\n")

        # Accessiblity model
        accessibility_model = BPNet(
            n_filters=n_filters,
            n_layers=n_layers,
            n_outputs=n_outputs,
            n_control_tracks=n_control_tracks,
            trimming=trimming,
            alpha=alpha,
            name=prefix + ".accessibility",
            verbose=True,
        )

        # Full ChromBPNet model
        arch = ChromBPNet(bias=bias_model, accessibility=accessibility_model, name=prefix)
        logger.info(f"Full model: {arch}")

        #-------------- Train the model --------------#
        logger.info("--- Training the model ---")
        learning_rate = params["training"]["learning_rate"]
        batch_size = params["training"]["batch_size"]
        max_epochs = params["training"]["max_epochs"]
        validation_iter = params["training"]["validation_iter"]
        rc_augment = params["training"]["rc_augment"]
        early_stopping = params["training"]["early_stopping"]
        rng = np.random.default_rng(random_state)
        if ctrl_var is not None:
            def transform(batch):
                if max_jitter > 0:
                    batch[seq_var], batch[cov_var], batch[ctrl_var] = sp.jitter(batch[seq_var], batch[cov_var], batch[ctrl_var], max_jitter=max_jitter, length_axis=-1, jitter_axes=0)
                    batch[cov_var] = batch[cov_var][..., trimming:-trimming]
                    batch[ctrl_var] = batch[ctrl_var][..., trimming:-trimming]
                else:
                    batch[seq_var] = batch[seq_var][..., seqs_start:seqs_start + seq_length]
                    batch[cov_var] = batch[cov_var][..., counts_start:counts_start + target_length]
                    batch[ctrl_var] = batch[ctrl_var][..., counts_start:counts_start + target_length]
                batch[seq_var] = sp.DNA.ohe(batch[seq_var]).transpose(0, 2, 1)
                if rc_augment and rng.choice(2) == 1:
                    batch[seq_var] = sp.reverse_complement(batch[seq_var], alphabet=sp.DNA, length_axis=-1, ohe_axis=1).copy()
                    batch[cov_var] = np.flip(batch[cov_var], axis=-1).copy()
                    batch[ctrl_var] = np.flip(batch[ctrl_var], axis=-1).copy()
                return batch
        else:
            def transform(batch):
                if max_jitter > 0:
                    batch[seq_var], batch[cov_var] = sp.jitter(batch[seq_var], batch[cov_var], max_jitter=max_jitter, length_axis=-1, jitter_axes=0)  # jitter
                    batch[cov_var] = batch[cov_var][..., trimming:-trimming]  # trim
                else:
                    batch[seq_var] = batch[seq_var][..., seqs_start:seqs_start + seq_length]
                    batch[cov_var] = batch[cov_var][..., counts_start:counts_start + target_length]
                batch[seq_var] = sp.DNA.ohe(batch[seq_var]).transpose(0, 2, 1)  # one hot encode
                if rc_augment and rng.choice(2) == 1:
                    batch[seq_var] = sp.reverse_complement(batch[seq_var], alphabet=sp.DNA, length_axis=-1, ohe_axis=1).copy()
                    batch[cov_var] = np.flip(batch[cov_var], axis=-1).copy()
                return batch
        
        # Get the train dataloader
        vars = [seq_var, cov_var] if ctrl_var is None else [seq_var, cov_var, ctrl_var]
        logger.info(f"Creating the train dataloader with variables: {vars}")
        train_idx = np.where(training_data[fold] == "train")[0]
        train_data = training_data.isel(_sequence=train_idx).load()
        train_dl = sd.get_torch_dataloader(
            train_data,
            sample_dims=['_sequence'],
            variables=vars,
            prefetch_factor=None,
            transform=transform,
            batch_size=batch_size,
            shuffle=True,
            return_tuples=True
        )
        logger.info(f"Dataloader contains {len(train_dl)} batches")
        batch = next(iter(train_dl))

        # Get the validation data
        valid_idx = np.where(training_data[fold] == "valid")[0]
        valid_data = training_data.isel(_sequence=valid_idx)
        X_valid = torch.tensor(sp.ohe(valid_data[seq_var].values[:, seqs_start:seqs_start + seq_length], alphabet=sp.DNA).transpose(0, 2, 1), dtype=torch.float32)
        y_valid = torch.tensor(valid_data[cov_var].values[..., counts_start:counts_start + target_length], dtype=torch.float32)
        logger.info(f"Validation data shapes: {X_valid.shape}, {y_valid.shape}")

        # Move the model to the GPU and prepare the optimizer
        arch.cuda()
        optimizer = torch.optim.Adam(arch.parameters(), lr=learning_rate)
        logger.info(f"Optimizer: {optimizer}")

        # Use the models fit_generator method to train the model
        logger.info("Fitting the model")
        arch.fit(
            train_dl,
            optimizer,
            X_valid=X_valid,
            y_valid=y_valid,
            max_epochs=max_epochs,
            validation_iter=validation_iter,
            early_stopping=early_stopping,
        )

    
    #-------------- Test data evaluation the model --------------#
    if "evaluation" in params:
        logger.info("--- Evaluating the model on test data ---")
        batch_size = params["evaluation"]["batch_size"]

        # Load the model
        logger.info("Loading best model from file")
        arch = torch.load(prefix + ".torch").cuda()

        # log dataframe
        plot_training_curves(
            prefix + ".log", 
            alpha=arch.accessibility.alpha,
            ax=None,
            save=prefix + "_loss.png"
        )

        # Test data        
        test_idx = np.where(sdata[fold] == "test")[0]
        test_data = sdata.isel(_sequence=test_idx)
        logger.info(f"# test seqs: {test_data.dims['_sequence']}\n")
        
        # Get predictions
        test_out = os.path.join(path_out, f"{name}.test.seqdata")
        generate = True
        if os.path.exists(test_out):
            if not overwrite:
                logger.info("Test SeqData already exists. Set overwrite to true in config to overwrite.")
                generate = False
                logger.info(f"Loaded test SeqData from {test_out}")
                test_data = sd.open_zarr(test_out)
        if generate:
            logger.info("Predicting on test data")
            true_counts = torch.tensor(test_data[cov_var].values[..., counts_start:counts_start + target_length].sum(axis=-1), dtype=torch.float32)
            test_data["log_counts"] = xr.DataArray(torch.log(true_counts+1).numpy(), dims=["_sequence", "cov_sample"])
            X_test = torch.tensor(sp.ohe(test_data[seq_var].values[:, seqs_start:seqs_start + seq_length], alphabet=sp.DNA).transpose(0, 2, 1), dtype=torch.float32)
            y_profiles, y_counts = predict(arch, X_test, batch_size=batch_size, device="cuda", verbose=True)
            test_data[f"{fold}_log_counts_pred"] = xr.DataArray(y_counts.cpu().numpy(), dims=["_sequence", "cov_sample"])
            test_data[f"{fold}_profile_pred"] = xr.DataArray(y_profiles.cpu().numpy(), dims=["_sequence", "cov_sample", "_target_length"])     
            sd.to_zarr(test_data, test_out, mode="w")        
            logger.info(f"Saved test SeqData to {test_out}")

            # Scatter plots
            logger.info("Creating scatter plots")
            type_msk = (test_data["type"] == "loci")
            loci_counts = test_data["log_counts"].values[type_msk]
            loci_counts_pred = test_data[f"{fold}_log_counts_pred"].values[type_msk]
            scatter(
                loci_counts,
                loci_counts_pred,
                figsize=(4, 4),
                xlabel="Observed Log Counts",
                ylabel="Predicted Log Counts",
                save=prefix + "_loci_counts_scatter.png",
                density=False,
                add_reference_line=False,
                return_axes=False,
            )
            neg_counts = test_data["log_counts"].values[~type_msk]
            neg_counts_pred = test_data[f"{fold}_log_counts_pred"].values[~type_msk]
            scatter(
                neg_counts,
                neg_counts_pred,
                figsize=(4, 4),
                xlabel="Observed Log Counts",
                ylabel="Predicted Log Counts",
                save=prefix + "_neg_counts_scatter.png",
                density=False,
                add_reference_line=False,
                return_axes=False,
            )
            counts = test_data["log_counts"].values
            counts_pred = test_data[f"{fold}_log_counts_pred"].values
            scatter(
                counts,
                counts_pred,
                figsize=(4, 4),
                xlabel="Observed Log Counts",
                ylabel="Predicted Log Counts",
                save=prefix + "_counts_scatter.png",
                density=False,
                add_reference_line=False,
                return_axes=False,
            )

    #-------------- Attribution --------------#
    if "attribution" in params:
        logger.info("--- Computing attribution on 30k subsample ---")
        batch_size = params["attribution"]["batch_size"]
        subsample = params["attribution"]["subsample"]
        n_shuffles = params["attribution"]["n_shuffles"]
        
        # Write test SeqData
        attr_out = os.path.join(path_out, f"{name}.sub.attr.seqdata")
        generate = True
        if os.path.exists(attr_out):
            if not overwrite:
                logger.info("Subsampled SeqData already exists. Set overwrite to true in config to overwrite.")
                generate = False
                logger.info(f"Loading attribution subsample from {attr_out}")
                test_loci = sd.open_zarr(attr_out)
                X = test_loci["X_ohe"].values
                X_attr_counts = test_loci["X_attr_counts"].values
                X_attr_profile = test_loci["X_attr_profile"].values
        
        if generate:
            # Subsample loci
            logger.info(f"Targeting {subsample} loci for attributions")
            test_loci_idx = np.where(test_data["type"] == "loci")[0]
            test_loci = test_data.isel(_sequence=test_loci_idx)
            if test_loci.dims["_sequence"] > subsample:
                random_idx = np.random.choice(test_loci.dims["_sequence"], subsample, replace=False)
                test_loci = test_loci.isel(_sequence=sorted(random_idx))
            X = torch.tensor(sp.ohe(test_loci["seq"].values[:, seqs_start:seqs_start + seq_length], alphabet=sp.DNA).transpose(0, 2, 1), dtype=torch.float32)
            n_msk = X.sum(dim=(1, 2)) == X.shape[-1]
            X = X[n_msk]
            n_idx = np.where(n_msk)[0]
            test_loci = test_loci.isel(_sequence=n_idx)
            logger.info(f"Subsampled {X.shape[0]} loci with no non-alphabet characters")
        
            # Get count attributions
            logger.info("Computing count attributions")
            count_wrapper = CountWrapper(ControlWrapper(arch.accessibility)).cuda().eval()
            dtype = torch.float64
            X_attr_counts = deep_lift_shap(
                count_wrapper.type(dtype), 
                X.type(dtype),
                hypothetical=True,
                n_shuffles=n_shuffles,
                batch_size=batch_size,
                verbose=True,
                warning_threshold=1e-4
            )

            # Get profile attributions
            logger.info("Computing profile attributions")
            profile_wrapper = ProfileWrapper(ControlWrapper(arch.accessibility)).cuda().eval()
            X_attr_profile = deep_lift_shap(
                profile_wrapper.type(dtype), 
                X.type(dtype),
                hypothetical=True,
                additional_nonlinear_ops={_ProfileLogitScaling: _nonlinear}, 
                n_shuffles=n_shuffles,
                batch_size=batch_size,
                verbose=True,
                warning_threshold=1e-4
            )
            
            # Save attributions
            X = X.cpu().numpy()
            X_attr_counts = X_attr_counts.cpu().numpy()
            X_attr_profile = X_attr_profile.cpu().numpy()
            test_loci["X_ohe"] = xr.DataArray(X, dims=["_sequence", "_alphabet", "_trimmed_length"])
            test_loci["X_attr_counts"] = xr.DataArray(X_attr_counts, dims=["_sequence", "_alphabet", "_trimmed_length"])
            test_loci["X_attr_profile"] = xr.DataArray(X_attr_profile, dims=["_sequence", "_alphabet", "_trimmed_length"])
            sd.to_zarr(test_loci, attr_out, mode="w")
            logger.info(f"Saved attribution subset SeqData to {attr_out}")
                

    if "modisco" in params:
        logger.info("--- Running TFMoDISco ---")
        n_seqlets = params["modisco"]["n_seqlets"]
        window = params["modisco"]["window"]
        motif_db = params["modisco"]["motif_db"]
        
        # Get sequences for modisco
        center = X.shape[2] // 2
        start, end = calculate_window_offsets(center, window)
        sequences = X[:, :, start:end].transpose(0, 2, 1).astype('float32')

        # Counts modisco
        modisco_out = os.path.join(path_out, f"{name}_modisco_counts.h5")
        generate = True
        if os.path.exists(modisco_out):
            if not overwrite:
                logger.info("TFMoDISco output already exists. Set overwrite to true in config to overwrite.")
                generate = False
                logger.info(f"Loading TFMoDISco results from {modisco_out}")
        if generate:
            logger.info("Using count attributions for TFMoDISco")
            attributions = X_attr_counts[:, :, start:end].transpose(0, 2, 1).astype('float32')
            pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
                hypothetical_contribs=attributions, 
                one_hot=sequences,
                max_seqlets_per_metacluster=n_seqlets,
                sliding_window_size=20,
                flank_size=5,
                target_seqlet_fdr=0.05,
                n_leiden_runs=2,
                verbose=True
            )
            modiscolite.io.save_hdf5(prefix + "_modisco_counts.h5", pos_patterns, neg_patterns, window)
            logger.info(f"Saved count attributions TFMoDISco results to {prefix}_modisco_counts.h5")
        
        # Profile modisco
        modisco_out = os.path.join(path_out, f"{name}_modisco_profile.h5")
        generate = True
        if os.path.exists(modisco_out):
            if not overwrite:
                logger.info("TFMoDISco output already exists. Set overwrite to true in config to overwrite.")
                generate = False
                logger.info(f"Loading TFMoDISco results from {modisco_out}")
        if generate:
            logger.info("Using profile attributions for TFMoDISco")
            attributions = X_attr_profile[:, :, start:end].transpose(0, 2, 1).astype('float32')
            pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
                hypothetical_contribs=attributions, 
                one_hot=sequences,
                max_seqlets_per_metacluster=n_seqlets,
                sliding_window_size=20,
                flank_size=5,
                target_seqlet_fdr=0.05,
                n_leiden_runs=2,
                verbose=True
            )
            modiscolite.io.save_hdf5(prefix + "_modisco_profile.h5", pos_patterns, neg_patterns, window)
            logger.info(f"Saved profile attributions TFMoDISco results to {prefix}_modisco_profile.h5")
    
        logger.info("--- Creating TFMoDISco reports ---")
        report_out = os.path.join(path_out, f"{name}_modisco_counts_report")
        generate = True
        if os.path.exists(report_out):
            if not overwrite:
                logger.info("Counts TFMoDISco report already exists. Set overwrite to true in config to overwrite.")
                generate = False
        if generate:
            modiscolite.report.report_motifs(
                modisco_h5py=prefix + "_modisco_counts.h5",
                output_dir=prefix + "_modisco_counts_report/",
                img_path_suffix=prefix + "_modisco_counts_report/",
                meme_motif_db=motif_db,
                is_writing_tomtom_matrix=False,
                top_n_matches=3
            )
            logger.info(f"Saved counts TFMoDISco report to {prefix}_modisco_counts_report")

        report_out = os.path.join(path_out, f"{name}_modisco_profile_report")
        generate = True
        if os.path.exists(report_out):
            if not overwrite:
                logger.info("Profile TFMoDISco report already exists. Set overwrite to true in config to overwrite.")
                generate = False
        if generate:
            modiscolite.report.report_motifs(
                modisco_h5py=prefix + "_modisco_profile.h5",
                output_dir=prefix + "_modisco_profile_report/",
                img_path_suffix=prefix + "_modisco_profile_report/",
                meme_motif_db=motif_db,
                is_writing_tomtom_matrix=False,
                top_n_matches=3
            )
            logger.info(f"Saved profile TFMoDISco report to {prefix}_modisco_profile_report")

    # TODO: Generate a static report
    if report:
        generate = True
        report_out = os.path.join(path_out, f"{name}.report.html")
        if os.path.exists(report_out):
            if not overwrite:
                logger.info("Static report already exists. Set overwrite to true in config to overwrite.")
                generate = False
        if generate:
            path_report = generate_html_report(
                name,
                path_out, 
                f"{name}.report.html"
            )
            logger.info(f"Generated static report at {path_report}")

    logger.info("Done!")
