import os
import pickle
import pandas as pd
import numpy as np
import seqdata as sd
import seqpro as sp
import yaml
import logging
import xarray as xr
import json
import tqdm.auto as tqdm
import pyBigWig
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging

from eugene.utils import merge_parameters

logger = logging.getLogger("eugene")

default_params = {
    "seqdata": {
        "seq_var": "seq",
        "batch_size": 1000,
        "fixed_length": 501,
        "alphabet": "DNA",
        "max_jitter": 128,
        "threads": 1,
        "random_state": 1234
    },
    "loci": None,
    "remove_non_overlap_regions" : False,
    "target_name": "targets"
}

def main( 
    path_params,
    path_out,
    report=False,
    overwrite=True,
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
    
    # Infer seqpro alphabet
    if params["seqdata"]["alphabet"] == "DNA":
        alphabet = sp.DNA
    elif params["seqdata"]["alphabet"] == "RNA":
        alphabet = sp.RNA
    
    # Grab parameters
    path_in = params["bed_dir"]
    name = params["name"]
    fasta = params["seqdata"]["fasta"]
    seq_var = params["seqdata"]["seq_var"]
    batch_size = params["seqdata"]["batch_size"]
    fixed_length = params["seqdata"]["fixed_length"]
    max_jitter = params["seqdata"]["max_jitter"]
    threads = params["seqdata"]["threads"]
    random_state = params["seqdata"]["random_state"]
    loci = params["loci"]
    bed_dir = params["bed_dir"]
    remove_non_overlap_regions = params["remove_non_overlap_regions"]
    target_name = params["target_name"]
    
    #-------------- Create BED file containing all regions --------------#
    logger.info(f"--- Compiling all regions into a single BED file ---")
    
    # Concatenate regions (with repeats present)
    bed = []
    for f in tqdm.tqdm(os.listdir(bed_dir)):
        if f.endswith(".bed"):
            overlap_name = f.split(".")[0]
            curr_bed = pd.read_csv(
                    os.path.join(bed_dir, f),
                    sep="\t",
                    header=None,
                    names=["chrom", "chromStart", "chromEnd"],
                )
            curr_bed["name"] = overlap_name
            bed.append(curr_bed)
    bed_df = pd.concat(bed, axis=0)
    bed_df.to_csv(os.path.join(path_out, "all_regions.bed"), sep="\t", header=False, index=False)
    logger.info(f"\tSuccessfully compiled regions. Saving BED file to {path_out}")
        
    #-------------- Build SeqData --------------#
    logger.info(f"--- Building SeqData ---")
    
    # Create separate directory for SeqData
    sdata_dir = os.path.join(path_out, f"{name}.seqdata")
    generate = True
    
    # read in SeqData if overwrite is False
    if os.path.exists(sdata_dir):
        if not overwrite:
            logger.info("\tSeqData already exists. Set overwrite to true in config to overwrite.")
            generate = False
            logger.info(f"\tLoading existing SeqData from {os.path.basename(sdata_dir)}")
            sdata = sd.open_zarr(sdata_dir)
            
    # Construct initial SeqData
    if generate:
        if loci:
            logger.info(f"\tBuilding SeqData from loci file at {loci}")
            sdata = sd.from_region_files(
                sd.GenomeFASTA(
                    name = seq_var,
                    fasta = fasta,
                    batch_size = batch_size,
                    n_threads = threads,
                ),
                path = sdata_dir,
                fixed_length = fixed_length,
                bed = loci,
                overwrite = overwrite,
                max_jitter = max_jitter,
            )
        else:
            logger.info(f"\tNo loci file provided")
            # merge duplicate regions, concatenate names
            union_df = bed_df.groupby(["chrom", "chromStart", "chromEnd"]).agg({"name": lambda x: ";".join(x)}).reset_index()
            union_df.to_csv(os.path.join(path_out, "union.bed"), sep="\t", header=False, index=False)
            sdata = sd.from_region_files(
                    sd.GenomeFASTA(
                        name = seq_var,
                        fasta = fasta,
                        batch_size = batch_size,
                        n_threads = threads,
                    ),
                    path = sdata_dir,
                    fixed_length = fixed_length,
                    bed = os.path.join(path_out, "union.bed"),
                    overwrite = overwrite,
                    max_jitter = max_jitter,
                )

        # Construct labels and add to SeqData
        sdata[target_name] = sd.label_overlapping_regions(
            sdata,
            os.path.join(path_out, "all_regions.bed"),
            mode="multitask",
            label_dim="_targets"
        )
       
        # Remove regions that do not overlap
        if remove_non_overlap_regions:
            logger.info(f"\tRemoving non-overlapping regions")
            non_overlap_mask = sdata[target_name].sum(dim="_targets") > 0
            sdata = sdata.sel(_sequence = non_overlap_mask)
            total_removed = (~non_overlap_mask).sum().item()
            logger.info(f"\tRemoved {total_removed} non-overlapping regions")
        else: 
            logger.info(f"\tKeeping non-overlapping regions")
       
        #-------------------------------------#
        logger.info(f"--- Preprocessing SeqData ---")

        sdata["seq"] = sdata["seq"].str.upper()
        mask = np.array([False if b"N" in seq else True for seq in sdata["seq"].values])
        sdata = sdata.sel(_sequence=mask)
        sdata["ohe_seq"] = xr.DataArray(sp.ohe(sdata[seq_var].values, alphabet=alphabet), dims=["_sequence", "_length", "_ohe"])
        sdata["ohe_seq"] = sdata["ohe_seq"].transpose("_sequence", "_ohe", "_length")
        logger.info(f"\tSuccessfully built SeqData. Saved to {sdata_dir}")
    
    #-------------- Splits --------------#
    
    # Read splits
    logger.info(f"--- Reading splits from {params['splits']} ---")
    splits = params["splits"]
    with open(params["splits"], "r") as infile:
        splits = json.load(infile)

    # Turn the above structure into a dataframe where rows are chromosomes and columns are the fold
    unique_chroms = set()
    folds = []
    for fold in splits:
        unique_chroms.update(splits[fold]["train"])
        unique_chroms.update(splits[fold]["valid"])
        unique_chroms.update(splits[fold]["test"])
        folds.append(fold)
    df = pd.DataFrame(index=sorted(unique_chroms), columns=sorted(folds))
    for fold in splits:
        for chrom in splits[fold]["train"]:
            df.loc[chrom, fold] = "train"
        for chrom in splits[fold]["valid"]:
            df.loc[chrom, fold] = "valid"
        for chrom in splits[fold]["test"]:
            df.loc[chrom, fold] = "test"
    
    # Check and warn for chromosomes present in data but not in splits.
    missing_chroms = set(sdata.chrom.values) - set(df.index)
    if missing_chroms:
        logger.warning(f"The following chromosomes are not included in the splits: {missing_chroms}")
        chrom_values = sdata["chrom"].values
        chrom_values = chrom_values.astype(str)
        mask = np.isin(chrom_values, df.index)
        sdata = sdata.sel(_sequence=mask)
        logger.info(f"\tThe following chromosomes were successfully removed: {missing_chroms}")
    
    # Create dictionary where keys are folds and values are numpy arrays of splits for each sequence in sdata
    for fold in sorted(splits):
        sdata[fold] = xr.DataArray(np.array([df.loc[chrom, fold] for chrom in sdata.chrom.values]), dims=["_sequence"])
    logger.info(f"\tSuccessfully added splits to SeqData\n")
    
    #-------------- Save minimal SeqData --------------#
    
    # Save minimal SeqData
    minimal_out = os.path.join(path_out, f"{name}.minimal.seqdata")
    generate = True        
    if os.path.exists(minimal_out):
        if not overwrite:
            logger.info("Minimal SeqData already exists. Set overwrite to true in config to overwrite.")
            generate = False
            logger.info(f"\tLoading existing minimal SeqData from {os.path.basename(out.replace('.seqdata', '.minimal.seqdata'))}")
            sdata = sd.open_zarr(minimal_out)
    if generate:
        sd.to_zarr(sdata, minimal_out, mode="w")
        logger.info(f"Saved minimal SeqData to {minimal_out}\n")