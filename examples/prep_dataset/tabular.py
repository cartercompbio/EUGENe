#!/usr/bin/env python3
# coding: utf-8

# Import necessary libraries
import os
import logging
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import seqdata as sd
import seqpro as sp

from eugene.prep_dataset.utils import (
    merge_parameters,
    infer_covariate_types,
    run_continuous_correlations,
    run_binary_correlations,
    run_categorical_correlations,
)
from eugene.prep_dataset.tabular import default_params

import polygraph.sequence
from tangermeme.tools.fimo import fimo
import scanpy as sc
from anndata import AnnData
from scipy.io import mmwrite

from sklearn.model_selection import KFold
from sklearn.decomposition import NMF


def main(args):

    # Path to the parameters file
    path_params = args.params

    # Merge with default parameters
    params = merge_parameters(path_params, default_params)

    # Infer seqpro alphabet
    if params["seqdata"]["alphabet"] == "DNA":
        alphabet = sp.DNA
    elif params["seqdata"]["alphabet"] == "RNA":
        alphabet = sp.RNA

    # Set-up logging with date and time
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Log parameters
    logging.info("Parameters:")
    for key, value in params.items():
        logging.info(f"  {key}")
        for key, value in value.items():
            logging.info(f"    {key}: {value}")

    # Load SeqData
    out = os.path.join(params["base"]["path_out"], f"{params['base']['name']}.seqdata")
    logging.info(f"Writing to {out}")
    seq_col = params["seqdata"]["seq_col"]
    fixed_length = params["seqdata"]["fixed_length"]
    sdata = sd.read_table(
        name=params["seqdata"]["seq_col"],
        out=out,
        tables=params["seqdata"]["tables"],
        seq_col=seq_col,
        fixed_length=fixed_length,
        batch_size=params["seqdata"]["batch_size"],
        overwrite=params["base"]["overwrite"],
    )

    # Splits
    splits = params["splits"]["folds"]
    random_state = params["splits"]["random_seed"]

    # Split into folds such that no two decoded seqs are in the same fold
    seqs_idxs = np.arange(sdata.dims["_sequence"])
    skf = KFold(n_splits=splits, shuffle=True, random_state=random_state)
    train_seq_per_fold = {}
    valid_seq_per_fold = {}
    for i, (train_seq, valid_seq) in enumerate(skf.split(seqs_idxs)):
        train_seq_per_fold[i] = [seqs_idxs[j] for j in train_seq] 
        valid_seq_per_fold[i] = [seqs_idxs[j] for j in valid_seq]

    # Label the folds for each sequence using the idxs
    for i in range(10):
        sdata[f"fold_{i}"] = xr.DataArray(np.zeros(sdata.dims["_sequence"], dtype=bool), dims=["_sequence"])
        sdata[f"fold_{i}"].loc[{"_sequence": train_seq_per_fold[i]}] = True

    # Double check that folds make sense
    sdata[["fold_0", "fold_1", "fold_2", "fold_3", "fold_4", "fold_5", "fold_6", "fold_7", "fold_8", "fold_9"]].to_pandas().describe()

    # Save minimal SeqData
    if os.path.exists(out.replace(".seqdata", ".minimal.seqdata")):
        if params["base"]["overwrite"]:
            import shutil
            logging.info("Removing existing minimal SeqData")
            shutil.rmtree(out.replace(".seqdata", ".minimal.seqdata"))
        else:
            raise ValueError("Minimal SeqData already exists. Set overwrite to true in config to overwrite.")
    sd.to_zarr(sdata, out.replace(".seqdata", ".minimal.seqdata"))

    # Add sequence names
    sdata.coords["_sequence"] = np.array([f"seq_{i}" for i in range(sdata.dims["_sequence"])])

    # One-hot encode
    sdata["ohe"] = xr.DataArray(sp.ohe(sdata[seq_col].values, alphabet=alphabet), dims=["_sequence", "_length", "_alphabet"])
    sdata.coords["_alphabet"] = alphabet.array

    # Get all covariates that isn't fold, seq_col or ohe
    covariates = [col for col in sdata.data_vars if col not in [seq_col, "ohe"]]
    covariates = [col for col in covariates if not col.startswith("fold")]

    # Infer the type of each covariate as categorical, binary, continuous (create a dictionary)
    covariate_types = infer_covariate_types(sdata[covariates].to_pandas())

    # Get the number of sequences and the fixed length of each sequence
    seqs = sdata[params["seqdata"]["seq_col"]].values
    dims = seqs.shape

    # Get seqs as 1D array
    if len(dims) == 2:
        seqs = seqs.view('S{}'.format(dims[1])).ravel().astype(str)

    # Get targets and metadata
    if len(params["seqdata"]["target_cols"]) > 0:
        targets = sdata[params["seqdata"]["target_cols"]].to_pandas()
    else:    
        targets = None
    if len(params["seqdata"]["additional_cols"]) > 0:
        metadata = sdata[params["seqdata"]["additional_cols"]].to_pandas()
    else:
        metadata = None

    ### Basic sequence statistics

    # Sequence length distributions
    sdata["length"] = xr.DataArray(sp.length(seqs), dims=["_sequence"])
    covariate_types["length"] = "continuous"

    # Get unique characters in the sequences with numpy
    unique_chars = np.unique(list("".join(seqs)))

    # Get alphabet and non-alphabet counts
    sdata["alphabet_cnt"] = xr.DataArray(sp.nucleotide_content(seqs, normalize=False, alphabet=alphabet, length_axis=-1), dims=["_sequence", "_alphabet"])
    sdata["non_alphabet_cnt"] = sdata["length"] - sdata["alphabet_cnt"].sum(axis=-1)
    if params["seqdata"]["alphabet"] == "DNA" or params["seqdata"]["alphabet"] == "RNA":
        sdata["gc_percent"] = sdata["alphabet_cnt"].sel(_alphabet=[b"G", b"C"]).sum(axis=-1) / sdata["length"]
        covariate_types["gc_percent"] = "continuous"

    ### K-mer distribution analysis
    ks = params["kmer_analysis"]["k"]
    normalize = params["kmer_analysis"]["normalize"]

    # Structure of output is nested dictionary with 
    # level 1 keys: kmer length, level 1 values: dictionary with
    # level 2 keys: covariate type, level 2 values: dictionary with
    # level 3 keys: covariate name, level 3 values: pandas DataFrame with stats
    kmer_res = {}
    for k in ks:

        # Compute the k-mer frequencies
        kmers = polygraph.sequence.kmer_frequencies(seqs=seqs.tolist(), k=k, normalize=False)

        # Add the k-mer counts to the seqdata
        sdata[f"{k}mer_cnt"] = xr.DataArray(kmers.values, dims=["_sequence", f"_{k}mer"])
        sdata.coords[f"_{k}mer"] = kmers.columns

        # If normalize, normalize the k-mer counts by sequence lengths
        if normalize:
            kmers = kmers.div(sdata["length"].values - k + 1, axis=0)

        # Run PCA on the k-mer counts
        ad = AnnData(kmers, obs=sdata[covariate_types.keys()].to_pandas(), var=sdata[f"_{k}mer"].to_pandas().index.to_frame().drop(f"_{k}mer", axis=1))
        ad = ad[:, ad.X.sum(0) > 0]
        sc.pp.pca(ad)

        # For each covariate, run correlations with each k-mer count
        continuous_res = {}
        binary_res = {}
        categorical_res = {}
        diff_res = {}
        for covariate, _ in covariate_types.items():

            # For each continuous variable, run correlations with each count
            if covariate_types[covariate] == "continuous":
                corrs, pvals = run_continuous_correlations(
                    cnts=sdata[f"{k}mer_cnt"].values,
                    covariate=sdata[covariate].values,
                    method="pearson",
                )
                continuous_res[covariate] = pd.DataFrame(
                    {
                        f"{k}mer": sdata.coords[f"_{k}mer"].values,
                        "corr": corrs,
                        "pval": pvals,
                    }
                )
                continuous_res[covariate] = continuous_res[covariate].sort_values("corr", ascending=False)

            # For each binary variable, run correlations with each count
            elif covariate_types[covariate] == "binary":
                covariate_ = sdata[covariate].values
                covariate_ = np.where(covariate_ == covariate_[0], 0, 1)
                corrs, pvals = run_binary_correlations(
                    cnts=sdata[f"{k}mer_cnt"].values,
                    binary=covariate_,
                    method="mannwhitneyu",
                )
                binary_res[covariate] = pd.DataFrame(
                    {
                        f"{k}mer": sdata.coords[f"_{k}mer"].values,
                        "corr": corrs,
                        "pval": pvals,
                    }
                )
                binary_res[covariate] = binary_res[covariate].sort_values("corr", ascending=False)

            # For each categorical variable, run correlations with each count
            elif covariate_types[covariate] == "categorical":

                # Run the correlation
                corrs, pvals = run_categorical_correlations(
                    cnts=sdata[f"{k}mer_cnt"].values,
                    categorical=sdata[covariate].values,
                    method="kruskal",
                )
                categorical_res[covariate] = pd.DataFrame(
                    {
                        f"{k}mer": sdata.coords[f"_{k}mer"].values,
                        "corr": corrs,
                        "pval": pvals,
                    }
                )
                categorical_res[covariate] = categorical_res[covariate].sort_values("corr", ascending=False)
            
                # Run the differential analysis
                sc.tl.rank_genes_groups(
                    ad,
                    groupby=covariate,
                    groups="all",
                    reference="rest",
                    rankby_abs=True,
                    method="wilcoxon",
                )
                
                # Get the variable names
                diff = pd.DataFrame(ad.uns["rank_genes_groups"]["names"]).melt(var_name="group")

                # Get the statistics
                diff["score"] = pd.DataFrame(ad.uns["rank_genes_groups"]["scores"]).melt()["value"]
                diff["padj"] = pd.DataFrame(ad.uns["rank_genes_groups"]["pvals_adj"]).melt()["value"]
                diff["log2FC"] = pd.DataFrame(ad.uns["rank_genes_groups"]["logfoldchanges"]).melt()["value"]
                diff_res[covariate] = diff

        # Add to results
        kmer_res[k] = {
            "continuous": continuous_res,
            "binary": binary_res,
            "categorical": categorical_res,
            "diff": diff_res,
        }


    ### Motif analysis
    meme_file = params["motif_analysis"]["motif_database"]
    sig = float(params["motif_analysis"]["sig"])

    # Perform FIMO
    X = sp.ohe(seqs, alphabet=alphabet).transpose(0, 2, 1)
    hits = fimo(meme_file, X) 

    # Count up significant occurences of motif
    motif_match_df = pd.concat([hit for hit in hits])
    motif_match_df_ = motif_match_df.loc[motif_match_df["p-value"] < sig]
    logging.info(f"There are {motif_match_df_.shape[0]} significant motif matches.")
    motif_match_df_ = motif_match_df.value_counts(subset=['sequence_name', "motif_name"]).reset_index()
    motif_match_df_.columns = ['sequence_name', "motif_name", 'motif_count']
    motif_match_df_ = motif_match_df_.pivot(index='sequence_name', columns="motif_name", values='motif_count')
    motif_count_df = pd.DataFrame(index=range(len(seqs)), columns=motif_match_df_.columns)
    motif_count_df.loc[motif_match_df_.index.values] = motif_match_df_
    motif_count_df = motif_count_df.fillna(0)

    # Add to seqdata
    sdata["motif_cnt"] = xr.DataArray(motif_count_df.values, dims=["_sequence", "_motif"])
    sdata.coords["_motif"] = motif_count_df.columns.values
    sdata.attrs["motif_database"] = meme_file

    # If normalize, normalize the motif counts by sequence lengths
    if normalize:
        motif_count_df = motif_count_df.div(sdata["length"].values, axis=0)

    # Run PCA on the motif counts
    ad = AnnData(motif_count_df.values, obs=sdata[covariate_types.keys()].to_pandas(), var=pd.DataFrame(index=sdata.coords["_motif"].values))
    ad = ad[:, ad.X.sum(0) > 0]
    sc.pp.pca(ad)

    # normalize counts by sequence length
    n_components = params["motif_analysis"]["n_components"]

    # Run NMF
    model = NMF(n_components=n_components, init="random", random_state=0)

    # Obtain W and H matrices
    W = pd.DataFrame(model.fit_transform(motif_count_df.values))  # seqs x factors
    H = pd.DataFrame(model.components_)  # factors x motifs

    # Format W and H matrices
    factors = [f"factor_{i}" for i in range(n_components)]
    W.index = sdata["_sequence"].values
    W.columns = factors
    H.index = factors
    H.columns = sdata["_motif"].values

    # Add to seqdata
    sdata["seq_scores"] = xr.DataArray(W.values, dims=["_sequence", "_factor"])
    sdata["motif_loadings"] = xr.DataArray(H.values, dims=["_factor", "_motif"])
    sdata.coords["_factor"] = factors

    # Write full SeqData
    if os.path.exists(out.replace(".seqdata", ".full.seqdata")):
        if params["base"]["overwrite"]:
            # remove the directory
            import shutil
            logging.info("Removing existing full SeqData")
            shutil.rmtree(out.replace(".seqdata", ".full.seqdata"))
        else:
            raise ValueError("Full SeqData already exists. Set overwrite to true in config to overwrite.")
    sd.to_zarr(sdata, out.replace(".seqdata", ".full.seqdata"))

    # Write metadata
    metadata.to_csv(out.replace(".seqdata", ".metadata.csv"))

    # Write k-mer data
    for k in ks:
        kmer_cnt = sdata[f"{k}mer_cnt"].values
        mmwrite(out.replace(".seqdata", f".{k}mer_cnt.mtx"), kmer_cnt)
        pd.DataFrame(sdata.coords[f"_{k}mer"].values).to_csv(out.replace(".seqdata", f".{k}mers.tsv.gz"), sep="\t", index=False, header=False, compression="gzip")
        pd.DataFrame(sdata["_sequence"].values).to_csv(out.replace(".seqdata", f".seqs.tsv.gz"), sep="\t", index=False, header=False, compression="gzip")

    # Write motif data
    motif_cnt = sdata["motif_cnt"].values
    mmwrite(out.replace(".seqdata", ".motif_cnt.mtx"), motif_cnt)
    pd.DataFrame(sdata.coords["_motif"].values).to_csv(out.replace(".seqdata", ".motifs.tsv.gz"), sep="\t", index=False, header=False, compression="gzip")

    # 
    logging.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare tabular data for sequence analysis")
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="Path to the parameters file",
    )
    args = parser.parse_args()
    main(args)
