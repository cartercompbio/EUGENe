import os
import logging
import json
import numpy as np
import pandas as pd
import xarray as xr
import seqdata as sd
import seqpro as sp

from .utils import (
    merge_parameters,
    infer_covariate_types,
    run_continuous_correlations,
    run_binary_correlations,
    run_categorical_correlations,
)

import polygraph.sequence
from tangermeme.match import extract_matching_loci
from tangermeme.tools.fimo import fimo
import scanpy as sc
from anndata import AnnData
from scipy.io import mmwrite

from sklearn.model_selection import KFold
from sklearn.decomposition import NMF


logger = logging.getLogger("eugene")

default_params = {
    "seqdata": {
        "batch_size": 1000,
        "overwrite": False,
    }
}


def main(
    path_params,
    path_out,
    overwrite=False,
):

    # Merge with default parameters
    params = merge_parameters(path_params, default_params)

    # Infer seqpro alphabet
    if params["seqdata"]["alphabet"] == "DNA":
        alphabet = sp.DNA
    elif params["seqdata"]["alphabet"] == "RNA":
        alphabet = sp.RNA

    # Log parameters
    logging.info("Parameters:")
    for key, value in params.items():
        logging.info(f"  {key}")
        if isinstance(value, dict):
            for key, value in value.items():
                logging.info(f"    {key}: {value}")

    # Grab params
    name = params["base"]["name"]
    threads = params["base"]["threads"]
    random_state = params["base"]["random_state"]
    fasta = params["seqdata"]["fasta"]
    seq_var = params["seqdata"]["seq_var"]
    bws = params["seqdata"]["bws"]
    bw_names = params["seqdata"]["bw_names"]
    cov_var = params["seqdata"]["cov_var"]
    batch_size = params["seqdata"]["batch_size"]
    fixed_length = params["seqdata"]["fixed_length"]
    max_jitter = params["seqdata"]["max_jitter"]

    #-------------- Load peaks --------------#
    out = os.path.join(path_out, f"{name}.seqdata")
    sdata_peaks = sd.from_region_files(
        sd.GenomeFASTA(
            seq_var,
            fasta,
            batch_size=batch_size,
            n_threads=threads,
        ),
        sd.BigWig(
            cov_var,
            bws,
            bw_names,
            batch_size=batch_size,
            n_jobs=threads,
            threads_per_job=len(bws),
        ),
        path=out,
        fixed_length=fixed_length,
        bed=params["seqdata"]["peaks"],
        overwrite=True,
        max_jitter=max_jitter,
    )
    sdata_peaks["type"] = xr.DataArray(["peak"] * sdata_peaks.dims["_sequence"], dims=["_sequence"])
    sdata_peaks.coords["_sequence"] = np.array([f"peak_{i}" for i in range(sdata_peaks.dims["_sequence"])])
    
    #-------------- Negative loci extraction --------------#
    if "negatives" in params:
        negative_loci = extract_matching_loci(
            loci=params["seqdata"]["peaks"],
            fasta=params["seqdata"]["fasta"],
            gc_bin_width=params["negatives"]["gc_bin_width"],
            max_n_perc=params["negatives"]["max_n_perc"],
            bigwig=params["negatives"]["signal"],
            signal_beta=params["negatives"]["signal_beta"],
            in_window=params["negatives"]["in_window"],
            out_window=params["negatives"]["out_window"],
            chroms=None,
            random_state=random_state,
            verbose=True
        )

        # Write negative loci to bed file
        negatives_bed = os.path.join(path_out, f"{name}.negatives.bed")
        negative_loci.to_csv(negatives_bed, sep="\t", header=False, index=False)

        # Define negative seqdata out path
        negatives_out = os.path.join(path_out, f"{name}.negatives.seqdata")
        
        # Build SeqData from negatives
        sdata_neg = sd.from_region_files(
            sd.GenomeFASTA(
                seq_var,
                fasta,
                batch_size=batch_size,
                n_threads=threads,
            ),
            sd.BigWig(
                cov_var,
                bws,
                bw_names,
                batch_size=batch_size,
                n_jobs=threads,
                threads_per_job=len(bws),
            ),
            path=negatives_out,
            fixed_length=fixed_length,
            bed=negatives_bed,
            overwrite=True,
            max_jitter=max_jitter,
        )
        sdata_neg["type"] = xr.DataArray(["negative"] * sdata_neg.dims["_sequence"], dims=["_sequence"])
        sdata_neg.coords["_sequence"] = np.array([f"negative_{i}" for i in range(sdata_neg.dims["_sequence"])])

        # Concatenate the two datasets
        sdata = xr.concat([sdata_peaks, sdata_neg], dim="_sequence")
        # https://github.com/pydata/xarray/issues/3476#issuecomment-1115045538
        for v in list(sdata.coords.keys()):
            if sdata.coords[v].dtype == object:
                sdata.coords[v] = sdata.coords[v].astype("unicode")
        for v in list(sdata.variables.keys()):
            if sdata[v].dtype == object:
                sdata[v] = sdata[v].astype("unicode")
    else:
        sdata = sdata_peaks

    #-------------- Splits --------------#
    
    # Read splits
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

    # Create dictionary where keys are folds and values are numpy arrays of splits for each sequence in sdata
    for fold in sorted(splits):
        sdata[fold] = xr.DataArray(np.array([df.loc[chrom, fold] for chrom in sdata.chrom.values]), dims=["_sequence"])
    
    #-------------- Save minimal SeqData --------------#
    
    # Save minimal SeqData
    if os.path.exists(out.replace(".seqdata", ".minimal.seqdata")):
        if overwrite:
            import shutil
            logger.info("Removing existing minimal SeqData")
            shutil.rmtree(out.replace(".seqdata", ".minimal.seqdata"))
        else:
            raise ValueError("Minimal SeqData already exists. Set overwrite to true in config to overwrite.")
    sd.to_zarr(sdata, out.replace(".seqdata", ".minimal.seqdata"))
    
    #-------------- OHE --------------#
    sdata["ohe"] = xr.DataArray(sp.ohe(sdata[seq_var].values, alphabet=alphabet), dims=["_sequence", "_length", "_alphabet"])
    sdata.coords["_alphabet"] = alphabet.array

    #-------------- Sequence analysis pipeline --------------#

    # Get the number of sequences and the fixed length of each sequence
    target_length = params["seqdata"]["target_length"]
    seqs_start = (sdata.dims["_length"] // 2) - (fixed_length // 2)
    counts_start = (sdata.dims["_length"] // 2) - (target_length // 2)
    seqs = sdata[seq_var].values
    seqs = seqs[:, seqs_start:seqs_start + fixed_length]
    dims = seqs.shape
    if len(dims) == 2:
        seqs = seqs.view('S{}'.format(dims[1])).ravel().astype(str)

    # Get all vars that are not the sequence or the coverage
    metadata = sdata.drop_vars([seq_var, cov_var, "ohe", "_alphabet", "cov_sample"]).to_dataframe()
    covariate_types = infer_covariate_types(metadata)

    # Sequence length distribution
    sdata["length"] = xr.DataArray(sp.length(seqs), dims=["_sequence"])
    covariate_types["length"] = "continuous"

    # Get unique characters in the sequences with numpy
    #unique_chars = np.unique(list("".join(seqs)))
    sdata["alphabet_cnt"] = xr.DataArray(sp.nucleotide_content(seqs, normalize=False, alphabet=alphabet, length_axis=-1), dims=["_sequence", "_alphabet"])
    sdata["non_alphabet_cnt"] = sdata["length"] - sdata["alphabet_cnt"].sum(axis=-1)
    if params["seqdata"]["alphabet"] == "DNA" or params["seqdata"]["alphabet"] == "RNA":
        sdata["gc_percent"] = sdata["alphabet_cnt"].sel(_alphabet=[b"G", b"C"]).sum(axis=-1) / sdata["length"]
        covariate_types["gc_percent"] = "continuous"

    # Summed counts
    cov = sdata[cov_var].values
    sdata["total_counts"] = xr.DataArray(cov[..., counts_start:counts_start + target_length].sum(axis=(1,2)), dims=["_sequence"])
    metadata["total_counts"] = sdata["total_counts"].values
    
    # k-mer distribution analysis
    ks = params["kmer_analysis"]["k"]
    normalize = params["kmer_analysis"]["normalize"]
    selected_covariates = params["kmer_analysis"]["selected_covariates"]
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
        ad.write_h5ad(f"{out.replace('.seqdata', '')}.{k}mer.h5ad")

        # For each covariate, run correlations with each k-mer count
        continuous_res = {}
        binary_res = {}
        categorical_res = {}
        diff_res = {}
        for covariate in selected_covariates:
            print(f"Running correlations for {k}-mers with {covariate}")
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

    # Motif analysis
    meme_file = params["motif_analysis"]["motif_database"]
    sig = float(params["motif_analysis"]["sig"])
    
    # Perform FIMO
    X = sp.ohe(seqs, alphabet=alphabet).transpose(0, 2, 1)
    hits = fimo(meme_file, X) 

    # Count up significant occurences of motif
    motif_match_df = pd.concat([hit for hit in hits])
    motif_match_df_ = motif_match_df.loc[motif_match_df["p-value"] < sig]
    print(f"There are {motif_match_df_.shape[0]} significant motif matches.")
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
    motif_ad = AnnData(motif_count_df.values, obs=sdata[covariate_types.keys()].to_pandas(), var=pd.DataFrame(index=sdata.coords["_motif"].values))
    motif_ad = motif_ad[:, motif_ad.X.sum(0) > 0]
    sc.pp.pca(motif_ad)
    motif_ad.write_h5ad(f"{out.replace('.seqdata', '')}.motif.h5ad")

    # NMF
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

    sdata["seq_scores"] = xr.DataArray(W.values, dims=["_sequence", "_factor"])
    sdata["motif_loadings"] = xr.DataArray(H.values, dims=["_factor", "_motif"])
    sdata.coords["_factor"] = factors

    # Write full SeqData
    if os.path.exists(out.replace(".seqdata", ".full.seqdata")):
        if overwrite:
            # remove the directory
            import shutil
            logger.info("Removing existing full SeqData")
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
    logger.info("Done!")
    