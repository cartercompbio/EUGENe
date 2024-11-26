import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os
from pathlib import Path
from dominate import document
from dominate.tags import h1, h2, p, img

# Function to create sequence length distribution plot
def plot_sequence_length_distribution(seq_df, output_dir):
    plt.figure(figsize=(12, 6))
    sns.histplot(seq_df["length"], bins=100, kde=True)
    plt.xlabel("Sequence length")
    plt.ylabel("Count")
    plt.title("Sequence length distribution")
    plot_path = os.path.join(output_dir, "sequence_length_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    return "sequence_length_distribution.png"

# Function to create GC content distribution plot
def plot_gc_content_distribution(seq_df, output_dir):
    plt.figure(figsize=(12, 6))
    sns.histplot(seq_df, x="gc_percent", hue="type", bins=50, kde=True)
    plt.xlabel("GC Content")
    plt.ylabel("Density")
    plt.title("GC Content in Peaks and Negatives")
    plot_path = os.path.join(output_dir, "gc_content_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    return "gc_content_distribution.png"

# Function to compute KS statistics for GC content
def compute_gc_content_ks_test(seq_df):
    loci_gc = seq_df.query("type == 'region'")["gc_percent"].values
    matched_gc = seq_df.query("type == 'negative'")["gc_percent"].values
    stats = ks_2samp(loci_gc, matched_gc)
    return stats

# Function to create character percentage distribution plot
def plot_character_distribution(seq_df, output_dir):
    char_colors = {
        "A": "green",
        "C": "blue",
        "G": "orange",
        "T": "red",
        "non_alphabet_cnt": "grey"
    }
    plt.figure(figsize=(12, 6))
    melted_df = seq_df.melt(value_vars=["A", "C", "G", "T", "non_alphabet_cnt"])
    sns.violinplot(x="value", y="variable", data=melted_df, palette=char_colors)
    plt.xlabel("Non-alphabet character percentage")
    plt.ylabel("Character")
    plt.title("Character percentage distribution")
    plot_path = os.path.join(output_dir, "character_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    return "character_distribution.png"

# Function to create total counts distribution plot
def plot_total_counts_distribution(seq_df, output_dir):
    seq_df["log_total_counts"] = np.log1p(seq_df["total_counts"])
    plt.figure(figsize=(10, 6))
    sns.histplot(seq_df, x="log_total_counts", hue="type", bins=50)
    plt.xlabel("log(Sum of Counts) in Peaks and Negatives")
    plt.ylabel("Density")
    plt.title("Total Counts Distribution")
    plot_path = os.path.join(output_dir, "total_counts_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    return "total_counts_distribution.png"

# Function to compute statistics for total counts
def compute_total_counts_stats(seq_df):
    loci_df = seq_df.query("type == 'region'")
    neg_df = seq_df.query("type == 'negative'")
    max_loci = loci_df["total_counts"].max()
    min_loci = loci_df["total_counts"].min()
    max_neg = neg_df["total_counts"].max()
    min_neg = neg_df["total_counts"].min()
    outlier_threshold = 0.9999
    upper_thresh_neg = np.quantile(neg_df["total_counts"], outlier_threshold)
    lower_thresh_neg = np.quantile(neg_df["total_counts"], 1 - outlier_threshold)
    counts_loss_weight = np.median(
        neg_df["total_counts"][
            (neg_df["total_counts"] < upper_thresh_neg) & (neg_df["total_counts"] > lower_thresh_neg)
        ]
    ) / 10
    return {
        "max_loci": max_loci,
        "min_loci": min_loci,
        "max_neg": max_neg,
        "min_neg": min_neg,
        "upper_thresh_neg": upper_thresh_neg,
        "lower_thresh_neg": lower_thresh_neg,
        "counts_loss_weight": counts_loss_weight
    }

# Function to create GC content vs. total counts scatter plot
def plot_gc_vs_total_counts(seq_df, output_dir):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="gc_percent", y="log_total_counts", hue="type", data=seq_df)
    plt.xlabel("GC Content")
    plt.ylabel("log(Sum of Counts)")
    plt.title("GC Content vs. log(Sum of Counts)")
    plot_path = os.path.join(output_dir, "gc_vs_total_counts.png")
    plt.savefig(plot_path)
    plt.close()
    return "gc_vs_total_counts.png"

# Master function to generate HTML report
def generate_html_report(seq_df, output_dir="report_output"):
    Path(output_dir).mkdir(exist_ok=True)
    
    length_plot = plot_sequence_length_distribution(seq_df, output_dir)
    gc_plot = plot_gc_content_distribution(seq_df, output_dir)
    gc_stats = compute_gc_content_ks_test(seq_df)
    char_plot = plot_character_distribution(seq_df, output_dir)
    total_counts_plot = plot_total_counts_distribution(seq_df, output_dir)
    total_counts_stats = compute_total_counts_stats(seq_df)
    scatter_plot = plot_gc_vs_total_counts(seq_df, output_dir)
    
    report_path = os.path.join(output_dir, "report.html")
    with document(title="Sequence Analysis Report") as doc:
        h1("Sequence Analysis Report")
        
        h2("1. Sequence Length Distribution")
        img(src=length_plot, width="600px")
        
        h2("2. GC Content Distribution")
        img(src=gc_plot, width="600px")
        p(f"KS Test Statistic: {gc_stats.statistic:.3f}, p-value: {gc_stats.pvalue:.3g}")
        
        h2("3. Character Percentage Distribution")
        img(src=char_plot, width="600px")
        
        h2("4. Total Counts Distribution")
        img(src=total_counts_plot, width="600px")
        p(f"Max Loci Counts: {total_counts_stats['max_loci']}, Min Loci Counts: {total_counts_stats['min_loci']}")
        p(f"Max Negative Counts: {total_counts_stats['max_neg']}, Min Negative Counts: {total_counts_stats['min_neg']}")
        p(f"Upper Threshold Negative Counts: {total_counts_stats['upper_thresh_neg']}, "
          f"Lower Threshold Negative Counts: {total_counts_stats['lower_thresh_neg']}")
        p(f"Suggested Bias Counts Loss Weight: {total_counts_stats['counts_loss_weight']:.3f}")
        
        h2("5. GC Content vs. log(Sum of Counts)")
        img(src=scatter_plot, width="600px")
    
    with open(report_path, "w") as f:
        f.write(doc.render())
    
    print(f"Report generated at: {report_path}")
    return report_path
