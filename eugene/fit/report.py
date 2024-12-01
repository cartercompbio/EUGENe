import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os
from pathlib import Path
from dominate.tags import h1, h2, h3, div, img, p
from dominate.util import raw
from dominate import document


def generate_html_report(
    prefix: str,
    path_out: str,
    filename: str
):
    
    # Validate path_out and ensure filename is within it
    if not os.path.isdir(path_out):
        raise ValueError(f"The specified output directory does not exist: {path_out}")
    
    save_path = os.path.join(path_out, filename)

    # Relative paths to images and logs
    loss_plot_path = f"{prefix}_loss.png"
    counts_scatter_path = f"{prefix}_counts_scatter.png"
    loci_counts_scatter_path = f"{prefix}_loci_counts_scatter.png"
    neg_counts_scatter_path = f"{prefix}_neg_counts_scatter.png"

    # Relative paths to motif discovery reports
    modisco_counts_report_path = os.path.join(f"{prefix}_modisco_counts_report", "motifs.html")
    modisco_profile_report_path = os.path.join(f"{prefix}_modisco_profile_report", "motifs.html")
    
    # Read and preprocess motif discovery tables
    def preprocess_motif_table(table_path, prefix):
        full_path = os.path.join(path_out, table_path)
        if not os.path.exists(full_path):
            return ""
        table_html = open(full_path).read()
        # Example ./K562_ATAC-seq_bias_fold_0_modisco_counts_report/
        # path out is ./
        # Prefix is K562_ATAC-seq_bias_fold_0
        table_html = (table_html.replace("width=\"240\"", "width=\"240\", class=\"cover\"")
                                .replace(">pos_patterns.pattern", ">pos_")
                                .replace(">neg_patterns.pattern", ">neg_")
                                .replace("modisco_cwm_fwd", "cwm_fwd")
                                .replace("modisco_cwm_rev", "cwm_rev")
                                .replace("num_seqlets", "NumSeqs")
                                .replace("dataframe", "new")
                                .replace(os.path.abspath(path_out), "."))  # Replace absolute paths with relative
        return remove_negs(table_html)
    
    def remove_negs(tables):
        new_lines = []
        set_flag = True
        lines = tables.split("\n")
        jdx = 0
        for idx in range(len(lines) - 1):
            if jdx == 15:
                set_flag = True
                jdx = 0
            if "neg_" in lines[idx + 1]:
                set_flag = False
                jdx = 0
            if set_flag:
                new_lines.append(lines[idx])
            else:
                jdx += 1
        new_lines.append(lines[-1])
        return "\n".join(new_lines)
    
    table_counts = preprocess_motif_table(modisco_counts_report_path, "modisco_counts")
    table_profile = preprocess_motif_table(modisco_profile_report_path, "modisco_profile")
    
    # Generate HTML report
    doc = document(title="Model Training Report")
    
    with doc:
        h1("Model Training Report")
        
        # Section 1: Training Log
        h2("1. Training Log")
        p("This section includes details of the model's training process, such as loss over epochs.")
        if os.path.exists(os.path.join(path_out, loss_plot_path)):
            img(src=loss_plot_path, width="600px", alt="Loss Plot")
        else:
            p("Training log not found.")
        
        # Section 2: Performance
        h2("2. Performance")
        p("This section includes scatter plots to evaluate the model's performance metrics.")
        with div(style="display: flex; flex-wrap: nowrap; justify-content: space-around;"):
            with div(style="text-align: center; margin: 10px;"):
                p("Counts Scatter")
                img(src=counts_scatter_path, width="300px", alt="Counts Scatter")
            with div(style="text-align: center; margin: 10px;"):
                p("Loci Counts Scatter")
                img(src=loci_counts_scatter_path, width="300px", alt="Loci Counts Scatter")
            with div(style="text-align: center; margin: 10px;"):
                p("Negative Counts Scatter")
                img(src=neg_counts_scatter_path, width="300px", alt="Negative Counts Scatter")
        
        # Section 3: Motif Discovery
        h2("3. Motif Discovery")
        p("This section provides the motifs discovered by the model during training. Both counts and profile motifs are displayed below.")
        
        h3("Counts Modisco")
        div(raw(table_counts))
        
        h3("Profile Modisco")
        div(raw(table_profile))
    
    # Save the report
    with open(save_path, "w") as f:
        f.write(doc.render())
    
    return save_path
