[![PyPI version](https://badge.fury.io/py/eugene-tools.svg)](https://badge.fury.io/py/eugene-tools)
[![Documentation Status](https://readthedocs.org/projects/eugene-tools/badge/?version=latest)](https://eugene-tools.readthedocs.io/en/latest/?badge=latest)
![PyPI - Downloads](https://img.shields.io/pypi/dm/eugene-tools)
![GitHub stars](https://img.shields.io/github/stars/ML4GLand/EUGENe)

<img src="docs/_static/logos/eugene_logo.png" alt="EUGENe Logo" width=350>

# **E**lucidating the **U**tility of **G**enomic **E**lements with **Ne**ural Nets

EUGENe is a Python package and command line tool for streamlining and customizing end-to-end deep-learning sequence analyses in regulatory genomics.

1. Scalable – data can be loaded out-of-core to substantially reduce memory footprint
2. Fast – Numba accelerated data processing that can be executed lazily (thanks to Dask)
3. Flexible – Xarray is built for N-dimensional data
4. Extensible – Core tools were not designed specifically for one use case

You can find the [current documentation](https://eugene-tools.readthedocs.io/en/latest/index.html) here for getting started.

If you use EUGENe for your research, please cite our preprint: [Klie *et al.* Nature Computational Science 2023](https://www.nature.com/articles/s43588-023-00544-w)


# Roadmap

## `0.2.0` (2024-12-31)
- [ ] `eugene prep-dataset` command for preparing datasets
    - [ ] `eugene prep-dataset tabular` subcommand for preparing datasets from sequence files
    - [ ] `eugene prep-dataset tracks` subcommand for preparing datasets from track files
    - [ ] `eugene prep-dataset regions` subcommand for preparing datasets from region files
- [ ] `eugene fit` command for training models
    - [ ] `eugene fit bpnet-lite` subcommand for using bpnet-lite repo for training
    - [ ] `eugene fit chrombpnet-lite` subcommand for using chrombpnet-lite repo for training
    - [ ] `eugene fit config` subcommand for using custom architectures, losses, metrics and optimizers for training
- [ ] `eugene report` command for generating reports (static or dynamic with dask)
    - [ ] `eugene report prep-dataset` subcommand for generating reports for dataset preparation
    - [ ] `eugene report fit` subcommand for generating reports for model training
- [ ] `tutorials` use cases
    - [ ] `mpra` tutorial on sample of DREAM challenge data
        - [ ] `prep-dataset tabular`
        - [ ] `fit` simple built-in & nasty custom model (DREAM-RNN)
    - [ ] `bulk_atac_basepair` ChromBPNet tutorial
        - [ ] `prep-dataset tracks` 
        - [ ] `bias_fit`
        - [ ] `ChromBPNet fit`
        - [ ] `interpret`
    - [ ] `sc_atac_topics tracks` topic classification tutorial (Joseph)
        - [ ] `prep-dataset`
        - [ ] `fit`
        - [ ] `interpret`
- [ ] `eugene utils` utility commands
    - [ ] `eugene utils splits` subcommand for generating splits or adding them to existing SeqData
    - [ ] `eugene utils negatives` subcommand for generating negatives or adding them to existing SeqData
    - [ ] `eugene utils frag2bw` subcommand for conversion of fragment files to coverage bigwig using frag2bw
    - [ ] `eugene utils bam2bw` subcommand for conversion of bam files to coverage bigwig using bam2bw
    - [ ] `eugene utils peaks` subcommand for generating peaks from coverage bigwigs
