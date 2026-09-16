# SpaRMA

SpaRMA integrates multiple spatial transcriptomics slices with two training stages: masked spot reconstruction pretraining and clean-input joint training with multi-positive attention alignment.

## Method

Stage I replaces complete spot profiles with a learned mask token, zeros the corresponding latent rows before decoding, and evaluates mean squared reconstruction error only on masked spots. Stage II restores the clean input, refreshes mutual-nearest-neighbor correspondences, weights up to `K` positive candidates with trainable query and key projections, and jointly optimizes reconstruction and attention-weighted triplet loss.

## Installation

```bash
conda create -n spaarma python=3.10
conda activate spaarma
pip install -e .
```

Install a PyTorch build appropriate for the local CUDA version before installing PyTorch Geometric when GPU acceleration is required.

For the complete notebooks, create the tested Conda environment with `conda env create -f environment.yml`, or install the tutorial dependencies with `pip install -e ".[tutorial]"`. DLPFC clustering with mclust additionally requires R, the R package `mclust`, and `rpy2`.

## Tutorials

The `Tutorials/` directory contains complete notebooks for four-slice DLPFC, 12-slice DLPFC, and mouse embryo integration. They cover data loading, preprocessing, spatial-graph construction, model training, clustering, and visualization.

Documentation source files are provided in `docs/` and can be built with Sphinx or connected directly to Read the Docs.

## Citation

Citation information for SpaRMA is provided in `CITATION.cff`.
