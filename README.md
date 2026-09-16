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

## Input

SpaRMA accepts either a prepared multi-slice `.h5ad` file or slice-level files assembled by the tutorial notebooks. The training object contains:

- `adata.X`: normalized and log-transformed shared-gene matrix;
- `adata.obsm["spatial"]`: spatial coordinates;
- `adata.obs["batch_name"]`: slice or training-group identifier.

Manual spatial-domain annotations are not read during training.

## Tutorials

The `Tutorials/` directory contains complete notebooks for four-slice DLPFC, 12-slice DLPFC, and mouse embryo integration. They cover data loading, preprocessing, spatial-graph construction, model training, post-training clustering and visualization, and saving the integrated object.

Documentation source files are provided in `docs/` and can be built with Sphinx or connected directly to Read the Docs.

## Output

The integrated 30-dimensional representation is written to `adata.obsm["SpaRMA"]`. Training settings and the final calibrated margin are stored in `adata.uns["SpaRMA"]`.

## Reproducibility

The repository contains no expression matrices, annotations, checkpoints, or private paths. Obtain DLPFC data from spatialLIBD and mouse embryo data from MOSTA, then prepare a combined AnnData object with the fields described above.

## Citation

Citation information for SpaRMA is provided in `CITATION.cff`.
