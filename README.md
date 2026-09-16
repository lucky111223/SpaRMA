# SpaRMA

SpaRMA integrates multiple spatial transcriptomics slices with two training stages: masked spot reconstruction pretraining and clean-input joint training with multi-positive attention alignment.

## Method

Stage I replaces complete spot profiles with a learned mask token, zeros the corresponding latent rows before decoding, and evaluates mean squared reconstruction error only on masked spots. Stage II restores the clean input, refreshes mutual-nearest-neighbor correspondences, weights up to `K` positive candidates with trainable query and key projections, and jointly optimizes reconstruction and attention-weighted triplet loss.

The graph auto-encoder implementation is based on and modified from STAligner. SpaRMA adds the two-stage masking workflow and multi-positive attention alignment used in the accompanying manuscript.

## Installation

```bash
conda create -n spaarma python=3.10
conda activate spaarma
pip install -e .
```

Install a PyTorch build appropriate for the local CUDA version before installing PyTorch Geometric when GPU acceleration is required.

## Input

Each example accepts one `.h5ad` file containing all slices after preprocessing:

- `adata.X`: normalized and log-transformed shared-gene matrix;
- `adata.obsm["spatial"]`: spatial coordinates;
- `adata.obs["batch_name"]`: slice or training-group identifier.

Manual spatial-domain annotations are not read during training.

## Examples

```bash
python examples/run_dlpfc4.py --input data/dlpfc4.h5ad --output results/dlpfc4.h5ad
python examples/run_dlpfc12.py --input data/dlpfc12.h5ad --output results/dlpfc12.h5ad
python examples/run_mouse_embryo.py --input data/mouse_embryo.h5ad --output results/mouse_embryo.h5ad
```

Dataset settings are stored in `configs/`. DLPFC4 uses `K=3`; DLPFC12 and mouse embryo use `K=15`. Training runs for 500 masked-pretraining epochs followed by 500 clean joint-training epochs.

## Output

The integrated 30-dimensional representation is written to `adata.obsm["SpaRMA"]`. Training settings and the final calibrated margin are stored in `adata.uns["SpaRMA"]`.

## Reproducibility

The repository contains no expression matrices, annotations, checkpoints, or private paths. Obtain DLPFC data from spatialLIBD and mouse embryo data from MOSTA, then prepare a combined AnnData object with the fields described above.

## Citation

Please cite the SpaRMA manuscript and STAligner when using this implementation.

## License

MIT. The original STAligner copyright and permission notice are retained in `LICENSE`.
