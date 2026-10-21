# SpaRMA

SpaRMA integrates multiple spatial transcriptomics slices with two training stages: masked spot reconstruction pretraining and clean-input joint training with multi-positive attention alignment.

## Method

Stage I replaces complete spot profiles with a learned mask token, zeros the corresponding latent rows before decoding, and evaluates mean squared reconstruction error only on masked spots. Stage II restores the clean input, refreshes mutual-nearest-neighbor correspondences, weights up to `K` positive candidates with trainable query and key projections, and jointly optimizes reconstruction and attention-weighted triplet loss.

## Installation

Implementation details and the scope of the public-release validation are in
[implementation alignment](docs/implementation_alignment.md). The small-input
tests do not constitute a rerun of the manuscript experiments.

```bash
conda create -n spaarma python=3.10
conda activate spaarma
pip install -e .
```

Install a PyTorch build appropriate for the local CUDA version before installing PyTorch Geometric when GPU acceleration is required.

For the notebooks, create a Conda environment with `conda env create -f environment.yml`, or install the tutorial dependencies with `pip install -e ".[tutorial]"`. This environment specification is not a lockfile of the original experiments; see the validation scope below. DLPFC clustering with mclust additionally requires R, the R package `mclust`, and `rpy2`.

## Tutorials

The `Tutorials/` directory contains complete notebooks for four-slice DLPFC, 12-slice DLPFC, and mouse embryo integration. They cover data loading, preprocessing, spatial-graph construction, model training, clustering, and visualization.

Documentation source files are provided in `docs/` and can be built with Sphinx or connected directly to Read the Docs.

## Reproduction scope

The notebooks demonstrate the full SpaRMA model; they are not a complete runner
for the four ablation configurations and 15-run summaries in Supplementary
Tables S1 and S2. In particular, setting `mask_rate=0` is not a clean-pretraining
ablation: the current masking helper selects at least one spot per batch.
Do not use this setting to claim reproduction of the baseline.

The supplementary material records the original experiment environment
(including Python 3.8.20, PyTorch 2.1.0+cu121, Scanpy 1.9.8 and AnnData 0.9.2).
The public package's Python 3.10 installation recipe is a separate environment,
not an assertion of identical outputs. Full notebook training and the manuscript
benchmark scores have not been revalidated across all allowed dependency versions.

DLPFC preprocessing must select HVGs on raw counts, normalize and log-transform
the full count matrix, then subset HVGs and inner-concatenate slices. Run
`python tests/tutorial_preprocessing_test.py` to check this order and the
resulting matrix on small diagnostic inputs (requires Scanpy).

## Citation

Citation information for SpaRMA is provided in `CITATION.cff`.
