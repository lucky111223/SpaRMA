# Implementation alignment

This maintenance update aligns the public implementation with the executed
training mechanisms. It does not change manuscript scores or claim that the
paper's full training experiments have been rerun with this packaged release.

- Spatial graphs retain up to 50 neighbors strictly inside the radius, plus self
  loops, independently for each slice.
- MNN candidates are constructed separately for each batch pair. The same spot
  may contribute multiple anchor constraints. Truncation is per pair, not global.
- Candidate order follows the STAligner mutual-pair set to NetworkX sparse
  adjacency construction. There is no additional distance ranking. Original spot
  identifiers and their input order must be preserved. Set iteration and the
  approximate neighbor backend can depend on the Python environment and runtime;
  do not interpret a seed alone as a bitwise reproduction guarantee.
- The default neighbor backend is HNSW (L2); `exact` is an explicit diagnostic
  alternative, not a claim about the original run. Neighbor counts are clipped
  only for small test inputs with fewer spots than the requested count.
- Negative draws use the seed, absolute training epoch, batch-pair name and anchor
  identifier in a BLAKE2b key, and sample from the anchor-side alignment batch.
- Query/key normalization has a norm floor of 0.001. Pretraining clips model and
  mask-token gradients separately at 5; joint training clips model and attention
  parameters together at 5.
- Margin calibration uses anchor-level active rates. The single-positive target
  uses the dataset reference margin (1 for DLPFC, 2.5 for embryo); calibration
  candidates and tie breaking use 1.0, independently of that target reference.

Run `PYTHONPATH=. python tests/smoke_test.py` and
`PYTHONPATH=. python tests/contract_test.py` for small-input checks.
These checks are not repeated-training benchmarks or proof of exact numerical
reproduction of selected manuscript runs.

## DLPFC tutorial preprocessing correction (2026-10-22)

The earlier public DLPFC notebooks subsetted shared HVGs before normalization.
The corrected notebooks follow the manuscript and original experiment loaders:
select HVGs on raw counts, normalize the full matrix to 10,000 counts per spot,
apply log1p, subset each slice's HVGs, then inner-concatenate. Gene order is left
to the same input-derived AnnData concatenation rather than explicitly sorted;
duplicate gene names use the original loader's `++` suffix convention.
Raw input slices are copied so that rerunning preprocessing does not transform
already transformed values.

`python tests/tutorial_preprocessing_test.py` executes both notebook
preprocessing cells on dense and sparse diagnostic inputs. It checks HVG-call
arguments and raw input, normalization values against an independent formula,
gene/spot order, and rerun safety. HVG selection is stubbed in this test; this
does not validate the Seurat estimator or rerun full-data training.

The preprocessing test, smoke test, contract test and optional original-source
candidate/negative/margin parity test passed for this maintenance change.
Core model/training code and reported manuscript scores were not modified.
