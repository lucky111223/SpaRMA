import argparse
from pathlib import Path
import anndata as ad
import numpy as np
from .graph import spatial_edges
from .presets import get_preset
from .training import fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preset", required=True, choices=["dlpfc4", "dlpfc12", "mouse_embryo"])
    args = parser.parse_args()
    preset = get_preset(args.preset)
    radius, batch_key, cfg = preset.radius, preset.batch_key, preset.training
    adata = ad.read_h5ad(args.input)
    edges = spatial_edges(adata.obsm["spatial"], adata.obs[batch_key].astype(str), radius)
    x = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    batches = adata.obs[batch_key].astype(str).to_numpy()
    alignment_batches = (
        adata.obs[preset.alignment_key].astype(str).to_numpy()
        if preset.alignment_key is not None
        else batches
    )
    embedding, settings, _ = fit(
        x,
        edges,
        batches,
        cfg,
        alignment_batches=alignment_batches,
        batch_pairs=preset.batch_pairs,
    )
    adata.obsm["SpaRMA"] = embedding; adata.uns["SpaRMA"] = settings
    args.output.parent.mkdir(parents=True, exist_ok=True); adata.write_h5ad(args.output)


if __name__ == "__main__": main()
