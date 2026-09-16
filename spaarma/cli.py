import argparse, json
from pathlib import Path
import anndata as ad
import numpy as np
from .graph import spatial_edges
from .training import SpaRMAConfig, fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg_raw = json.loads(args.config.read_text(encoding="utf-8"))
    radius = cfg_raw.pop("radius")
    batch_key = cfg_raw.pop("batch_key", "batch_name")
    cfg = SpaRMAConfig(**cfg_raw)
    adata = ad.read_h5ad(args.input)
    edges = spatial_edges(adata.obsm["spatial"], adata.obs[batch_key].astype(str), radius)
    x = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    embedding, settings, _ = fit(x, edges, adata.obs[batch_key].astype(str).to_numpy(), cfg)
    adata.obsm["SpaRMA"] = embedding; adata.uns["SpaRMA"] = settings
    args.output.parent.mkdir(parents=True, exist_ok=True); adata.write_h5ad(args.output)


if __name__ == "__main__": main()
