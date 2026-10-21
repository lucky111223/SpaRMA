"""Execute notebook preprocessing cells on small dense/sparse diagnostic inputs.

HVG selection is stubbed to make the expected intersection deterministic; real
Scanpy normalization/log1p and AnnData concatenation are exercised. This is not
a full-dataset experiment or a test of the Seurat HVG estimator itself.
"""
import json
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def source(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in notebook["cells"]
             if c["cell_type"] == "code"]
    for cell in cells:
        compile(cell, str(path), "exec")
    matches = [c for c in cells if "sc.pp.highly_variable_genes(" in c]
    assert len(matches) == 1
    return matches[0]


for name, top in [("Tutorial_DLPFC4.ipynb", 5000),
                  ("Tutorial_DLPFC12.ipynb", 10000)]:
    code = source(ROOT / "Tutorials" / name)
    for use_sparse in (False, True):
        slices = []
        for i, genes in enumerate([["z", "a", "b", "c"],
                                   ["c", "b", "z", "a"]]):
            x = np.array([[10, 20, 30, 100], [3, 4, 5, 80]], dtype=np.float32)
            if use_sparse:
                x = sparse.csr_matrix(x)
            slices.append(ad.AnnData(x, obs=pd.DataFrame(
                {"batch_name": [str(i)] * 2}, index=[f"{i}_0", f"{i}_1"]),
                var=pd.DataFrame(index=genes)))
        originals = [s.copy() for s in slices]
        calls = []

        def choose_hvgs(sample, flavor, n_top_genes):
            assert flavor == "seurat_v3" and n_top_genes == top
            i = int(sample.obs["batch_name"].iloc[0])
            # HVGs must be chosen on the unmodified full raw-count input.
            np.testing.assert_array_equal(dense(sample.X), dense(originals[i].X))
            assert list(sample.var_names) == list(originals[i].var_names)
            selected = {"z", "a", "b"} if i == 0 else {"z", "a", "c"}
            sample.var["highly_variable"] = sample.var_names.isin(selected)
            calls.append(i)

        expected_parts = []
        for i, raw in enumerate(originals):
            selected = {"z", "a", "b"} if i == 0 else {"z", "a", "c"}
            x = dense(raw.X)
            transformed = np.log1p(x * (1e4 / x.sum(axis=1))[:, None])
            expected = ad.AnnData(transformed, obs=raw.obs.copy(), var=raw.var.copy())
            expected_parts.append(expected[:, expected.var_names.isin(selected)].copy())
        expected = ad.concat(expected_parts, join="inner")

        # Execute twice to ensure rerunning the notebook cell does not renormalize
        # or log-transform the source slices a second time.
        for _ in range(2):
            ns = dict(slices=slices, sc=sc, ad=ad)
            with patch.object(sc.pp, "highly_variable_genes", choose_hvgs):
                exec(code, ns)
            actual = ns["adata"]
            assert list(actual.var_names) == list(expected.var_names)
            assert list(actual.obs_names) == list(expected.obs_names)
            np.testing.assert_allclose(dense(actual.X), dense(expected.X), rtol=1e-6)
            for before, after in zip(originals, slices):
                np.testing.assert_array_equal(dense(before.X), dense(after.X))
            # The previous subset-first algorithm must fail this numerical oracle.
            raw = originals[0][:, expected.var_names]
            x = dense(raw.X)
            wrong = np.log1p(x * (1e4 / x.sum(axis=1))[:, None])
            assert not np.allclose(dense(actual.X)[:2], wrong)
        assert calls == [0, 1, 0, 1]
print("tutorial_preprocessing=PASS (dense/sparse; order; values; rerun safety)")
