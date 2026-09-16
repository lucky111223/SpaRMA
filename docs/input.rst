Input data
==========

SpaRMA accepts a combined AnnData object containing normalized and log-transformed shared genes. Spatial coordinates must be stored in ``obsm['spatial']`` and slice identifiers in ``obs['batch_name']``. Manual spatial-domain annotations are not used during training.
