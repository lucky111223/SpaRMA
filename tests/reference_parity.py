"""Optional parity checks against an author-supplied STAligner source directory.

Usage: PYTHONPATH=. python tests/reference_parity.py /path/to/staligner_official
No experimental input or output is modified.
"""
import ast
import sys
from pathlib import Path
import hashlib
import numpy as np
import torch
import pandas as pd
import anndata as ad
import networkx as nx
import hnswlib
from spaarma.alignment import build_candidates, keyed_negatives, calibrate_margin

root = Path(sys.argv[1])
def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace

import itertools
ns = dict(np=np,pd=pd,nx=nx,hnswlib=hnswlib,itertools=itertools)
load_functions(root/'STAligner/mnn_utils.py', ['nn_approx','mnn','create_dictionary_mnn'], ns)
rng=np.random.default_rng(12)
x=rng.normal(size=(18,4)).astype('float32')
batches=np.repeat(['a','b','c'],6)
names=np.array([f'spot{i}' for i in range(18)])
adata=ad.AnnData(x,obs=pd.DataFrame({'batch_name':batches},index=names))
adata.obsm['z']=x
reference=ns['create_dictionary_mnn'](adata,'z','batch_name',k=4,verbose=0)
lookup={n:i for i,n in enumerate(names)}
expected=[(pair,lookup[a],[lookup[p] for p in ps[:2]]) for pair,group in reference.items() for a,ps in group.items()]
a,p,v,pairs=build_candidates(x,batches,2,4,spot_names=names,return_pairs=True)
actual=[(pair,int(anchor),row[valid].tolist()) for pair,anchor,row,valid in zip(pairs,a,p,v)]
assert actual == expected, 'Candidate sequence differs from reference'
from typing import Optional, Tuple
ns2=dict(np=np,hashlib=hashlib,Optional=Optional,torch=torch,Tuple=Tuple)
load_functions(root/'analysis/dlpfc4_positive_centroid_ablation.py',['official_random_negative'],ns2)
negative=keyed_negatives(a,batches,names,pairs,42,500)
for i,anchor in enumerate(a):
    pool=names[batches==batches[anchor]]
    expected_name=ns2['official_random_negative'](pool,f'42|500|{pairs[i]}|{names[anchor]}')
    assert names[negative[i]] == expected_name
load_functions(root/'analysis/dlpfc4_k2_attention_strength_runner.py',['calibrated_margin'],ns2)
for target in [0.,.2,.5,.8,1.]:
    gaps=torch.tensor([-1.,0.,0.,1.,2.],dtype=torch.float32)
    expected,_=ns2['calibrated_margin'](gaps,target)
    actual=calibrate_margin(gaps,torch.ones_like(gaps,dtype=torch.bool),target)
    assert actual == expected
print('reference_candidate_negative_margin_parity=PASS')
