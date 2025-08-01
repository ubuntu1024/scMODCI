import os
import time

import anndata as ad
import networkx as nx
from itertools import chain

from scMODCI.models import configure_dataset,scMODCI_trainer


dataset = "Chen-2019"
result_path = f"./dataset/{dataset}/result"
if(not os.path.exists(result_path)):
            os.makedirs(result_path)
rna = ad.read_h5ad(f"./dataset/{dataset}/rna-pp.h5ad")
atac = ad.read_h5ad(f"./dataset/{dataset}/atac-pp.h5ad")
guidance = nx.read_graphml(f"./dataset/{dataset}/guidance.graphml.gz")

configure_dataset(
    rna, "NB", use_highly_variable=True,
    use_layer="counts", use_rep="X_pca"
)
configure_dataset(
    atac, "NB", use_highly_variable=True,
    use_rep="X_lsi"
)


guidance_hvf = guidance.subgraph(chain(
    rna.var.query("highly_variable").index,
    atac.var.query("highly_variable").index
)).copy()

starttime = time.time()


scMODCI = scMODCI_trainer(
    {"rna": rna, "atac": atac},guidance_hvf,
    fit_kws={"directory": f"{result_path}/scMODCI"}
)
scMODCI.save(f"{result_path}/scMODCI.dill")



endtime = time.time()
print("#########################################################################")
print('spent {:.5f} s'.format(endtime-starttime))
print("#########################################################################")

