
import anndata as ad
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import scanpy as sc
import pandas as pd
import numpy as np
from scMODCI.models  import configure_dataset
import scglue
from itertools import chain
import dill
from scMODCI.metrics import (
    avg_silhouette_width,
    neighbor_conservation,
    graph_connectivity
)
from matplotlib import rcParams
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)



dataset = "Chen-2019"
result_path = f"./dataset/{dataset}/result"

rna = ad.read_h5ad(f"./dataset/{dataset}/rna-pp.h5ad")
atac = ad.read_h5ad(f"./dataset/{dataset}/atac-pp.h5ad")
guidance = nx.read_graphml(f"./dataset/{dataset}/guidance.graphml.gz")
print(atac)
configure_dataset(
   rna, "NB", use_highly_variable=True,
   use_layer="counts", use_rep="X_pca"
)
configure_dataset(
    atac, "NB", use_highly_variable=True,
    use_rep="X_lsi"
)

modal_names=["RNA", "ATAC"]
guidance_hvf = guidance.subgraph(chain(
    rna.var.query("highly_variable").index,
    atac.var.query("highly_variable").index
)).copy()
with open(f'{result_path}/scMODCI.dill', 'rb') as f:
    glue = dill.load(f)
rna.obsm["X_scMODCI"] = glue.encode_data("rna", rna)
atac.obsm["X_scMODCI"] = glue.encode_data("atac", atac)


combined = ad.concat([rna, atac])
print(combined)


sc.pp.neighbors(combined, use_rep="X_scMODCI", metric="cosine")
sc.tl.umap(combined)
sc.pl.umap(combined, color=["cell_type", "domain"], wspace=0.65)
combined.obs["domain"] = pd.Categorical(
        combined.obs["domain"],
)




##############################

embedding = combined.obsm["X_scMODCI"]  # (n_cells, latent_dim)

# 获取 cell type / domain
cell_types = combined.obs["cell_type"].values
domains   = combined.obs["domain"].values
print(domains)


gc_score=graph_connectivity(embedding, cell_types)

uni_data = np.concatenate([rna.obsm["X_pca"], atac.obsm["X_lsi"]])
nc_score = neighbor_conservation(embedding, uni_data, domains)

asw_score = avg_silhouette_width(embedding, cell_types)



print(f"Graph Connectivity Metric: {gc_score}")

print(f"Neighbor Conservation: {nc_score:.4f}")

print(f"Average Silhouette Width (ASW, cell_type): {asw_score:.4f}")


