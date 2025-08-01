import anndata as ad
import networkx as nx
import scanpy as sc
import scglue
import numpy as np
from matplotlib import rcParams

scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)

dataset = "Chen-2019"
rna = ad.read_h5ad(f"./dataset/{dataset}/raw/RNA.h5ad")
atac = ad.read_h5ad(f"./dataset/{dataset}/raw/ATAC.h5ad")

#RNA processed
rna.layers["counts"] = rna.X.copy()
sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor="seurat_v3")
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
sc.pp.scale(rna)
sc.tl.pca(rna, n_comps=50, svd_solver="auto")
sc.pp.neighbors(rna, metric="cosine")
sc.tl.umap(rna)

#ATAC processed
scglue.data.lsi(atac, n_components=50, n_iter=15)
sc.pp.neighbors(atac, use_rep="X_lsi", metric="cosine")
sc.tl.umap(atac)

#scglue.data.get_gene_annotation(
#    rna, gtf="./src/bedtools/annotation/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz",
#    gtf_by="gene_name"
#)
#rna.var.loc[:, ["chrom", "chromStart", "chromEnd"]].head()
split = atac.var_names.str.split(r"[:-]")
atac.var["chrom"] = split.map(lambda x: x[0])
atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
atac.var.head()

guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
scglue.graph.check_graph(guidance, [rna, atac])
rna.write(f"./dataset/{dataset}/rna-pp.h5ad", compression="gzip")
atac.write(f"./dataset/{dataset}/atac-pp.h5ad", compression="gzip")
nx.write_graphml(guidance, f"./dataset/{dataset}/guidance.graphml.gz")