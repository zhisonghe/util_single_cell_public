import scvelo as scv
from scipy.io import mmwrite

def run_scvelo_pt(filename, use_rep, n_neighbors = 30, saved_h5ad_filename = None, do_filter = True, min_shared_counts = 10, n_top_genes = 3000):
  adata = scv.read(filename, cache=False) # all cells
  
  if do_filter: scv.pp.filter_genes(adata, min_shared_counts = min_shared_counts)
  scv.pp.normalize_per_cell(adata)
  if do_filter: scv.pp.filter_genes_dispersion(adata, n_top_genes = n_top_genes)
  scv.pp.log1p(adata)
  
  scv.pp.moments(adata, n_neighbors = n_neighbors, use_rep = use_rep) # all cells
  scv.tl.velocity(adata)
  scv.tl.velocity_graph(adata)
  scv.tl.recover_dynamics(adata)
  scv.tl.recover_latent_time(adata)
  scv.tl.velocity_pseudotime(adata)
  if saved_h5ad_filename is not None: adata.write(saved_h5ad_filename)
  
  return adata.obs.loc[:,["latent_time","velocity_pseudotime"]]
