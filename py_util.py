import numpy as np
import pandas as pd

import scipy
from scipy import sparse
from scipy.stats import norm
from scipy.stats import rankdata
from sklearn.preprocessing import scale
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import NMF, PCA, non_negative_factorization

import scanpy as sc
import igraph
import anndata as ad

# Run NMF on the input matrix X
def run_NMF(X, model = None, W = None, H = None, n_components = 10, max_iter = 200, return_model = False, **kwargs):
  if model is None or (not isinstance(model, NMF)):
    if H is None and W is None:
      model = NMF(n_components = int(n_components), max_iter = int(max_iter), **kwargs)
      W = model.fit_transform(X)
      H = model.components_
    else:
      W, H, n_iter = non_negative_factorization(X, W = W, H = H, n_components = H.shape[0], max_iter = int(max_iter), **kwargs)
  else:
    W = model.transform(X)
    H = model.components_
  
  if return_model and model is not None:
    return(model)
  else:
    return({"W" : W, "H" : H})

# Run PAGA given the path to the h5ad file
def run_PAGA_only(path, rep, k, grp, overwrite = True):
  h5ad = sc.read_h5ad(path)
  sc.pp.neighbors(h5ad, n_neighbors=k, use_rep=rep)
  sc.tl.paga(h5ad, groups=grp)
  if overwrite:
    h5ad.write_h5ad(path)
  return h5ad

# Calculate correlation between columns in matrix X and those in matrix Y, with X and Y being dense or sparse
def corSparse(X, Y = None):
  if Y is None: Y = X
  
  n = X.shape[0]
  muX = np.ravel(X.mean(0))
  muY = np.ravel(Y.mean(0))
  covmat = ( X.T.dot(Y) - (n * muX[:,np.newaxis].dot(muY[:,np.newaxis].T)) ) / (n-1)
  sdvecX = np.ravel(np.sqrt(((X.power(2)).sum(0) - n*(muX**2)) / (n-1)) if sparse.issparse(X) else np.sqrt(((X**2).sum(0) - n*(muX**2)) / (n-1)))
  sdvecY = np.ravel(np.sqrt(((Y.power(2)).sum(0) - n*(muY**2)) / (n-1)) if sparse.issparse(Y) else np.sqrt(((Y**2).sum(0) - n*(muY**2)) / (n-1)))
  cormat = covmat / sdvecX[:,np.newaxis].dot(sdvecY[:,np.newaxis].T)
  
  return cormat

# For each column of the input matrix X, convert the values into the ranks (for calculation of Spearman correlation, for example)
def rankMatrix(X):
  if sparse.issparse(X):
    idx_row, idx_col, dat = sparse.find(X)
    df = pd.DataFrame({'i' : idx_row, 'j' : idx_col, 'x' : dat}).sort_values('j')
    df['r'] = np.concatenate([ rankdata(x) + (df.shape[0]-len(x)) - (1+(df.shape[0]-len(x)))/2 for x in np.split(df.x, np.unique(df.j, return_index=True)[1][1:]) ])
    ranked = sparse.csr_matrix((df['r'], (df['i'], df['j'])), shape = X.shape)
  else:
    ranked = rankdata(X, method = "average", axis=0)
  
  return ranked

# Similar to rankMatrix but to use "dense" method for ranking non-zero values
def rankMatrix_dense(X):
  if sparse.issparse(X):
    idx_row, idx_col, dat = sparse.find(X)
    df = pd.DataFrame({'i' : idx_row, 'j' : idx_col, 'x' : dat}).sort_values('j')
    df['r'] = np.concatenate([ rankdata(x, method = "dense") for x in np.split(df['x'], np.unique(df['j'], return_index=True)[1][1:]) ])
    ranked = sparse.csr_matrix((df['r'], (df['i'], df['j'])), shape = X.shape)
  else:
    ranked = rankdata(X, method = "average", axis=0)
  
  return ranked

# Similar to rankMatrix, but only ranking non-zero values with zero values remaining zero
def rankMatrix_nonzero(X):
  if sparse.issparse(X):
    idx_row, idx_col, dat = sparse.find(X)
    df = pd.DataFrame({'i' : idx_row, 'j' : idx_col, 'x' : dat}).sort_values('j')
    value_split = np.split(df.x, np.unique(df.j, return_index=True)[1][1:])
    df['r'] = np.concatenate([ rankdata(x) for x in value_split ])
    ranked = sparse.csr_matrix((df['r'], (df['i'], df['j'])), shape = X.shape)
  else:
    ranked = rankdata(X, method = "average", axis=0)
  
  return ranked


# Generate a subset from the complete single-cell data, based on either completely random selection, sketching, or hybrid of the two
def select_subset_cells(adata,
                        use_rep = None,
                        ratio = 0.1,
                        alpha = 0,
                        seed = 123):
    import geosketch as gs
    N = adata.shape[0]
    ratio_random = ratio * (1-alpha)
    num_sketch = int(N * ratio * alpha)
    base_filter = np.repeat(False, N)
    
    if num_sketch > 0:
        if use_rep is None and 'X_pca' in adata.obsm.keys():
            use_rep = 'X_pca'
        X_rep = adata.obsm[use_rep]
        idx_sketch = gs.gs(X_rep, num_sketch, replace=False)
        base_filter[idx_sketch] = True
    if ratio_random > 0:
        if seed is not None: np.random.seed(seed)
        base_rand = np.random.uniform(0, 1, N - np.sum(base_filter)) < ratio_random
        base_filter[np.where(np.logical_not(base_filter))[0][np.where(base_rand)[0]]] = True
    
    return base_filter


# AggreCell algorithm to obtain the pseudocell index, given the anndata as the input
def generate_aggrecell_idx(adata,
                           n_neighbors = 20,
                           use_rep = None,
                           n_pcs = 20,
                           ratio = 0.1,
                           alpha_sketch = 0,
                           initdist = 1,
                           maxdist = 2,
                           mergedist = 2,
                           minsize = 3,
                           weighted = False,
                           seed = 123,
                           constraint_on_grp = None,
                           thres_grp_connectivities = 0.5):
  ## re-calculate the knn network
  sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep = use_rep)
  
  ## extract the knn network
  adj_knn = adata.obsp['connectivities']
  adj_knn.setdiag(adj_knn.max())
  dist_knn = adata.obsp['distances']
  
  ## if required, constrain the knn network on cells in the same cluster or connected cluster
  if constraint_on_grp is not None and constraint_on_grp in adata.obs.keys():
    mat_grp_ident = sparse.csr_matrix((np.repeat(1, adata.shape[0]), (np.arange(adata.shape[0]), adata.obs[constraint_on_grp].cat.codes)), shape = [adata.shape[0], len(adata.obs[constraint_on_grp].cat.categories)])
    sc.tl.paga(adata, groups=constraint_on_grp)
    adj_knn_grp = ((adata.uns['paga']['connectivities'] >= thres_grp_connectivities) + sparse.diags(np.repeat(1, len(adata.obs[constraint_on_grp].cat.categories)))).astype('bool')
    constr = mat_grp_ident.dot(adj_knn_grp).dot(mat_grp_ident.T)
    adj_knn = adj_knn.multiply(constr)
  
  ## build knn graph
  tup_knn = adj_knn.nonzero()
  tup_knn = [(adata.obs_names[tup_knn[0][i]], adata.obs_names[tup_knn[1][i]], dist_knn[tup_knn[0][i], tup_knn[1][i]]+1e-5) for i in range(len(tup_knn[0]))]
  graph = igraph.Graph.TupleList(tup_knn, weights=weighted)
  avg_weight = dist_knn[dist_knn.nonzero()].mean() if graph.is_weighted() else 1
  
  ## initial pseudocell territories and randomly assign cells being close to multiple pseudocell capitals
  N = adata.shape[0]
  node_index = np.ones(N) * -1
  base_filter = select_subset_cells(adata, use_rep = use_rep, ratio = ratio, alpha = alpha_sketch, seed = seed)
  
  shortestDistance = pd.DataFrame(graph.distances(source=np.where(base_filter)[0]))
  initdist = initdist if graph.is_weighted() or initdist >= 1 else 1
  selected = shortestDistance <= initdist * avg_weight
  for i in range(sum(base_filter)):
    selected.iloc[:, np.where(base_filter)[0][i]] = False
    selected.iloc[i, np.where(base_filter)[0][i]] = True
  belongToMulti = selected.apply(lambda x: np.sum(x) > 1, axis=0)
  for i in np.where(belongToMulti)[0]:
    x = selected.iloc[:,i]
    if seed is not None: np.random.seed(seed + i)
    selected.iloc[:,i] = np.array(range(len(x))) == np.where(x)[0][np.random.rand(np.sum(x)).argmax()]
  
  ## merge the too small but neighboring pseudocells
  idx_too_small = np.where(selected.sum(axis=1) < minsize)[0]
  idx_too_small_raw = np.where(base_filter)[0][idx_too_small]
  is_neighbors_too_small = np.array(shortestDistance.iloc[idx_too_small,idx_too_small_raw] <= initdist * avg_weight)
  mergable = pd.DataFrame(np.nonzero(is_neighbors_too_small)).transpose()
  
  while mergable.shape[0] > 0:
    if seed is not None: np.random.seed(seed)
    idx = np.arange(mergable.shape[0])
    np.random.shuffle(idx)
    mergable = mergable.iloc[idx,:]
    
    idx_mergedto = idx_too_small[mergable.iloc[0,0]]
    idx_mergedto_raw = idx_too_small_raw[mergable.iloc[0,0]]
    idx_cleanup = idx_too_small[mergable.iloc[0,1]]
    idx_cleanup_raw = idx_too_small_raw[mergable.iloc[0,1]]
    
    selected.iloc[idx_mergedto,:] = np.logical_or(selected.iloc[idx_mergedto,:], selected.iloc[idx_cleanup,:])
    selected.iloc[idx_cleanup,:] = False
    idx_too_small = np.setdiff1d(idx_too_small, idx_cleanup)
    idx_too_small_raw = np.setdiff1d(idx_too_small_raw, idx_cleanup_raw)
    if selected.iloc[idx_mergedto,:].sum() >= minsize:
        idx_too_small = np.setdiff1d(idx_too_small, idx_mergedto)
        idx_too_small_raw = np.setdiff1d(idx_too_small_raw, idx_mergedto_raw)
    is_neighbors_too_small = np.array(shortestDistance.iloc[idx_too_small,idx_too_small_raw] <= initdist * avg_weight)
    mergable = pd.DataFrame(np.nonzero(is_neighbors_too_small)).transpose()
  
  ## eliminate too small pseudocells
  selected = selected.loc[selected.sum(axis=1) >= minsize, :]
  
  ## assign cells without pseudocell assignment but close to at least one
  if maxdist > initdist:
    belongToNowhereButNotFar = np.where(selected.mean(0) == 0 & shortestDistance.apply(lambda x: np.min(x[x != 0]) <= maxdist * avg_weight, axis=0))[0]
    candidates = [np.where(shortestDistance.iloc[:, i] == np.min(shortestDistance.iloc[:, i]))[0] for i in belongToNowhereButNotFar]
    for i in range(len(candidates)):
      x = candidates[i]
      if seed is not None: np.random.seed(seed * i + 11)
      candidates[i] = x[np.random.rand(len(x)).argmax()]
    newSel = pd.DataFrame([np.repeat([False, True, False], [i, 1, np.sum(base_filter) - i - 1]) for i in candidates]).transpose()
    newSel.columns = belongToNowhereButNotFar
    selected.update(newSel)
  
  ## generate output data frame
  node_index = list(selected.apply(lambda x: np.where(x)[0][0] if sum(x) > 0 else -1, axis=0))
  node_id = list(graph.get_vertex_dataframe().iloc[:, 0])
  aggrecell_idx = dict(zip(node_id, node_index))
  aggrecell_idx = pd.DataFrame({'cell': adata.obs_names, 'idx_aggrecell': [aggrecell_idx[x] for x in adata.obs_names]})
  aggrecell_idx.iloc[aggrecell_idx.iloc[:,1] == -1,1] = None
  
  return aggrecell_idx

## summarize single-cell data to pseudocell-level
def group_vec_to_ident_mat(group, norm = True):
  group = group.astype("category")
  i = np.where(group.notnull())[0]
  j = group.iloc[i].cat.codes
  mat_ident = sparse.csr_matrix((np.repeat(1, len(i)), (i, j)), shape=(len(group), len(np.unique(j))))
  if norm:
      num_cells_per_group = np.ravel(mat_ident.sum(axis=0))
      mat_ident = mat_ident @ sparse.diags(1 / num_cells_per_group)
  
  return mat_ident

def summarize_numeric_matrix(mat, group, use_mean = True): # mat.shape = cell*feature; len(group) = cell
  mat_ident = group_vec_to_ident_mat(group, norm = use_mean)
  mat_summ = mat_ident.transpose() @ mat
  return mat_summ

def summarize_categorical_vector(fac, group): # len(fac) = cell; len(group) = cell
  mat_ident = group_vec_to_ident_mat(group, norm=False)
  mat_fac = pd.get_dummies(fac)
  mat_fac_summ = mat_ident.transpose() @ mat_fac
  if fac.isna().sum() == len(fac):
    return np.repeat(np.nan, mat_fac_summ.shape[0])
  fac_summ = mat_fac.columns[mat_fac_summ.argmax(axis = 1)]
  return fac_summ

def summarize_dataframe(df, group): # df.shape = cell*meta; len(group) = cell
  mats = list()
  # numeric columns
  if (df.select_dtypes("number").shape[1] > 0):
    mat_num = pd.DataFrame(summarize_numeric_matrix(df.select_dtypes("number"), group))
    mat_num.columns = df.select_dtypes("number").columns
    mats.append(mat_num)
  # boolean columns
  if (df.select_dtypes("bool").shape[1] > 0):
    mat_bool_raw = df.select_dtypes("bool")
    mat_bool = pd.concat([pd.DataFrame(summarize_categorical_vector(mat_bool_raw[x], group) == True) for x in mat_bool_raw.columns], axis=1)
    mat_bool.columns = mat_bool_raw.columns
    mats.append(mat_bool)
  # categorical columns
  if (df.select_dtypes(exclude=[ "number", "bool"]).shape[1] > 0):
    mat_categorical_raw = df.select_dtypes(exclude=[ "number", "bool"])
    mat_categorical = pd.concat([pd.DataFrame(summarize_categorical_vector(mat_categorical_raw[x], group)) for x in mat_categorical_raw.columns], axis=1)
    mat_categorical.columns = mat_categorical_raw.columns
    mats.append(mat_categorical)
  
  df_summ = pd.concat(mats, axis = 1)
  df_summ = df_summ[df.columns]
  return df_summ

def generate_aggrecells(adata, aggrecell_idx, prefix = "aggrcell_", X_is_count = False, counts_layer = ['counts'], **kwargs):
  expr_aggrecell = summarize_numeric_matrix(adata.X, group=aggrecell_idx.iloc[:, 1], use_mean = not X_is_count)
  
  meta_aggrecell = summarize_dataframe(adata.obs, group=aggrecell_idx.iloc[:, 1])
  meta_aggrecell.index = [prefix + str(x) for x in meta_aggrecell.index]
  
  ad_aggrecells = ad.AnnData(expr_aggrecell)
  ad_aggrecells.obs_names = meta_aggrecell.index
  ad_aggrecells.var_names = adata.var_names.copy()
  ad_aggrecells.obs = meta_aggrecell
  ad_aggrecells.var = adata.var.copy()
  
  for layer in adata.layers.keys():
    ad_aggrecells.layers[layer] = summarize_numeric_matrix(adata.X, group=aggrecell_idx.iloc[:, 1], use_mean = layer in counts_layer)
  
  return ad_aggrecells


# function to find valley of a bimodal distribution.
# uses to determine cutoff for 10x data with emptydrop cell calling which usually ends up with a low peak of n_genes_by_counts.
def find_valley_between_bimodal(x, min_peak_size = 0.1, nstep = 100, bandwidth_ratio = 1/30, do_plot = False):
  x_d = np.linspace(x.min(), x.max(), nstep)
  kde = KernelDensity(kernel = 'gaussian', bandwidth = (x.max()-x.min()) * bandwidth_ratio)
  kde.fit(x[:,None])
  logprob = kde.score_samples(x_d[:, None])
  
  d = logprob[1:len(logprob)] - logprob[0:(len(logprob)-1)]
  idx_max = np.where(np.logical_and(d[0:(len(d)-1)]>0, d[1:len(d)]<0))[0]
  idx_max = idx_max[np.exp(logprob)[idx_max] >= np.exp(logprob)[idx_max].max() * min_peak_size]
  if len(idx_max) == 2:
    idx_min = np.where(np.logical_and(d[0:(len(d)-1)]<0, d[1:len(d)]>0))[0]
    idx_min = idx_min[np.logical_and(idx_min < idx_max.max(), idx_min > idx_max.min())]
  else:
    idx_min = -1
  
  max_den = np.array([x_d[idx_max], x_d[idx_max+1]]).mean(0)
  min_den = np.array([x_d[idx_min], x_d[idx_min+1]]).mean(0) if idx_min >= 0 else None
  
  if do_plot:
    import matplotlib.pyplot as plt
    plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
    plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
    plt.vlines(x = max_den, color = 'b', ymin=0, ymax=max(np.exp(logprob)))
    if min_den is not None:
      plt.vlines(x = min_den, color = 'r', ymin=0, ymax=max(np.exp(logprob)))
  
  return min_den

def find_maximums_in_distribution(x, min_peak_size = 0.1, nstep = 100, bandwidth_ratio = 1/30, do_plot = False, ax=None):
  x_d = np.linspace(x.min(), x.max(), nstep)
  kde = KernelDensity(kernel = 'gaussian', bandwidth = (x.max()-x.min()) * bandwidth_ratio)
  kde.fit(x[:,None])
  logprob = kde.score_samples(x_d[:, None])
  
  d = logprob[1:len(logprob)] - logprob[0:(len(logprob)-1)]
  idx_max = np.where(np.logical_and(d[0:(len(d)-1)]>0, d[1:len(d)]<0))[0]
  idx_max = idx_max[np.exp(logprob)[idx_max] >= np.exp(logprob)[idx_max].max() * min_peak_size]
  
  max_den = np.array([x_d[idx_max], x_d[idx_max+1]]).mean(0)
    
  if do_plot:
    if ax is None:
        import matplotlib.pyplot as plt
        plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
        plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
        plt.vlines(x = max_den, color = 'b', ymin=0, ymax=max(np.exp(logprob)))
    else:
        ax.fill_between(x_d, np.exp(logprob), alpha=0.5)
        ax.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
        ax.vlines(x = max_den, color = 'b', ymin=0, ymax=max(np.exp(logprob)))
  
  return max_den

# function to identify outliers based on normal distribution
def is_not_outliers(x, min_p = 0.025, max_p = 0.975):
  x_scale = scale(x)
  cut_min = norm.ppf(min_p)
  cut_max = norm.ppf(max_p)
  return np.logical_and(x_scale > cut_min, x_scale < cut_max)

# function to do QC on ngenes and percent.mt
def do_QC(ad, check_valley = True, col_valley = "n_genes_by_counts", col_ngene = "log1p_n_genes_by_counts", col_mt = "pct_counts_mt", outlier_p = [0.025, 0.975], **kwargs):
  if check_valley and ad.shape[0] > 500:
    valley_ngenes = find_valley_between_bimodal(np.array(ad.obs[col_valley]), **kwargs)
    if valley_ngenes is not None:
      ad = ad[ad.obs[col_valley] > valley_ngenes[0],:]
  
  to_keep = is_not_outliers(np.array(ad.obs[col_ngene]), min_p = outlier_p[0], max_p = outlier_p[1])
  if col_mt is not None:
    if ad.obs[col_mt].max() > 0:
      to_keep = np.logical_and(to_keep, is_not_outliers(np.log1p(np.array(ad.obs[col_mt])), min_p = 0, max_p = outlier_p[1]))
  
  return ad[to_keep,:]

# function to guess whether the matrix is log-transformed
def is_logged(mat,
               max_row_num = 10000,
               max_col_num = None):
  max_vals = mat.max(0)
  if scipy.sparse.issparse(mat): max_vals = np.array(max_vals.todense()).flatten()
  diff_from_int = max_vals - np.round(max_vals)
  return np.max(np.abs(diff_from_int)) > 1e-5

# function to do subsampling and return a matrix showing the subsampling results
def subsample(x, size=None, N=1, use_all_attempt=False, debug_log=False):
    x = np.array(x).flatten()
    if size is None or size > len(x):
        size=len(x)
    
    if size < len(x) and use_all_attempt:
        x_dis = x.copy()
        if size*N < len(x):
            if debug_log:
                print('more samples than the max number to cover. selecting the items to cover...')
            x_dis = x[np.sort(np.random.choice(range(len(x)), size*N, replace=False))]
        if debug_log:
            print('initializing the assignment...')
        df0 = pd.DataFrame({'i' : range(len(x_dis)),
                            'j' : np.floor(np.random.uniform(size=len(x_dis))*N),
                            'x' : True
                           })
        if debug_log:
            print('done. check whether modification is needed...')
        num_sel = df0.groupby('j').count()['x']
        num_iter = 0
        while (num_sel > size).sum() > 0:
            if debug_log and (num_iter % 10 == 0):
                print('start iterative no.'+str(num_iter+1)+'...')
            idx_sel_max = num_sel.argmax()
            idx_sel_min = num_sel.argmin()
            num2move = num_sel[idx_sel_max] - size
            idx2move = np.random.choice(np.where(df0['j']==idx_sel_max)[0], num2move, replace=False)
            df0.loc[idx2move,'j'] = idx_sel_min
            num_sel = df0.groupby('j').count()['x']
            num_iter = num_iter + 1
        if debug_log:
            print('done. ' + 'in total '+str(num_iter)+' iteration is done.' if num_iter>0 else 'no modification is needed.')
        res0 = sparse.csr_matrix((df0['x'], (df0['i'], df0['j'])), shape = (len(x_dis), N))
        res0 = pd.DataFrame(res0.todense(), index=x_dis)
        res = pd.concat([res0, pd.DataFrame(np.full((len(x)-len(x_dis), N), False), index=np.setdiff1d(x, x_dis))]).loc[x,:]
        if debug_log:
            print('max amount of samples have been distributed to at least one subsample. add more selections if needed.')
        if (num_sel < size).sum() > 0:
            idx_sel_need_more = np.where(num_sel < size)[0]
            for i in idx_sel_need_more:
                idx_false = np.where(~res.iloc[:,i])[0]
                num2sel = size - num_sel[i]
                idx_sel_add = np.random.choice(idx_false, num2sel, replace=False)
                res.iloc[idx_sel_add,i] = True
    else:
        sel = [ np.sort(np.random.choice(range(len(x)), size, replace=False)) for i in range(N) ]
        df = pd.DataFrame({'i' : np.concatenate(sel),
                           'j' : np.repeat(range(N), size),
                           'x' : True
                          })
        res = sparse.csr_matrix((df['x'], (df['i'], df['j'])), shape = (len(x), N))
        res = pd.DataFrame(res.todense(), index=x)
    
    return res

def split_embedding(adata, split_by, basis = 'X_umap', ncols=2, nrows=None, figsize=(3,3), return_fig = True, **kwargs):
    import scanpy as sc
    import numpy as np
    import matplotlib.pyplot as plt
    categories = adata.obs[split_by].cat.remove_unused_categories().cat.categories
    if nrows is None:
        nrows = int(np.ceil(len(categories) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    axs = axs.flatten()
    for i, cat in enumerate(categories):
        ax = axs[i]
        sc.pl.embedding(adata[adata.obs[split_by] == cat], basis=basis, ax=ax, show=False, title=cat, **kwargs)
        ax.set_xlim(np.min(np.array(adata.obsm[basis])[:,0]), np.max(np.array(adata.obsm[basis])[:,0]))
        ax.set_ylim(np.min(np.array(adata.obsm[basis])[:,1]), np.max(np.array(adata.obsm[basis])[:,1]))
    plt.tight_layout()
    if return_fig:
        return fig

