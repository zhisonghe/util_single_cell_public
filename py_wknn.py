import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import torch
from pynndescent import NNDescent

from scipy import sparse
from typing import Optional, Union, Mapping, Literal
import warnings
import sys
import os
import importlib.util
import argparse

from tqdm import tqdm
tqdm.pandas()

warnings.filterwarnings("ignore")

def gaussian_kernel(d, sigma = None):
    if sigma is None:
        sigma = np.max(d) / 3
    gauss = np.exp(-0.5 * np.square(d) / np.square(sigma))
    return gauss

def nn2adj(nn,
           n1 = None,
           n2 = None,
           weight: Literal['unweighted','dist','gaussian_kernel'] = 'unweighted',
           sigma = None
          ):
    if n1 is None:
        n1 = nn[0].shape[0]
    if n2 is None:
        n2 = np.max(nn[0].flatten())
    
    df = pd.DataFrame({'i' : np.repeat(range(nn[0].shape[0]), nn[0].shape[1]),
                       'j' : nn[0].flatten(),
                       'x' : nn[1].flatten()})
    
    if weight == 'unweighted':
        adj = sparse.csr_matrix((np.repeat(1, df.shape[0]), (df['i'], df['j'])), shape=(n1, n2))
    else:
        if weight == 'gaussian_kernel':
            df['x'] = gaussian_kernel(df['x'], sigma)
        adj = sparse.csr_matrix((df['x'], (df['i'], df['j'])), shape=(n1, n2))
    
    return adj

def build_nn(ref,
             query = None,
             k = 100,
             weight: Literal['unweighted','dist','gaussian_kernel'] = 'unweighted',
             sigma = None,
             nogpu = False,
             verbose=True
            ):
    if query is None:
        query = ref
    
    if torch.cuda.is_available() and importlib.util.find_spec('cuml') and not nogpu:
        if verbose:
            print('GPU detected and cuml installed. Use cuML for neighborhood estimation...')
        from cuml.neighbors import NearestNeighbors
        model = NearestNeighbors(n_neighbors=k)
        model.fit(ref)
        knn = (model.kneighbors(query)[1], model.kneighbors(query)[0])
    else:
        if verbose:
            print('Falling back to neighborhood estimation using CPU with pynndescent')
        index = NNDescent(ref)
        knn = index.query(query, k=k)
    
    adj = nn2adj(knn, n1 = query.shape[0], n2 = ref.shape[0], weight = weight, sigma = sigma)
    return adj

def build_mutual_nn(dat1, dat2 = None, k1 = 100, k2 = None):
    if dat2 is None:
        dat2 = dat1
    if k2 is None:
        k2 = k1
    
    index_1 = NNDescent(dat1)
    index_2 = NNDescent(dat2)
    knn_21 = index_1.query(dat2, k=k1)
    knn_12 = index_2.query(dat1, k=k2)
    adj_21 = nn2adj(knn_21, n1 = dat2.shape[0], n2 = dat1.shape[0])
    adj_12 = nn2adj(knn_12, n1 = dat1.shape[0], n2 = dat2.shape[0])
    
    adj_mnn = adj_12.multiply(adj_21.T)
    return adj_mnn

def get_transition_prob_mat(dat, k = 50, symm = True):
    if torch.cuda.is_available() and importlib.util.find_spec('cuml'):
        print('Use cuML and GPU for kNN construction')
        from cuml.neighbors import NearestNeighbors
        model = NearestNeighbors(n_neighbors=k)
        model.fit(dat)
        knn = (model.kneighbors(dat)[1], model.kneighbors(dat)[0])
    else:
        print('Use NNDescent and CPU for kNN construction')
        index = NNDescent(dat)
        knn = index.query(dat, k = k)
    adj = nn2adj(knn, n1 = dat.shape[0], n2 = dat.shape[0])
    if symm:
        adj = ((adj + adj.T) > 0) + 0
    prob = sparse.diags(1 / np.array(adj.sum(1)).flatten()) @ adj.transpose()
    return prob

def random_walk_with_restart(init, transition_prob, alpha = 0.5, num_rounds = 100):
    init = np.array(init).flatten()
    heat = init[:,None]
    for i in range(num_rounds):
        heat = init[:,None] * alpha + (1 - alpha) * (transition_prob.transpose() @ heat)
    return heat

def get_wknn(ref,  # the ref representation to build ref-query neighbor graph
             query,  # the query representation to build ref-query neighbor graph
             ref2 = None,  # the ref representation to build ref-ref neighbor graph
             k: int = 100,  # number of neighbors per cell
             query2ref: bool = True,  # consider query-to-ref neighbors
             ref2query: bool = True,  # consider ref-to-query neighbors
             weighting_scheme: Literal['n','top_n','jaccard','jaccard_square','gaussian','dist'] = 'jaccard_square', # how to weight edges in the ref-query neighbor graph
             top_n: Optional[int] = None,
             sigma: Optional[float] = None,
             return_adjs: bool = False,
             nogpu: bool = False,
             verbose=True
            ):
    adj_q2r = build_nn(ref = ref, query = query, k = k, weight = 'dist' if weighting_scheme in ['gaussian', 'dist'] else 'unweighted', nogpu = nogpu, verbose=verbose)
    
    adj_r2q = None
    if ref2query:
        adj_r2q = build_nn(ref = query, query = ref, k = k, weight = 'dist' if weighting_scheme in ['gaussian', 'dist'] else 'unweighted', nogpu = nogpu, verbose=verbose)
    
    if query2ref and not ref2query:
        adj_knn = adj_q2r.T
    elif ref2query and not query2ref:
        adj_knn = adj_r2q
    elif ref2query and query2ref:
        adj_knn_shared = (adj_r2q > 0).multiply(adj_q2r.T > 0)
        adj_knn = adj_r2q + adj_q2r.T - adj_r2q.multiply(adj_knn_shared)
    else:
        warnings.warn('At least one of query2ref and ref2query should be True. Reset to default with both being True.')
        adj_knn_shared = (adj_r2q > 0).multiply(adj_q2r.T > 0)
        adj_knn = adj_r2q + adj_q2r.T - adj_r2q.multiply(adj_knn_shared)
    
    if weighting_scheme in ['n','top_n','jaccard','jaccard_square']:
        if ref2 is None:
            ref2 = ref
        adj_ref = build_nn(ref = ref2, k=k, nogpu = nogpu, verbose=verbose)
        num_shared_neighbors = adj_q2r @ adj_ref.T
        num_shared_neighbors_nn = num_shared_neighbors.multiply(adj_knn.T)

        wknn = num_shared_neighbors_nn.copy()
        if weighting_scheme == 'top_n':
            if top_n is None:
                top_n = k//4 if k > 4 else 1
            wknn = (wknn > top_n) * 1
        elif weighting_scheme == "jaccard":
            wknn.data = wknn.data / (k+k-wknn.data)
        elif weighting_scheme == "jaccard_square":
            wknn.data = (wknn.data / (k+k-wknn.data)) ** 2
    else:
        wknn = adj_knn.T
        if weighting_scheme == 'gaussian':
            wknn.data = gaussian_kernel(wknn.data, sigma = sigma)
    
    if return_adjs:
        adjs = {'q2r' : adj_q2r,
                'r2q' : adj_r2q,
                'knn' : adj_knn,
                'r2r' : adj_ref}
        return (wknn, adjs)
    else:
        return wknn

def estimate_presence_score(ref_adata,
                            query_adata,
                            wknn = None,
                            use_rep_ref_wknn = 'X_pca',
                            use_rep_query_wknn = 'X_pca',
                            k_wknn = 100,
                            query2ref_wknn = True,
                            ref2query_wknn = False,
                            weighting_scheme_wknn = 'jaccard_square',
                            ref_trans_prop = None,
                            use_rep_ref_trans_prop = None,
                            k_ref_trans_prop = 50,
                            symm_ref_trans_prop = True,
                            split_by = None,
                            do_random_walk = True,
                            alpha_random_walk = 0.1,
                            num_rounds_random_walk = 100,
                            log = True,
                            verbose = True
                           ):
    if wknn is None:
        if verbose:
            print('[PROGRESS] The wknn is not provided')
            print('[PROGRESS] Calculating wknn between ref and query...')
        ref = ref_adata.obsm[use_rep_ref_wknn]
        query = query_adata.obsm[use_rep_query_wknn]
        wknn = get_wknn(ref = ref,
                        query = query,
                        k = k_wknn,
                        query2ref = query2ref_wknn,
                        ref2query = ref2query_wknn,
                        weighting_scheme = weighting_scheme_wknn)
    
    if ref_trans_prop is None and do_random_walk:
        if verbose:
            print('[PROGRESS] Ref-ref transition matrix is not provided while random-walk is requested')
            print('[PROGRESS] Calculating ref-ref transition matrix...')
        if use_rep_ref_trans_prop is None: use_rep_ref_trans_prop = use_rep_ref_wknn
        ref = ref_adata.obsm[use_rep_ref_trans_prop]
        ref_trans_prop = get_transition_prob_mat(ref, k=k_ref_trans_prop)
    
    if verbose:
        print('[PROGRESS] Calculating presence scores...')
    if split_by and split_by in query_adata.obs.columns:
        presence_split = [ np.array(wknn[np.array(query_adata.obs[split_by] == x),:].sum(axis = 0)).flatten() for x in query_adata.obs[split_by].unique() ]
    else:
        presence_split = [ np.array(wknn.sum(axis = 0)).flatten() ]
    
    if do_random_walk:
        if verbose:
            print('[PROGRESS] Smoothing presence scores by random-walk...')
        presence_split_sm = [ random_walk_with_restart(init = x, transition_prob = ref_trans_prop, alpha = alpha_random_walk, num_rounds = num_rounds_random_walk) for x in presence_split ]
    else:
        presence_split_sm = [ x[:,None] for x in presence_split ]
    
    columns = query_adata.obs[split_by].unique() if split_by and split_by in query_adata.obs.columns else ['query']
    if len(columns) > 1:
        df_presence = pd.DataFrame(np.concatenate(presence_split_sm, axis=1), columns=columns, index=ref_adata.obs_names)
    else:
        df_presence = pd.DataFrame({columns[0] : presence_split_sm[0].flatten()}).set_index(ref_adata.obs_names)
    
    if log:
        df_presence = df_presence.apply(lambda x: np.log1p(x), axis=0)
    df_presence_norm = df_presence.apply(lambda x: np.clip(x, np.percentile(x,1), np.percentile(x,99)), axis=0).apply(lambda x: (x-np.min(x))/(np.max(x)-np.min(x)), axis=0)
    max_presence = df_presence_norm.max(1)
    
    return {'max' : max_presence, 'per_group' : df_presence_norm, 'wknn' : wknn, 'ref_trans_prop' : ref_trans_prop}

def get_matched_transcriptome(adata,
                              adata_ref,
                              wknn, # output of wknn.get_wknn, based on the scPoli[HNOCA]-scANVI[Braun] concatenated representation
                              use_raw = None,
                              rescale_factor_ref = 1): # HNOCA uses targetsum=1m, disease data uses targetsum=10k, so rescale_factor_ref should be 10k/1m = 1/100
    if (use_raw is None and adata_ref.raw is not None) or use_raw:
        expr_ref = adata_ref.raw.X.copy()
        var_ref = adata_ref.raw.var.copy()
    else:
        expr_ref = adata_ref.X.copy()
        var_ref = adata_ref.var.copy()
    
    if rescale_factor_ref != 1:
        expr_ref = ((expr_ref.expm1()) * rescale_factor_ref).log1p()
    normmat = sparse.diags(1/np.array(wknn.sum(axis=1)).flatten())
    softmax_wknn = normmat.dot(wknn)
    expr_bg = softmax_wknn @ expr_ref
    adata_bg = ad.AnnData(expr_bg, obs = adata.obs.copy(), var=var_ref, obsm = adata.obsm.copy())
    return adata_bg

def calculate_pairwise_correlation(adata1,
                                   adata2,
                                   features = None,
                                   use_raw = True,
                                   nthreads=1,
                                   verbose=True
                                  ):
    import numpy as np
    if use_raw:
        use_raw_1 = adata1.raw is not None
        varnames1 = adata1.raw.var_names if use_raw_1 else adata1.var_names
        use_raw_2 = adata2.raw is not None
        varnames2 = adata2.raw.var_names if use_raw_2 else adata2.var_names
    else:
        use_raw_1 = False
        use_raw_2 = False
        varnames1 = adata1.var_names
        varnames2 = adata2.var_names
    
    shared_features = np.intersect1d(varnames1, varnames2)
    if features is not None:
        shared_features = np.intersect1d(shared_features, features)
    
    if verbose:
        print('Extract expression matrices for comparison...')
    if use_raw_1:
        mat1 = adata1.raw[:,shared_features].X.T
    else:
        mat1 = adata1[:,shared_features].X.T
    if use_raw_2:
        mat2 = adata2.raw[:,shared_features].X.T
    else:
        mat2 = adata2[:,shared_features].X.T
    
    import joblib
    import contextlib
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from scipy.stats import pearsonr
    
    @contextlib.contextmanager
    def tqdm_joblib(tqdm_object):
        """Context manager to patch joblib to report into tqdm progress bar given as argument"""
        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)
    
        old_batch_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield tqdm_object
        finally:
            joblib.parallel.BatchCompletionCallBack = old_batch_callback
            tqdm_object.close()
    
    def calculate_correlation(col_pair):
        col1, col2 = col_pair
        return pearsonr(col1.toarray().flatten(), col2.toarray().flatten())[0]
    
    if verbose:
        column_pairs = [(mat1.getcol(i), mat2.getcol(i)) for i in tqdm(range(mat1.shape[1]), position=0, leave=True, desc='Group cell pair expression profiles')]
    else:
        column_pairs = [(mat1.getcol(i), mat2.getcol(i)) for i in range(mat1.shape[1])]
    
    if verbose:
        if nthreads == 1:
            correlations = [ calculate_correlation(pair) for pair in tqdm(column_pairs, position=0, leave=True, desc='Correlation calculation') ]
        else:
            with tqdm_joblib(tqdm(desc="Correlation calculation", total=len(column_pairs))) as progress_bar:
                correlations = Parallel(n_jobs=nthreads)(delayed(calculate_correlation)(pair) for pair in column_pairs)
    else:
        if nthreads==1:
            correlations = [ calculate_correlation(pair) for pair in column_pairs ]
        else:
            correlations = Parallel(n_jobs=nthreads)(delayed(calculate_correlation)(pair) for pair in column_pairs)
    
    return correlations


def transfer_labels(ref_adata,
                    query_adata,
                    wknn,
                    label_key="celltype"
                   ):
    scores = pd.DataFrame(
        wknn @ pd.get_dummies(ref_adata.obs[label_key]),
        columns=pd.get_dummies(ref_adata.obs[label_key]).columns,
        index=query_adata.obs_names,
    )
    scores["best_label"] = scores.idxmax(1)
    scores["best_score"] = scores.max(1)
    return scores


def hierarchical_region_lab_transfer(region_incl,
                                     region_hier,
                                     df_labs_region,
                                     root_region,
                                     do_random_walk=False,
                                     trans_prob_bo=None,
                                     use_rep_query='X_pca',
                                     k=20,
                                     alpha_rwwr=0.8,
                                     verbose=True,
                                    ):
    def find_commonnestn_recursive(x, hier, start_at, thres_over = 0.5, nmax=2):
        idx_max = x[hier[start_at]].idxmax()
        if idx_max in hier.keys():
            return find_commonnestn_recursive(x, hier, idx_max, thres_over = thres_over, nmax=nmax)
        largestn = x[hier[start_at]].nlargest(nmax)
        if nmax == 1:
            return largestn.index[0]
        else:
            if largestn.max() - largestn.min() > largestn.min() * thres_over:
                largestn = largestn[largestn > largestn.min() + largestn.min() * thres_over]
            return list(largestn.index)
    
    if do_random_walk:
        if trans_prob_bo is None:
            if verbose:
                print('[PROGRESS] Random-walk-based smoothening is requested without providing transition matrix of the query data. Calculate the transition matrix...')
            trans_prob_bo = get_transition_prob_mat(adata_query.obsm[use_rep_query], k=k)
        if verbose:
            print('[PROGRESS] Perform random walk with restart to smooth the scores...')
        scores_region_sm = df_labs_region.iloc[:,:(df_labs_region.shape[1]-2)].progress_apply(lambda x: random_walk_with_restart(x, trans_prob_bo, alpha = alpha_rwwr).flatten(), axis=0)
        scores_region = scores_region_sm
    else:
        scores_region = df_labs_region.iloc[:,:(df_labs_region.shape[1]-2)]
    freq_hier_regions = pd.concat([ pd.DataFrame(scores_region.loc[:,region_incl[x]].sum(1), columns=[x]) for x in region_incl.keys()], axis=1)
    
    if verbose:
        print('[PROGRESS] Perform hierarchical label assignment...')
    comm_region = freq_hier_regions.progress_apply(lambda x: find_commonnestn_recursive(x, region_hier, root_region, nmax=1), axis=1)
    if verbose:
        print('[DONE]')
    
    return comm_region


def hierarchical_region_lab_transfer_braun(adata_ref,
                                           adata_query,
                                           wknn,
                                           do_random_walk=False,
                                           trans_prob_bo=None,
                                           use_rep_query='X_pca',
                                           k=20,
                                           alpha_rwwr=0.8,
                                           verbose=True):
    region_incl = {'Forebrain' : ['Forebrain','Telencephalon','Cortex','Subcortex','Striatum','Hippocampus','Diencephalon','Hypothalamus','Thalamus'],
                   'Telencephalon': ['Telencephalon','Cortex','Subcortex','Striatum'],
                   'Cortex': ['Cortex'],
                   'Subcortex': ['Subcortex','Striatum'],
                   'Diencephalon': ['Diencephalon','Thalamus','Hypothalamus'],
                   'Hypothalamus': ['Hypothalamus'],
                   'Thalamus': ['Thalamus'],
                   'Midbrain': ['Midbrain','Midbrain dorsal','Midbrain ventral'],
                   'Midbrain dorsal': ['Midbrain dorsal'],
                   'Midbrain ventral': ['Midbrain ventral'],
                   'Hindbrain': ['Hindbrain','Cerebellum','Pons','Medulla'],
                   'Cerebellum': ['Cerebellum'],
                   'Pons': ['Pons'],
                   'Medulla': ['Medulla']}
    region_hier = {'Brain' : ['Forebrain','Midbrain','Hindbrain'],
                   'Forebrain' : ['Telencephalon','Diencephalon'],
                   'Telencephalon' : ['Cortex','Subcortex'],
                   'Diencephalon' : ['Hypothalamus','Thalamus'],
                   'Midbrain' : ['Midbrain dorsal','Midbrain ventral'],
                   'Hindbrain' : ['Cerebellum','Pons','Medulla']}
    
    if verbose:
        print('[PROGRESS] Calculate scores for each Subregion label...')
    df_labs_region = transfer_labels(adata_ref, adata_query, wknn, label_key='Subregion')
    
    return hierarchical_region_lab_transfer(
        region_incl = region_incl,
        region_hier = region_hier,
        df_labs_region = df_labs_region,
        root_region = 'Brain',
        do_random_walk=do_random_walk,
        trans_prob_bo=trans_prob_bo,
        use_rep_query=use_rep_query,
        k=k,
        alpha_rwwr=alpha_rwwr,
        verbose=verbose,
    )


def cmd_interface():
    parser = argparse.ArgumentParser(description="Estimate weighted kNN between reference and query for query presence scores and label transfer")
    
    parser.add_argument(
        "-q","--h5ad",
        type=str,
        dest="H5AD",
        required=True,
        help='H5AD file of the query data',
    )
    parser.add_argument(
        "-r","--ref_h5ad",
        type=str,
        dest="REF_H5AD",
        required=True,
        help='H5AD file of the reference data'
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        dest="output",
        default="output",
        help='Path to the output folder (default: ./output)'
    )
    parser.add_argument(
        "--use_rep_ref",
        type=str,
        dest="use_rep_ref",
        default="X_pca",
        help="Dimension reduction of the reference data"
    )
    parser.add_argument(
        "--use_rep_query",
        type=str,
        dest="use_rep_query",
        default="X_pca",
        help="Dimension reduction of the query data"
    )
    parser.add_argument(
        "-k", "--k_wknn",
        type=int,
        dest="k_wknn",
        default=100,
        help="Number of NNs per query cell in reference"
    )
    parser.add_argument(
        "-n", "--k_ref",
        type=int,
        dest="k_ref",
        default=50,
        help="Number of NNs per reference cell in reference"
    )
    parser.add_argument(
        "--smooth_presence",
        action="store_true",
        dest="smooth_presence",
        help="Apply random-walk-based smoothing of the presence score"
    )
    parser.add_argument(
        "--split_by",
        type=str,
        dest="split_by",
        help="Metadata columns of query data to split"
    )
    parser.add_argument(
        "--col_transfer",
        type=str,
        dest="col_transfer",
        help="Metadata column of reference data to be transferred to query"
    )
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cmd_interface()
    
    adata_ref = sc.read(args.ref_h5ad)
    adata_query = sc.read(args.h5ad)
    
    presence = estimate_presence_score(adata_ref, adata_query, use_rep_ref_wknn = args.use_rep_ref, use_rep_query_wknn = args.use_rep_query, k_wknn = args.k_wknn, k_ref_trans_prop = args.k_ref, split_by = args.split_by, do_random_walk = args.smooth_presence)
    max_presence, group_presence = (presence['max'], presence['per_group'])
    wknn = presence['wknn']
    sparse.save_npz(os.path.join(args.output, 'wknn.npz'), wknn)
    
    df_presence = pd.concat([max_presence, group_presence], axis=1).set_axis(['max']+list(group_presence.columns), axis=1)
    df_presence.to_csv(os.path.join(args.output, 'presence_scores.tsv'), sep='\t')
    
    if args.col_transfer:
        print('[PROGRESS] Transferring labels of ' + args.col_transfer + '...')
        df_labs = transfer_labels(adata_ref, adata_query, wknn, label_key=args.col_transfer)
        df_labs.to_csv(os.path.join(args.output, 'label_transfer.tsv'), sep='\t')
    

