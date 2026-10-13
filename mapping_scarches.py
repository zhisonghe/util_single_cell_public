import warnings

warnings.filterwarnings("ignore")

import os
import sys
import scvi
import scarches
import anndata

def is_lognorm(mat,
               max_row_num = 10000,
               max_col_num = None):
    import numpy as np
    import scipy
    max_vals = mat.max(0)
    if scipy.sparse.issparse(mat): max_vals = np.array(max_vals.todense()).flatten()
    diff_from_int = max_vals - np.round(max_vals)
    return np.max(np.abs(diff_from_int)) > 1e-5

def load_vae(model_path,
             model_type,
             ref,
             **kwargs):
    if model_type not in ['SCVI','SCANVI','SCPOLI']:
        raise RuntimeError('The model_type value should be SCVI, SCANVI or SCPOLI')
    
    if model_type == 'SCVI':
        model = scvi.model.SCVI
    elif model_type == 'SCANVI':
        model = scvi.model.SCANVI
    elif model_type == 'SCPOLI':
        model = scarches.models.scPoli
    
    vae = model.load(model_path, ref, **kwargs)
    return vae

def prepare_query_anndata(adata,
                          ref,
                          vae,
                          layer = None,
                          col_batch = None,
                          clean_up_ct = True,
                          unknown_ct_mask = 'Unknown',
                          skip_validation=False
                         ):
    adata = adata.copy()
    if isinstance(vae, scvi.model._scvi.SCVI):
        model = scvi.model.SCVI
    elif isinstance(vae, scvi.model._scanvi.SCANVI):
        model = scvi.model.SCANVI
    elif isinstance(vae, scarches.models.scpoli.scPoli):
        model = scarches.models.scPoli
    
    # Validate the scale of the used layer in reference and query
    if not skip_validation:
        if (isinstance(vae, scvi.model._scanvi.SCANVI) or isinstance(vae, scvi.model._scvi.SCVI)):
            used_layer = vae.adata_manager._registry["setup_args"]["layer"]
            if used_layer is None:
                expected_log = is_lognorm(ref.X)
                if layer:
                    adata.X = adata.layers[layer].copy()
                q_log = is_lognorm(adata.X)
            else:
                expected_log = is_lognorm(ref.layers[used_layer])
                if layer:
                    adata.layers[used_layer] = adata.layers[layer].copy()
                q_log = is_lognorm(adata.layers[used_layer])
                    
            # make sure the query data share the same scale as ref
            if q_log != expected_log:
                raise RuntimeError("Validation failed: the related layer is found in the query data, but it doesn't have the same expression scale as in the reference data")
        else:
            expected_log = is_lognorm(ref.X)
            if layer:
                adata.X = adata.layers[layer].copy()
            q_log = is_lognorm(adata.X)
            if q_log != expected_log:
                raise RuntimeError("Validation failed: the related layer is found in the query data, but it doesn't have the same expression scale as in the reference data")
    
    # Make sure the batch label is consistent
    if col_batch:
        if isinstance(vae, scvi.model._scanvi.SCANVI) or isinstance(vae, scvi.model._scvi.SCVI):
            ref_batch_key = vae.adata_manager._registry["setup_args"]["batch_key"]
            adata.obs[ref_batch_key] = adata.obs[col_batch].copy()
        else:
            ref_batch_key = vae.condition_keys_
            if len(ref_batch_key) == 1:
                ref_batch_key = ref_batch_key[0]
                adata.obs[ref_batch_key] = adata.obs[col_batch].copy()
            elif isinstance(col_batch, str) or len(col_batch) != len(ref_batch_key):
                raise RuntimeError('The number of provided condition keys in the query data is different from the model: ' + str(ref_batch_key))
            else:
                for i in range(len(ref_batch_key)):
                    adata.obs[ref_batch_key[i]] = adata.obs[col_batch[i]].copy()
    
    # Make sure the cell type labels are available, if it is a SCPOLI model; clean up the cell type information when needed
    if isinstance(vae, scarches.models.scpoli.scPoli):
        ref_ct_key = vae.cell_type_keys_
        for ct in ref_ct_key:
            if (ct not in adata.obs.columns) or clean_up_ct:
                adata.obs[ct] = unknown_ct_mask
    
    # Make query data feature space consistent with the reference
    if isinstance(vae, scvi.model._scanvi.SCANVI) or isinstance(vae, scvi.model._scvi.SCVI):
        model.prepare_query_anndata(adata, vae)
    elif isinstance(vae, scarches.models.scpoli.scPoli):
        import numpy as np
        import pandas as pd
        from scipy.sparse import csr_matrix
        from scipy.sparse import hstack
        
        var_names = vae.adata.var_names
        missing_features = np.setdiff1d(var_names, adata.var_names)
        extra_X = csr_matrix((adata.shape[0], len(missing_features)), dtype=adata.X.dtype)
        new_X = csr_matrix(hstack((adata.X, extra_X)))
        new_var = pd.DataFrame({'symbol' : list(adata.var_names)+list(missing_features)}).set_axis(list(adata.var_names)+list(missing_features))
        adata = anndata.AnnData(X = new_X, obs = adata.obs.copy(), var = new_var, obsm = adata.obsm.copy())
        adata = adata[:,var_names]
    
    return adata

def train_scarches(adata,
                   vae,
                   unknown_ct_names = ['Unknown'],
                   labeled_indices = [],
                   batch_size = 1024,
                   epochs = 500,
                   pretraining_epochs_ratio=0.8,
                   eta=10,
                   save_model_path = None):    
    if isinstance(vae, scvi.model._scanvi.SCANVI):
        model = scvi.model.SCANVI
    elif isinstance(vae, scvi.model._scvi.SCVI):
        model = scvi.model.SCVI
    elif isinstance(vae, scarches.models.scpoli.scPoli):
        model = scarches.models.scPoli
    else:
        raise RuntimeError('This VAE model is not yet supported')
    
    if isinstance(vae, scvi.model._scanvi.SCANVI) or isinstance(vae, scvi.model._scvi.SCVI):
        vae_q = model.load_query_data(adata, vae)
    else:
        vae_q = model.load_query_data(
            adata=adata,
            reference_model=vae,
            unknown_ct_names=unknown_ct_names,
            labeled_indices=labeled_indices,
        )
    if (isinstance(vae, scvi.model._scanvi.SCANVI) or isinstance(vae, scvi.model._scvi.SCVI)):
        vae_q.train(
            batch_size=batch_size,
            max_epochs=epochs,
            plan_kwargs=dict(weight_decay=0.0),
        )
        if save_model_path is not None:
            import os
            os.makedirs(save_model_path, exist_ok=True)
            vae_q.save(
                save_model_path,
                overwrite=True,
                save_anndata=False,
            )
    else:
        vae_q.train(
            n_epochs=epochs,
            pretraining_epochs=int(epochs * pretraining_epochs_ratio),
            eta=eta,
            batch_size = batch_size
        )
        if save_model_path is not None:
            vae_q.save(save_model_path, overwrite=True)
            vae_q.get_conditional_embeddings().write_h5ad(os.path.join(save_model_path, "sample_embedding.h5ad"))
    
    return vae_q

def get_latent_space(adata,
                     vae,
                     ref_adata = None,
                     col_batch = None
                    ):
    obs_q = adata.obs.copy()
    if ref_adata:
        obs_ref = ref_adata.obs.copy()
    
    if col_batch:
        if isinstance(vae, scvi.model._scanvi.SCANVI) or isinstance(vae, scvi.model._scvi.SCVI):
            ref_batch_key = vae.adata_manager._registry["setup_args"]["batch_key"]
            adata.obs[ref_batch_key] = adata.obs[col_batch].copy()
        else:
            ref_batch_key = vae.condition_keys_
            if len(ref_batch_key) == 1:
                ref_batch_key = ref_batch_key[0]
                adata.obs[ref_batch_key] = adata.obs[col_batch].copy()
            elif isinstance(col_batch, str) or len(col_batch) != len(ref_batch_key):
                raise RuntimeError('The number of provided condition keys in the query data is different from the model: ' + str(ref_batch_key))
            else:
                for i in range(len(ref_batch_key)):
                    adata.obs[ref_batch_key[i]] = adata.obs[col_batch[i]].copy()
        
    if ref_adata is None:
        adata_full = adata
    else:
        adata.obs['ref'] = 'query'
        ref_adata.obs['ref'] = 'ref'
        adata_full = adata.concatenate(ref_adata)
        adata_full.obs['batch'] = adata.obs["batch"].tolist() + ref_adata.obs["batch"].tolist()

    if isinstance(vae, scvi.model._scanvi.SCANVI) or isinstance(vae, scvi.model._scvi.SCVI):
        lat_rep = vae.get_latent_representation(adata_full)
    else:
        lat_rep = vae.get_latent(adata_full, mean=True)
    
    adata.obs = obs_q
    if ref_adata:
        ref_adata.obs = obs_ref
    
    if ref_adata is None:
        return lat_rep
    else:
        lat_rep_ref = lat_rep[adata_full.obs['ref'] == 'ref',:]
        lat_rep_q = lat_rep[adata_full.obs['ref'] == 'query',:]
        return {'ref' : lat_rep_ref, 'query' : lat_rep_q}

