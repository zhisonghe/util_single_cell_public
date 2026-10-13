def run_scvi(adata,
             batch_key = 'batch',
             layer = None,
             n_latent = 20,
             n_layers = 2,
             n_hidden = 512,
             batch_size = 1024,
             max_epochs = 50,
             save_model_path = None,
            ):
    import scvi
    scvi.model.SCVI.setup_anndata(
        adata,
        batch_key=batch_key,
        layer=layer,
    )
    arches_params = dict(
        use_layer_norm="both",
        use_batch_norm="none",
        encode_covariates=True,
        dropout_rate=0.2,
    )
    vae = scvi.model.SCVI(
        adata,
        n_latent=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
        **arches_params,
    )
    vae.train(
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stopping=True,
    )
    if save_model_path is not None:
        import os
        os.makedirs(save_model_path, exist_ok=True)
        vae.save(
            save_model_path,
            overwrite=True,
            save_anndata=False,
        )
    return vae

def run_scanvi_from_scvi(adata,
                         vae_scvi,
                         labels_key,
                         unlabeled_category,
                         batch_size = 1024,
                         max_epochs = 50,
                         n_samples_per_label=500,
                         save_model_path = None):
    import scvi
    lvae = scvi.model.SCANVI.from_scvi_model(
        vae_scvi,
        adata=adata,
        labels_key=labels_key,
        unlabeled_category=unlabeled_category,
    )
    lvae.train(
        batch_size=batch_size,
        max_epochs=max_epochs,
        n_samples_per_label=n_samples_per_label,
        early_stopping=True,
    )
    if save_model_path is not None:
        import os
        os.makedirs(save_model_path, exist_ok=True)
        lvae.save(
            save_model_path,
            overwrite=True,
            save_anndata=False,
        )
    return lvae

def run_scpoli(adata,
               condition_keys,
               cell_type_keys,
               unknown_ct_names,
               hidden_layer_sizes=[512],
               latent_dim=20,
               embedding_dims=5,
               recon_loss='nb',
               n_epochs=50,
               pretraining_epochs=40,
               eta=10,
               batch_size=1024,
               save_model_path = None
              ):
    import scarches
    import os
    from scipy import sparse
    adata.X = sparse.csr_matrix(adata.X)
    scpoli_model = scarches.models.scpoli.scPoli(
        adata=adata,
        condition_keys=condition_keys,
        cell_type_keys=cell_type_keys,
        unknown_ct_names=unknown_ct_names,
        hidden_layer_sizes=hidden_layer_sizes,
        latent_dim=latent_dim,
        embedding_dims=embedding_dims,
        recon_loss=recon_loss,
    )
    early_stopping_kwargs = {
        "early_stopping_metric": "val_prototype_loss",
        "mode": "min",
        "threshold": 0,
        "patience": 20,
        "reduce_lr": True,
        "lr_patience": 13,
        "lr_factor": 0.1,
    }
    scpoli_model.train(
        n_epochs=n_epochs,
        pretraining_epochs=pretraining_epochs,
        early_stopping_kwargs=early_stopping_kwargs,
        eta=eta,
        alpha_epoch_anneal=20,
        batch_size=batch_size,
    )
    if save_model_path is not None:
        scpoli_model.save(save_model_path, overwrite=True)
        scpoli_model.get_conditional_embeddings().write_h5ad(os.path.join(save_model_path, "sample_embedding.h5ad"))
    return scpoli_model

def plot_scvi_scanvi_lost(vae):
    import matplotlib.pyplot as plt
    plt.plot(vae.history['reconstruction_loss_train']['reconstruction_loss_train'], label='train')
    if 'reconstruction_loss_validation' in vae.history.keys():
        plt.plot(vae.history['reconstruction_loss_validation']['reconstruction_loss_validation'], label='validation')
    plt.legend()

def plot_scpoli_lost(scpoli_model):
    import matplotlib.pyplot as plt
    plt.plot(scpoli_model.trainer.logs['epoch_loss'], label='epoch')
    plt.plot(scpoli_model.trainer.logs['val_loss'], label='val')
    plt.legend()

def make_artificial_sample_labels(
    adata,
    dataset_key,
    sample_key,
    new_sample_key,
    prop_fixed = 0.67,
    sample_pairs = None,
    weights_samples = None,
    selectable_samples = None,
    prioritized_dataset_pairs = None,
    avoid_pairs = None,
    seed=12345,
    return_lab=False,
    copy=False,
    verbose=True
):
    import pandas as pd
    import numpy as np
    import anndata
    from tqdm import tqdm
    if copy:
        adata = adata.copy()
    
    # split data by samples
    if verbose:
        print('Split data by samples...')
    adata_samples = dict(zip(adata.obs[sample_key].cat.categories,
                             [ adata[adata.obs[sample_key]==x,:] for x in adata.obs[sample_key].cat.categories ]))
    
    # define sample-specific seeds
    np.random.seed(seed)
    seeds = dict(zip(adata_samples.keys(),
                     np.random.randint(0, 10000, size=len(adata_samples.keys()))))
    
    # split data into fixed and unfixed portion
    if verbose:
        print('Split data into fixed and unfixed portion...')
    adata_samples_fixed = dict()
    adata_samples_unfixed = dict()
    for sample in tqdm(adata_samples.keys()):
        adata_sample = adata_samples[sample]
        num_fixed = np.round(adata_sample.shape[0] * prop_fixed)
        np.random.seed(seeds[sample])
        obs_fixed = np.random.choice(adata_sample.obs_names, int(num_fixed), replace=False)
        obs_unfixed = np.setdiff1d(adata_sample.obs_names, obs_fixed)
        adata_samples_fixed[sample] = adata_sample[obs_fixed,:]
        adata_samples_unfixed[sample] = adata_sample[obs_unfixed,:]
    
    if isinstance(sample_pairs, pd.DataFrame) and np.sum(sample_pairs.columns[:3] == np.array(['sample1','sample2','id']))==3:
        if verbose:
            print('Use the provided sample pairing...')
        sample_pairs = sample_pairs.copy()
    else:
        # make dataset-sample table
        if verbose:
            print('Do iterative cross-dataset sample pairing...')
        source_samples = adata.obs[[dataset_key,sample_key]].value_counts(sort=False).reset_index().iloc[:,:2].set_axis(['dataset','sample'], axis=1)
        source_samples = source_samples.set_axis(source_samples['sample'],axis=0)
        source_samples['weight'] = 1
        if weights_samples is not None:
            if not isinstance(weights_samples, pd.Series):
                weights_samples = pd.Series(weights_samples)
            weights_samples = weights_samples[np.intersect1d(weights_samples.index, source_samples.index)]
            source_samples.loc[weights_samples.index,'weight'] = weights_samples
        source_samples['selectable'] = True
        if selectable_samples is not None:
            if not isinstance(selectable_samples, pd.Series):
                selectable_samples = pd.Series(selectable_samples)
            selectable_samples = selectable_samples[np.intersect1d(selectable_samples.index, source_samples.index)]
            source_samples.loc[selectable_samples.index,'selectable'] = selectable_samples
        
        # do iterative sample pairing
        if prioritized_dataset_pairs is None:
            prioritized_dataset_pairs = []
        if avoid_pairs is None or len(avoid_pairs)==0:
            avoid_pairs = pd.DataFrame(columns=['dataset1','dataset2'])
        elif not isinstance(avoid_pairs, pd.DataFrame):
            nump = len(avoid_pairs)
            for i in range(nump):
                avoid_pairs.append((avoid_pairs[i][1], avoid_pairs[i][0]))
            avoid_pairs = pd.DataFrame(avoid_pairs).set_axis(['dataset1','dataset2'], axis=1)
        else:
            avoid_pairs
        
        prioritized_dataset_pairs = [ pair for pair in prioritized_dataset_pairs if (source_samples['selectable'][source_samples['dataset']==pair[0]].sum() > 0) and (source_samples['selectable'][source_samples['dataset']==pair[1]].sum() > 0) and (pair[1]!=pair[0]) ]

        np.random.seed(seed)
        seeds = list(np.random.randint(1,10000, int(np.ceil(source_samples['selectable'].sum()/2))))
        source_samples_current = source_samples.copy()
        sample_pairs = pd.DataFrame(columns=['sample1','sample2'])
        while((source_samples_current.groupby('dataset').sum()['selectable']>0).sum() > 1):
            seed = seeds.pop()
            # select two data sets to pair
            prioritized_dataset_pairs = [ pair for pair in prioritized_dataset_pairs if (source_samples['selectable'][source_samples['dataset']==pair[0]].sum() > 0) and (source_samples['selectable'][source_samples['dataset']==pair[1]].sum() > 0) and (pair[1]!=pair[0]) ]
            if len(prioritized_dataset_pairs) > 0:
                ds1, ds2 = prioritized_dataset_pairs.pop(0)
            else:
                num_sel_per_ds = source_samples_current.groupby('dataset').sum()['selectable'].sort_values(ascending=False)
                ds2 = None
                while ds2 is None and len(num_sel_per_ds)>0:
                    ds1 = num_sel_per_ds.index[0]
                    others = num_sel_per_ds[~num_sel_per_ds.index.isin([ds1])]
                    if np.sum(avoid_pairs['dataset1']==ds1)>0:
                        others = others[~np.isin(others.index, avoid_pairs['dataset2'][avoid_pairs['dataset1']==ds1])]
                    others = np.repeat(others.index, others)
                    if len(others) > 0:
                        np.random.seed(seed)
                        ds2 = np.random.choice(others, 1)[0]
                    else:
                        num_sel_per_ds = num_sel_per_ds[1:]
            
            if ds2 is None:
                break
            
            # select one sample per selected data set
            def select_sample(df, seed=None):
                weight = df.set_index('sample')['weight'].sort_values(ascending = False)
                if np.sum(weight==weight.max()) == 1:
                    return weight.index[0]
                else:
                    cand = weight.index[weight==weight.max()]
                    if seed is not None:
                        np.random.seed(seed)
                    return np.random.choice(cand, 1)[0]
            sample1 = select_sample(source_samples_current.loc[(source_samples_current['selectable'] & (source_samples_current['dataset']==ds1)),:], seed)
            sample2 = select_sample(source_samples_current.loc[(source_samples_current['selectable'] & (source_samples_current['dataset']==ds2)),:], seed)
            
            # remove samples from selectable list and update the paired sample list
            sample_pairs = pd.concat([sample_pairs, pd.DataFrame({'sample1' : [sample1], 'sample2' : [sample2]})], axis=0)
            source_samples_current.loc[source_samples_current['sample'].isin([sample1,sample2]),'selectable'] = False
            
            # assign ID to the pair
            sample_pairs['id'] = [ "".join(['Mixed/',x]) for x in sample_pairs.iloc[:,:2].agg('/'.join, axis=1) ]
    
    # concatenate unfixed cells anndata of the paired samples, and remove the concatenated cells from the unfixed list
    if verbose:
        print('Generate new sample labels...')
    adata_samples_mixed = dict()
    for mix in tqdm(sample_pairs['id']):
        sample1, sample2 = sample_pairs.loc[sample_pairs['id']==mix,['sample1','sample2']].values.flatten().tolist()
        adata_mixed = anndata.concat([ adata_samples_unfixed[sample1], adata_samples_unfixed[sample2] ])
        adata_samples_mixed[mix] = adata_mixed
        del adata_samples_unfixed[sample1]
        del adata_samples_unfixed[sample2]
    
    # add sample-with-mixture labels as the new batch label for integration
    for sample in tqdm(adata_samples_fixed.keys()):
        adata_sample = adata_samples_fixed[sample]
        adata_sample.obs['artificial_sample'] = sample
        adata_samples_fixed[sample] = adata_sample
    for sample in tqdm(adata_samples_mixed.keys()):
        adata_sample = adata_samples_mixed[sample]
        adata_sample.obs['artificial_sample'] = sample
        adata_samples_mixed[sample] = adata_sample
    if len(adata_samples_unfixed.keys()) > 0:
        for sample in tqdm(adata_samples_unfixed.keys()):
            adata_sample = adata_samples_unfixed[sample]
            adata_sample.obs['artificial_sample'] = sample
            adata_samples_unfixed[sample] = adata_sample

    # concatenate the artificial sample labels and assign back to the original anndata
    new_sample_labs = [ x.obs['artificial_sample'] for x in adata_samples_fixed.values() ] + [ x.obs['artificial_sample'] for x in adata_samples_mixed.values() ]
    if len(adata_samples_unfixed) > 0:
        new_sample_labs = new_sample_labs + [ x.obs['artificial_sample'] for x in adata_samples_unfixed.values() ]
    if return_lab:
        return pd.concat(new_sample_labs)
    adata.obs[new_sample_key] = pd.concat(new_sample_labs)[adata.obs_names]
    
    if copy:
        return adata

