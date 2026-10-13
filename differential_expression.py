from scipy.sparse import issparse
from scipy.stats import f
from scipy.stats import wilcoxon, mannwhitneyu, ttest_1samp
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests, multitest_alias
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Optional, Union, Mapping, Literal

def f_nonzero(e, covar=None):
    if issparse(e):
        e = e.todense()
    e = np.array(e).flatten()
    
    if covar is None:
        m1 = ols('e ~ 1', data=pd.DataFrame({'e': e})).fit()
        a1 = anova_lm(m1)
        fval = (np.sum(e**2)/len(e)) / a1.loc['Residual','mean_sq']
        p = 1-np.abs(0.5 - f.cdf(fval, len(e), a1.loc['Residual','df']))*2
        coef = m1.params['Intercept']
    else:
        data = pd.DataFrame({'e': e, **covar})
        m0 = ols('e ~ '+" + ".join(covar.columns)+' -1', data=data).fit()
        a0 = anova_lm(m0)
        m1 = ols('e ~ '+" + ".join(covar.columns), data=data).fit()
        a1 = anova_lm(m1)
        sumsq0 = a0.loc['Residual','sum_sq'] + (a0.iloc[:-1]['sum_sq'].sum() - a1.iloc[:-1]['sum_sq'].sum())
        meansq0 = sumsq0 / (a0.loc['Residual','df']+1)
        meansq1 = a1.loc['Residual','mean_sq']
        fval = meansq0 / meansq1
        p = 1-np.abs(0.5 - f.cdf(fval, a0.loc['Residual','df']+1, a1.loc['Residual','df']))*2
        coef = m1.params['Intercept']
    
    return (fval, coef, p)

def f_nonzero_test(
    expr,
    covar=None,
    num_threads=1,
    var_names=None,
    padj_method: Literal[tuple(multitest_alias.keys())] = 'fdr_bh'
    ):
    if covar is not None:
        covar = pd.DataFrame(covar)
    
    num_cores = min(num_threads, multiprocessing.cpu_count())
    if num_cores > 1:
        results = Parallel(n_jobs=num_cores)(
            delayed(f_nonzero)(expr[:,i], covar) for i in range(expr.shape[1])
        )
    else:
        results = list()
        for i in tqdm(range(expr.shape[1])):
            results.append(f_nonzero(expr[:,i], covar=covar))
    
    df_res = pd.DataFrame(results, columns=["f", "coef", "pval"])
    isnull = df_res['pval'].isnull()
    if isnull.sum() > 0:
        df_res['padj'] = np.nan
        df_res.loc[~isnull,'padj'] = multipletests(df_res['pval'][~isnull], method=padj_method)[1]
    else:
        df_res['padj'] = multipletests(df_res['pval'], method=padj_method)[1]
    if var_names is not None and len(var_names) == df_res.shape[0]:
        df_res.index = var_names
    else:
        df_res.index = pd.RangeIndex(start=0, stop=expr.shape[1], step=1)
    return df_res

def scipy_test(
    expr,
    num_threads=1,
    var_names=None,
    method: Literal['ttest_1samp','wilcoxon','mannwhitneyu'] = 'ttest_1samp',
    padj_method: Literal[tuple(multitest_alias.keys())] = 'fdr_bh'
    ):
    def getcol(x,i):
        if issparse(x):
            return np.array(x[:,i].todense()).flatten()
        else:
            return np.array(x[:,i]).flatten()
    
    def runtest(x):
        if method == 'ttest_1samp':
            return ttest_1samp(x, popmean=0)
        elif method == 'wilcoxon':
            return wilcoxon(x)
        elif method == 'mannwhitneyu':
            return mannwhitneyu(x, [0])
    
    num_cores = min(num_threads, multiprocessing.cpu_count())
    if num_cores > 1:
        results = Parallel(n_jobs=num_cores)(
            delayed(runtest)(getcol(expr,i)) for i in range(expr.shape[1])
        )
    else:
        results = list()
        for i in tqdm(range(expr.shape[1])):
            results.append(runtest(getcol(expr,i)))
    
    df_res = pd.DataFrame([ tuple(x)[:2] for x in results ], columns=["stat", "pval"])
    avg = expr.mean(0)
    if issparse(avg):
        avg = avg.todense()
    avg = np.array(avg).flatten()
    df_res['avg'] = avg
    isnull = df_res['pval'].isnull()
    if isnull.sum() > 0:
        df_res['padj'] = np.nan
        df_res.loc[~isnull,'padj'] = multipletests(df_res['pval'][~isnull], method=padj_method)[1]
    else:
        df_res['padj'] = multipletests(df_res['pval'], method=padj_method)[1]
    df_res = df_res[['stat','avg','pval','padj']]
    if var_names is not None and len(var_names) == df_res.shape[0]:
        df_res.index = var_names
    else:
        df_res.index = pd.RangeIndex(start=0, stop=expr.shape[1], step=1)
    return df_res


def anova(e, group, covar=None, return_coef_group=None):
    if issparse(e):
        e = e.todense()
    e = np.array(e).flatten()
    
    if covar is None:
        m0 = ols('e ~ 1', data=pd.DataFrame({'e': e})).fit()
        m1 = ols('e ~ zz_group', data=pd.DataFrame({'e': e, 'zz_group': group})).fit()
    else:
        data = pd.DataFrame({'e': e, **covar, 'zz_group': group})
        m0 = ols('e ~ '+" + ".join(covar.columns), data=data).fit()
        m1 = ols('e ~ '+" + ".join(covar.columns)+' + zz_group', data=data).fit()
    
    a0 = anova_lm(m0)
    a1 = anova_lm(m1)
    a01 = anova_lm(m0, m1)
    p1 = a01.loc[:, 'Pr(>F)'].iloc[-1]
    p2 = 1-np.abs(0.5 - f.cdf(a0.loc['Residual','mean_sq']/a1.loc['Residual','mean_sq'], a0.loc['Residual','df'], a1.loc['Residual','df']))*2
    
    var_tot = a0['sum_sq'].sum()
    var_covar = a0.iloc[:-1]['sum_sq'].sum()
    var_g = a1.loc['zz_group', 'sum_sq']
    
    if return_coef_group is not None and return_coef_group in pd.Series(group).unique() and f'zz_group[T.{return_coef_group}]' in m1.params.index:
        coef_g = m1.params[f'zz_group[T.{return_coef_group}]']
    else:
        coef_g = np.nan
    
    return (var_g, var_covar, var_tot, p1, p2, coef_g)

def ancova_group_test(
    expr,
    group,
    covar=None,
    num_threads=1,
    return_coef_group=None,
    var_names=None,
    padj_method: Literal[tuple(multitest_alias.keys())] = 'fdr_bh'
):
    if not isinstance(group, pd.Series):
        group = pd.Series(group)
    idx = np.where(~group.isnull())[0]
    expr = expr[idx,:]
    group = group.iloc[idx]
    if covar is not None:
        covar = pd.DataFrame(covar).iloc[idx,:]
    
    if group.dtype.name != 'category':
        group = group.astype('category')
    group = group.cat.remove_unused_categories()
    g_lev = np.array(pd.Series(group).astype('category').cat.categories)
    if return_coef_group is not None and return_coef_group in g_lev:
        g_lev = np.concatenate([ g_lev[g_lev != return_coef_group], [return_coef_group] ])
        group = group.cat.reorder_categories(g_lev)
    
    num_cores = min(num_threads, multiprocessing.cpu_count())
    if num_cores > 1:
        results = Parallel(n_jobs=num_cores)(
            delayed(anova)(expr[:,i], group, covar, return_coef_group) for i in range(expr.shape[1])
        )
    else:
        results = list()
        for i in tqdm(range(expr.shape[1])):
            results.append(anova(expr[:,i], group, covar=covar, return_coef_group=return_coef_group))
    
    df_res = pd.DataFrame(results, columns=["var_group", "var_covar", "var_total", "p_ANCOVA", "p_Resi", "coef"])
    if var_names is not None and len(var_names) == df_res.shape[0]:
        df_res.index = var_names
    else:
        df_res.index = pd.RangeIndex(start=0, stop=expr.shape[1], step=1)
    
    isnull = df_res['p_ANCOVA'].isnull()
    if isnull.sum() > 0:
        df_res['padj_ANCOVA'] = np.nan
        df_res.loc[~isnull,'padj_ANCOVA'] = multipletests(df_res['p_ANCOVA'][~isnull], method=padj_method)[1]
    else:
        df_res['padj_ANCOVA'] = multipletests(df_res['p_ANCOVA'], method=padj_method)[1]
    isnull = df_res['p_Resi'].isnull()
    if isnull.sum() > 0:
        df_res['padj_Resi'] = np.nan
        df_res.loc[~isnull,'padj_Resi'] = multipletests(df_res['p_Resi'][~isnull], method=padj_method)[1]
    else:
        df_res['padj_Resi'] = multipletests(df_res['p_Resi'], method=padj_method)[1]
    df_res = df_res[["var_group", "var_covar", "var_total", "coef", "p_ANCOVA", "padj_ANCOVA", "p_Resi", "padj_Resi"]]
    if var_names is not None and len(var_names) == df_res.shape[0]:
        df_res.index = var_names
    else:
        df_res.index = pd.RangeIndex(start=0, stop=expr.shape[1], step=1)
    return df_res

def ancova_group_test_parametric(
    adata,
    layer = None,
    group = None,
    covar = None,
    ref_group = None,
    interactions_with_group = None,
    return_coefs = False,
    padj_method: Literal[tuple(multitest_alias.keys())] = 'fdr_bh',
    verbose=False
):
    from formulaic import model_matrix
    from scipy import stats
    import re
    import datetime
    def _log(msg):
        if verbose:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")
    def _build_design_formula(
            group_key_internal: str,
            factors_categorical: list,
            factors_continuous: list,
            interactions = None
            ) -> tuple:
        """Build a design formula string."""
        terms = [f"C({group_key_internal})"]
        if factors_categorical:
            terms += [f"C({f})" for f in factors_categorical]
        if factors_continuous:
            terms += factors_continuous
        if interactions is not None and len(np.intersect1d(factors_categorical, interactions)) > 0:
            for f in interactions:
                if f in factors_categorical:
                    terms += [f"C({group_key_internal}):C({f})"]
        if interactions is not None and len(np.intersect1d(factors_continuous, interactions)) > 0:
            for f in interactions:
                if f in factors_continuous:
                    terms += [f"C({group_key_internal}):{f}"]
        
        full_formula = " + ".join(terms)
        reduced_terms = [t for t in terms if not t.startswith(f"C({group_key_internal})")]
        reduced_formula = " + ".join(reduced_terms) if reduced_terms else "1"
        return full_formula, reduced_formula

    if group not in adata.obs.columns:
        raise TypeError('group should be a column in .obs.')
    idx = np.where(~adata.obs[group].isnull())[0]
    adata = adata[idx,:]
    _log(f"Input: {len(idx)} cells x {adata.shape[1]} features after dropping NaN in '{group}'.")

    if layer is None or layer not in adata.layers.keys():
        expr = adata.X
    else:
        expr = adata.layers[layer]
    
    cols = [group] + list(np.intersect1d(covar, adata.obs.columns)) if covar is not None else [group]
    meta = adata.obs[cols].copy()
    if meta[group].dtype.name != 'category':
        meta[group] = meta[group].astype('category')
    meta[group] = meta[group].cat.remove_unused_categories()
    g_lev = np.array(pd.Series(meta[group]).astype('category').cat.categories)
    if ref_group is not None and ref_group in g_lev:
        g_lev = np.concatenate([[ref_group], g_lev[g_lev != ref_group]])
        meta[group] = meta[group].cat.reorder_categories(g_lev)
    
    # Construct the design matrices for the full and reduced models
    Y = expr.astype(np.float32).toarray() if issparse(expr) else np.array(expr, dtype=np.float32)
    design_full, design_reduced = _build_design_formula(
        group_key_internal = group,
        factors_categorical = [ c for c in meta.columns if c != group and meta[c].dtype.name == 'category' and meta[c].isnull().sum()==0 ],
        factors_continuous = [ c for c in meta.columns if c != group and meta[c].dtype.name != 'category' and meta[c].isnull().sum()==0 ],
        interactions = interactions_with_group
    )
    mm_full = model_matrix(design_full, data=meta)
    mm_reduced = model_matrix(design_reduced, data=meta)
    X_full = mm_full.astype(np.float64).values
    X_reduced = mm_reduced.astype(np.float64).values
    _log(f"Design matrices built. Full formula: '{design_full}' ({X_full.shape[1]} params). Reduced formula: '{design_reduced}' ({X_reduced.shape[1]} params).")

    # Project Y onto the column spaces using QR decomposition
    # The mode='reduced' returns a non-square matrix Q with orthonormal columns,
    # which satisfies Q^tQ = Id (but QQ^t is not Id)
    Qf, Rf = np.linalg.qr(X_full, mode="reduced")
    Qr, _ = np.linalg.qr(X_reduced, mode="reduced")
    _log("QR decomposition done.")
    # Residuals
    res_full = Y - Qf @ (Qf.T @ Y)
    res_reduced = Y - Qr @ (Qr.T @ Y)
    rss_full = np.maximum(np.sum(res_full**2, axis=0), 0.0)
    rss_reduced = np.maximum(np.sum(res_reduced**2, axis=0), 0.0)
    _log("Residual sums of squares computed.")

    # Perform the F-test
    vars_tot = np.var(Y, axis=0, dtype=np.float64) * Y.shape[0]  # ddof=0, so multiply by N to get SS_total
    vars_full = np.maximum(vars_tot - rss_full, 0.0)   # full model explained variations
    vars_reduced = np.maximum(vars_tot - rss_reduced, 0.0)  # reduced model explained variations
    df_full = X_full.shape[1]
    df_rss_full = Y.shape[0] - df_full
    df_reduced = X_reduced.shape[1]
    df_rss_reduced = Y.shape[0] - df_reduced

    f1 = ((rss_reduced - rss_full) / (df_rss_reduced - df_rss_full)) / (rss_full / df_rss_full) # ANOVA
    f2 = (vars_full / df_full) / (vars_reduced / df_reduced) # Model F-test
    f3 = (rss_reduced / df_rss_reduced) / (rss_full / df_rss_full) # Residual F-test
    p1 = stats.f.sf(f1, df_rss_reduced - df_rss_full, df_rss_full)
    p2 = stats.f.sf(f2, df_full, df_reduced)
    p3 = 1 - abs(0.5 - stats.f.cdf(f3, df_rss_reduced, df_rss_full)) * 2
    _log("F-tests computed.")
    
    df_res = pd.DataFrame(
        {
            'var_group' : vars_full - vars_reduced,
             'var_covar' : vars_reduced,
             'var_total' : vars_tot,
             'p_anova' : p1,
             'p_model' : p2,
             'p_resi' : p3
        },
        index = adata.var_names
    )
    for col in ['p_anova', 'p_model', 'p_resi']:
        isnull = df_res[col].isnull()
        if isnull.sum() > 0:
            df_res[re.sub(r'p_', r'padj_', col)] = np.nan
            df_res.loc[~isnull,re.sub(r'p_', r'padj_', col)] = multipletests(df_res[col][~isnull], method=padj_method)[1]
        else:
            df_res[re.sub(r'p_', r'padj_', col)] = multipletests(df_res[col], method=padj_method)[1]
    _log("Multiple testing correction done.")
    
    if return_coefs:
        # beta_hat = R^{-1} Q^T Y  (shape: n_params x n_genes)
        # Use lstsq (pseudoinverse) instead of solve to handle rank-deficient design
        # matrices, e.g. when interaction terms are added but some covariate levels
        # only contain cells from a single group.
        coef_matrix = np.linalg.lstsq(Rf, Qf.T @ Y, rcond=None)[0]
        df_coefs = pd.DataFrame(
            np.array(coef_matrix).T,
            index=adata.var_names,
            columns=mm_full.columns
        )
        _log("Coefficient matrix extracted.")
        return df_res, df_coefs
    _log("Done.")
    return df_res


def dissect_variance_parametric(
    adata,
    layer = None,
    sources = None,
    covar = None,
    interactions = None,
    verbose = False
):
    """
    Dissect the variance of each feature into contributions from a set of
    independent `sources` and, optionally, a block of nuisance `covar`iates,
    using the same linear-model / QR-decomposition machinery as
    `ancova_group_test_parametric`, but generalized to an arbitrary number of
    sources and without any hypothesis testing or coefficient extraction.

    For each source (or interaction) term, the reported variance is the
    reduction in residual sum of squares obtained by adding that term to a
    model that already contains every other term (i.e. a type-II-like,
    marginal decomposition). Because sources can be correlated with one
    another, the reported components are not guaranteed to sum exactly to
    the total explained variance.

    Parameters
    ----------
    adata : AnnData
        Values to dissect (`adata.X` or `adata.layers[layer]`) as the
        dependent variable.
    layer : str, optional
        Layer of `adata` to use; defaults to `adata.X`.
    sources : list of str
        Columns in `adata.obs` to dissect the variance into. Columns with
        `category` dtype are treated as categorical, everything else as
        continuous.
    covar : list of str, optional
        Additional nuisance covariates to adjust for. Their combined
        contribution is reported as a single 'covar' component, not broken
        down further.
    interactions : list of tuple(str, str), optional
        Pairs of entries from `sources` for which an interaction term should
        also be added to the model and reported separately.
    verbose : bool
        Print progress messages.

    Returns
    -------
    pandas.DataFrame indexed by `adata.var_names`, with one 'var_<source>'
    column per source (and per interaction, named '<s1>:<s2>'), plus
    'var_covar' (if `covar` is given), 'var_residual' and 'var_total', and
    the corresponding 'prop_*' columns (variance divided by 'var_total').
    """
    from formulaic import model_matrix
    import datetime
    import warnings

    def _log(msg):
        if verbose:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")

    if sources is None or len(sources) == 0:
        raise TypeError('sources should be a non-empty list of .obs columns to dissect variance into.')
    sources = list(sources)
    covar = list(covar) if covar is not None else []
    interactions = list(interactions) if interactions is not None else []

    if len(set(sources)) != len(sources):
        raise TypeError(f"sources contains duplicate entries: {sources}")
    overlap = set(sources) & set(covar)
    if overlap:
        raise TypeError(f"Columns cannot appear in both `sources` and `covar`: {sorted(overlap)}")
    if 'covar' in sources:
        raise TypeError("'covar' is a reserved label and cannot be used as an entry of `sources`.")
    missing = [c for c in sources + covar if c not in adata.obs.columns]
    if len(missing) > 0:
        raise TypeError(f"The following columns are not found in .obs: {missing}")
    for s1, s2 in interactions:
        if s1 not in sources or s2 not in sources:
            raise TypeError(f"Interaction terms must be between two entries of `sources`, got ({s1}, {s2}).")
        if s1 == s2:
            raise TypeError(f"Interaction terms must be between two different entries of `sources`, got ('{s1}', '{s1}').")

    all_cols = sources + covar
    meta = adata.obs[all_cols].copy()
    idx = np.where(meta.notnull().all(axis=1))[0]
    adata = adata[idx, :]
    meta = meta.iloc[idx, :]
    _log(f"Input: {len(idx)} cells x {adata.shape[1]} features after dropping NaN in {all_cols}.")
    for c in all_cols:
        if meta[c].dtype.name == 'category':
            meta[c] = meta[c].cat.remove_unused_categories()

    if layer is None or layer not in adata.layers.keys():
        expr = adata.X
    else:
        expr = adata.layers[layer]
    Y = expr.astype(np.float32).toarray() if issparse(expr) else np.array(expr, dtype=np.float32)

    def _term(col):
        return f"C({col})" if meta[col].dtype.name == 'category' else col

    # each entry is (label, formula term); 'covar' terms share a single label so
    # their combined contribution is reported as one block
    term_list = [(s, _term(s)) for s in sources]
    for s1, s2 in interactions:
        term_list.append((f"{s1}:{s2}", f"{_term(s1)}:{_term(s2)}"))
    for c in covar:
        term_list.append(('covar', _term(c)))
    full_formula = " + ".join(t for _, t in term_list)
    labels = list(dict.fromkeys(lab for lab, _ in term_list))

    def _check_rank(formula):
        X = model_matrix(formula, data=meta).astype(np.float64).values
        rank = np.linalg.matrix_rank(X)
        if rank < X.shape[1]:
            warnings.warn(
                f"Design matrix for formula '{formula}' is rank-deficient "
                f"({rank} < {X.shape[1]} columns): some terms are collinear "
                "or confounded (e.g. two sources fully aliased with each other, "
                "or a covariate fully determined by a source), and the affected "
                "variance components may be unreliable."
            )

    def _fit_rss(formula):
        X = model_matrix(formula, data=meta).astype(np.float64).values
        Q, _ = np.linalg.qr(X, mode="reduced")
        res = Y - Q @ (Q.T @ Y)
        return np.maximum(np.sum(res**2, axis=0), 0.0)

    # a full-rank check on the full model suffices: any subset of a linearly
    # independent set of columns is itself linearly independent, so every
    # reduced model built by dropping terms is guaranteed full rank too.
    _check_rank(full_formula)
    rss_full = _fit_rss(full_formula)
    _log(f"Full model fitted. Formula: '{full_formula}'.")

    var_res = {}
    for lab in labels:
        # drop the term itself, and any interaction terms that involve it, to keep
        # the reduced model nested (marginality principle)
        drop = {t for l, t in term_list if l == lab or (':' in l and lab in l.split(':'))}
        reduced_terms = [t for l, t in term_list if t not in drop]
        reduced_formula = " + ".join(reduced_terms) if reduced_terms else "1"
        rss_reduced = _fit_rss(reduced_formula)
        var_res[lab] = np.maximum(rss_reduced - rss_full, 0.0)
        _log(f"Reduced model (without '{lab}') fitted.")

    vars_tot = np.var(Y, axis=0, dtype=np.float64) * Y.shape[0]  # ddof=0, so multiply by N to get SS_total
    df_res = pd.DataFrame({f"var_{lab}": var_res[lab] for lab in labels}, index=adata.var_names)
    df_res['var_residual'] = rss_full
    df_res['var_total'] = vars_tot
    denom = df_res['var_total'].replace(0.0, np.nan)  # avoid 0/0 and x/0 for zero-variance genes
    for lab in labels:
        df_res[f"prop_{lab}"] = df_res[f"var_{lab}"] / denom
    df_res['prop_residual'] = df_res['var_residual'] / denom
    _log("Done.")
    return df_res


def dissect_variance_sequential_parametric(
    adata,
    layer = None,
    sources = None,
    covar = None,
    interactions = None,
    verbose = False
):
    """
    Sequential (type-I-like) counterpart to `dissect_variance_parametric`.

    Terms are added to the model one at a time -- first the (combined)
    `covar` block if given, then each entry of `sources` in the order
    provided, then each requested interaction -- and the variance
    attributed to a term is the additional reduction in residual sum of
    squares obtained by adding it on top of all terms already in the model.

    Because each step's reduction is measured against the running model
    (not the full model), the components always sum exactly to the total
    explained variance, but the order in which terms are added now matters:
    variance shared between correlated sources is credited to whichever one
    enters the model first. See `dissect_variance_parametric` for the
    marginal (order-independent) alternative.

    Parameters are the same as `dissect_variance_parametric`; the order of
    `sources` (and of `interactions`) determines the order terms are added.
    """
    from formulaic import model_matrix
    import datetime
    import warnings

    def _log(msg):
        if verbose:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")

    if sources is None or len(sources) == 0:
        raise TypeError('sources should be a non-empty list of .obs columns to dissect variance into.')
    sources = list(sources)
    covar = list(covar) if covar is not None else []
    interactions = list(interactions) if interactions is not None else []

    if len(set(sources)) != len(sources):
        raise TypeError(f"sources contains duplicate entries: {sources}")
    overlap = set(sources) & set(covar)
    if overlap:
        raise TypeError(f"Columns cannot appear in both `sources` and `covar`: {sorted(overlap)}")
    if 'covar' in sources:
        raise TypeError("'covar' is a reserved label and cannot be used as an entry of `sources`.")
    missing = [c for c in sources + covar if c not in adata.obs.columns]
    if len(missing) > 0:
        raise TypeError(f"The following columns are not found in .obs: {missing}")
    for s1, s2 in interactions:
        if s1 not in sources or s2 not in sources:
            raise TypeError(f"Interaction terms must be between two entries of `sources`, got ({s1}, {s2}).")
        if s1 == s2:
            raise TypeError(f"Interaction terms must be between two different entries of `sources`, got ('{s1}', '{s1}').")

    all_cols = sources + covar
    meta = adata.obs[all_cols].copy()
    idx = np.where(meta.notnull().all(axis=1))[0]
    adata = adata[idx, :]
    meta = meta.iloc[idx, :]
    _log(f"Input: {len(idx)} cells x {adata.shape[1]} features after dropping NaN in {all_cols}.")
    for c in all_cols:
        if meta[c].dtype.name == 'category':
            meta[c] = meta[c].cat.remove_unused_categories()

    if layer is None or layer not in adata.layers.keys():
        expr = adata.X
    else:
        expr = adata.layers[layer]
    Y = expr.astype(np.float32).toarray() if issparse(expr) else np.array(expr, dtype=np.float32)

    def _term(col):
        return f"C({col})" if meta[col].dtype.name == 'category' else col

    # blocks are added in this fixed order: covar (combined), then sources, then interactions
    blocks = []
    if covar:
        blocks.append(('covar', [_term(c) for c in covar]))
    for s in sources:
        blocks.append((s, [_term(s)]))
    for s1, s2 in interactions:
        blocks.append((f"{s1}:{s2}", [f"{_term(s1)}:{_term(s2)}"]))

    def _check_rank(terms):
        formula = " + ".join(terms) if terms else "1"
        X = model_matrix(formula, data=meta).astype(np.float64).values
        rank = np.linalg.matrix_rank(X)
        if rank < X.shape[1]:
            warnings.warn(
                f"Design matrix for formula '{formula}' is rank-deficient "
                f"({rank} < {X.shape[1]} columns): some terms are collinear "
                "or confounded (e.g. two sources fully aliased with each other, "
                "or a covariate fully determined by a source), and the affected "
                "variance components may be unreliable."
            )

    def _fit_rss(terms):
        formula = " + ".join(terms) if terms else "1"
        X = model_matrix(formula, data=meta).astype(np.float64).values
        Q, _ = np.linalg.qr(X, mode="reduced")
        res = Y - Q @ (Q.T @ Y)
        return np.maximum(np.sum(res**2, axis=0), 0.0)

    vars_tot = np.var(Y, axis=0, dtype=np.float64) * Y.shape[0]  # ddof=0, so multiply by N to get SS_total
    rss_prev = _fit_rss([])
    _log("Intercept-only model fitted.")

    # a rank check on the model with every block's terms included suffices: any
    # subset of a linearly independent set of columns is itself linearly
    # independent, so every intermediate cumulative model is full rank too.
    _check_rank([t for _, terms in blocks for t in terms])

    cum_terms = []
    var_res = {}
    for lab, terms in blocks:
        cum_terms = cum_terms + terms
        rss_cur = _fit_rss(cum_terms)
        var_res[lab] = np.maximum(rss_prev - rss_cur, 0.0)
        rss_prev = rss_cur
        _log(f"Added '{lab}'. Cumulative formula: '{' + '.join(cum_terms)}'.")

    df_res = pd.DataFrame({f"var_{lab}": var_res[lab] for lab, _ in blocks}, index=adata.var_names)
    df_res['var_residual'] = rss_prev
    df_res['var_total'] = vars_tot
    denom = df_res['var_total'].replace(0.0, np.nan)  # avoid 0/0 and x/0 for zero-variance genes
    for lab, _ in blocks:
        df_res[f"prop_{lab}"] = df_res[f"var_{lab}"] / denom
    df_res['prop_residual'] = df_res['var_residual'] / denom
    _log("Done.")
    return df_res


def ancova_group_test_ridge(
    adata,
    layer = None,
    group = None,
    covar = None,
    ref_group = None,
    interactions_with_group = None,
    ridge_lambda = None,
    return_coefs = False,
    padj_method: Literal[tuple(multitest_alias.keys())] = 'fdr_bh',
    verbose = False
):
    """
    Like ancova_group_test_parametric but applies L2 (ridge) regularization
    exclusively to interaction-term coefficients (columns whose name contains ':').
    Main group and covariate effects are left unpenalized.

    The ridge penalty lambda is chosen by generalized cross-validation (GCV)
    averaged across all features when not provided. The F-test comparing the
    full ridge model to the reduced OLS model uses effective degrees of freedom
    tr(H) = tr((X^T X + lambda * Lambda)^{-1} X^T X) to account for the
    shrinkage introduced by the penalty.

    Results are stored in df_res.attrs['ridge_lambda'] and df_res.attrs['df_eff'].
    """
    from formulaic import model_matrix
    from scipy import stats, optimize
    import re
    import datetime
    import warnings

    def _log(msg):
        if verbose:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")

    def _build_design_formula(
            group_key_internal: str,
            factors_categorical: list,
            factors_continuous: list,
            interactions=None
            ) -> tuple:
        terms = [f"C({group_key_internal})"]
        if factors_categorical:
            terms += [f"C({f})" for f in factors_categorical]
        if factors_continuous:
            terms += factors_continuous
        if interactions is not None and len(np.intersect1d(factors_categorical, interactions)) > 0:
            for f in interactions:
                if f in factors_categorical:
                    terms += [f"C({group_key_internal}):C({f})"]
        if interactions is not None and len(np.intersect1d(factors_continuous, interactions)) > 0:
            for f in interactions:
                if f in factors_continuous:
                    terms += [f"C({group_key_internal}):{f}"]
        full_formula = " + ".join(terms)
        reduced_terms = [t for t in terms if not t.startswith(f"C({group_key_internal})")]
        reduced_formula = " + ".join(reduced_terms) if reduced_terms else "1"
        return full_formula, reduced_formula

    if group not in adata.obs.columns:
        raise TypeError('group should be a column in .obs.')
    idx = np.where(~adata.obs[group].isnull())[0]
    adata = adata[idx, :]
    _log(f"Input: {len(idx)} cells x {adata.shape[1]} features after dropping NaN in '{group}'.")

    if layer is None or layer not in adata.layers.keys():
        expr = adata.X
    else:
        expr = adata.layers[layer]

    cols = [group] + list(np.intersect1d(covar, adata.obs.columns)) if covar is not None else [group]
    meta = adata.obs[cols].copy()
    if meta[group].dtype.name != 'category':
        meta[group] = meta[group].astype('category')
    meta[group] = meta[group].cat.remove_unused_categories()
    g_lev = np.array(pd.Series(meta[group]).astype('category').cat.categories)
    if ref_group is not None and ref_group in g_lev:
        g_lev = np.concatenate([[ref_group], g_lev[g_lev != ref_group]])
        meta[group] = meta[group].cat.reorder_categories(g_lev)

    Y = expr.astype(np.float32).toarray() if issparse(expr) else np.array(expr, dtype=np.float32)

    design_full, design_reduced = _build_design_formula(
        group_key_internal=group,
        factors_categorical=[c for c in meta.columns if c != group and meta[c].dtype.name == 'category' and meta[c].isnull().sum() == 0],
        factors_continuous=[c for c in meta.columns if c != group and meta[c].dtype.name != 'category' and meta[c].isnull().sum() == 0],
        interactions=interactions_with_group
    )
    mm_full = model_matrix(design_full, data=meta)
    mm_reduced = model_matrix(design_reduced, data=meta)
    X_full = mm_full.astype(np.float64).values     # n x p
    X_reduced = mm_reduced.astype(np.float64).values  # n x q
    _log(f"Design matrices built. Full formula: '{design_full}' ({X_full.shape[1]} params). "
         f"Reduced formula: '{design_reduced}' ({X_reduced.shape[1]} params).")

    # Interaction columns are identified by ':' in the formulaic column name.
    interaction_col_mask = np.array([':' in col for col in mm_full.columns])
    n_interaction_cols = int(interaction_col_mask.sum())
    _log(f"Penalized (interaction) columns: {n_interaction_cols} / {X_full.shape[1]}.")

    if n_interaction_cols == 0:
        warnings.warn(
            "No interaction columns found. The ridge penalty has no effect; "
            "consider ancova_group_test_parametric instead.",
            UserWarning
        )

    # Precompute gram matrix and cross-products in float64 (Y stays float32 to save memory;
    # numpy promotes to float64 on the @ with X_full).
    XtX = X_full.T @ X_full                                      # p x p, float64
    XtY = X_full.T @ Y                                           # p x G, float64
    Y_sq_colsums = np.sum(Y ** 2, axis=0, dtype=np.float64)      # G

    Lambda_vec = np.zeros(X_full.shape[1])
    Lambda_vec[interaction_col_mask] = 1.0
    Lambda_mat = np.diag(Lambda_vec)

    def _solve(M, B):
        try:
            return np.linalg.solve(M, B)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(M, B, rcond=None)[0]

    # GCV criterion averaged across all genes (single lambda for all features).
    # GCV(lambda) = mean_gene RSS(lambda) / (1 - tr(H)/n)^2
    # where H = X (X^TX + lambda*Lambda)^{-1} X^T and tr(H) = tr(M^{-1} X^TX).
    # RSS per gene is computed via the quadratic form to avoid materialising n x G float64.
    def _gcv_criterion(log_lam: float) -> float:
        lam_ = np.exp(log_lam)
        M_ = XtX + lam_ * Lambda_mat
        beta_ = _solve(M_, XtY)
        rss_total_ = max(
            float(Y_sq_colsums.sum()
                  - 2.0 * np.sum(XtY * beta_)
                  + np.sum((XtX @ beta_) * beta_)),
            0.0
        )
        df_eff_ = float(np.trace(_solve(M_, XtX)))
        leverage_ = df_eff_ / Y.shape[0]
        if leverage_ >= 1.0:
            return np.inf
        return (rss_total_ / (Y.shape[0] * Y.shape[1])) / (1.0 - leverage_) ** 2

    if ridge_lambda is None:
        if n_interaction_cols == 0:
            lam = 0.0
        else:
            res_opt = optimize.minimize_scalar(
                _gcv_criterion,
                bounds=(-10.0, 20.0),
                method='bounded',
                options={'xatol': 1e-4}
            )
            lam = float(np.exp(res_opt.x))
            _log(f"GCV-selected lambda: {lam:.6g}")
    else:
        lam = float(ridge_lambda)
        _log(f"Using provided lambda: {lam:.6g}")

    # --- Fit ridge full model: beta = (X^T X + lambda * Lambda)^{-1} X^T Y ---
    M = XtX + lam * Lambda_mat
    beta_hat = _solve(M, XtY)   # p x G

    # Per-gene RSS via the quadratic form (avoids materialising n x G float64 Y_hat).
    XtX_beta = XtX @ beta_hat   # p x G
    rss_full = np.maximum(
        Y_sq_colsums
        - 2.0 * np.sum(XtY * beta_hat, axis=0)
        + np.sum(XtX_beta * beta_hat, axis=0),
        0.0
    )   # G

    # Effective df: tr(H) = tr(M^{-1} X^T X)  [cyclic trace identity]
    df_eff = float(np.trace(_solve(M, XtX)))
    df_rss_full = float(Y.shape[0] - df_eff)
    _log(f"Full model: df_eff={df_eff:.3f}, df_rss={df_rss_full:.3f}.")

    # --- Reduced model (OLS via QR) ---
    Qr, _ = np.linalg.qr(X_reduced, mode="reduced")
    rss_reduced = np.maximum(np.sum((Y - Qr @ (Qr.T @ Y)) ** 2, axis=0), 0.0)   # G
    df_rss_reduced = float(Y.shape[0] - X_reduced.shape[1])

    vars_tot = np.var(Y, axis=0, dtype=np.float64) * Y.shape[0]   # SS_total (ddof=0)

    # --- F-tests ---
    # ANOVA F-test: extra variation explained by group + interactions above the reduced model.
    # df_num = effective df consumed by the group + interaction terms above the reduced model.
    df_num = max(df_eff - X_reduced.shape[1], 1e-6)
    f_anova = np.maximum(0.0, (rss_reduced - rss_full) / df_num) / (rss_full / df_rss_full)
    p_anova = stats.f.sf(f_anova, df_num, df_rss_full)
    # Model F-test: ratio of explained variance per df between the two models.
    # Uses effective df for the ridge model.
    vars_full = np.maximum(vars_tot - rss_full, 0.0)
    vars_reduced_model = np.maximum(vars_tot - rss_reduced, 0.0)
    f_model = (vars_full / df_eff) / (vars_reduced_model / X_reduced.shape[1])
    p_model = stats.f.sf(f_model, df_eff, X_reduced.shape[1])
    # Residual F-test: two-tailed test on whether residual variances differ between the two models.
    # Uses effective residual df for the ridge model (df_rss_full = n - tr(H)).
    f_resi = (rss_reduced / df_rss_reduced) / (rss_full / df_rss_full)
    p_resi = 1.0 - np.abs(0.5 - stats.f.cdf(f_resi, df_rss_reduced, df_rss_full)) * 2
    _log("F-tests computed.")

    df_res = pd.DataFrame(
        {
            'var_group': np.maximum(rss_reduced - rss_full, 0.0),
            'var_covar': vars_reduced_model,
            'var_total': vars_tot,
            'p_anova': p_anova,
            'p_model': p_model,
            'p_resi': p_resi,
        },
        index=adata.var_names
    )
    for col in ['p_anova', 'p_model', 'p_resi']:
        padj_col = re.sub(r'p_', 'padj_', col)
        isnull = df_res[col].isnull()
        if isnull.sum() > 0:
            df_res[padj_col] = np.nan
            df_res.loc[~isnull, padj_col] = multipletests(df_res[col][~isnull], method=padj_method)[1]
        else:
            df_res[padj_col] = multipletests(df_res[col], method=padj_method)[1]

    df_res.attrs['ridge_lambda'] = lam
    df_res.attrs['df_eff'] = df_eff
    _log(f"Done. Lambda={lam:.4g}, df_eff={df_eff:.3f}.")

    if return_coefs:
        df_coefs = pd.DataFrame(
            beta_hat.T,
            index=adata.var_names,
            columns=mm_full.columns
        )
        _log("Coefficient matrix extracted.")
        return df_res, df_coefs
    return df_res


def make_pseudobulk(adata,
                    group,
                    stratum = None,
                    layer = None,
                    min_n_per_group = 3, # expected minimal number of pseudobulk per group
                    n_per_stratum = 3, # expected number of pseudobulk per stratum
                    min_ncells = 10, # expected minimal number of cells per pseudobulk
                    seed = None,
                    use: Literal['sum','mean'] = 'sum',
                    force_int = None,
                    return_idx = False,
                    verbose = False
                   ):
    import numpy as np
    import pandas as pd
    import itertools
    import anndata
    import datetime
    def _log(msg):
        if verbose:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")
    if seed is None:
        seed = 12345
    if force_int is None:
        force_int = (use == 'sum')
    
    if isinstance(group, str) and group in adata.obs.columns:
        group = adata.obs[group]
    elif len(group) == adata.shape[0]:
        group = pd.Series(group, index=adata.obs_names)
    else:
        raise TypeError('group should be either a .obs column or pandas.Seires object.')
    
    n_groups = len(group.unique()) if hasattr(group, 'unique') else len(set(group))
    _log(f"Input: {adata.shape[0]} cells, {n_groups} groups, {adata.shape[1]} features.")
    grouping = pd.DataFrame(group).set_axis(['__group__'], axis=1)
    grouping['__group__stratum__'] = grouping['__group__'].copy()
    if stratum is not None:
        if np.sum(np.isin(stratum, adata.obs.columns)) > 0:
            stratum = np.intersect1d(stratum, adata.obs.columns)
            stratum = adata.obs[stratum]
        if isinstance(stratum, pd.Series) and len(stratum) == adata.shape[0]:
            stratum = pd.DataFrame(stratum).set_axis(adata.obs_names).set_axis(['__stratum__'], axis=1)
        elif isinstance(stratum, list) and len(stratum) == adata.shape[0]:
            stratum = pd.DataFrame(pd.Series(stratum, index=adata.obs_names)).set_axis(['__stratum__'], axis=1)
        elif isinstance(stratum, pd.DataFrame) and stratum.shape[0] == adata.shape[0]:
            stratum = stratum.set_axis(adata.obs_names)
            stratum['__stratum__'] = stratum.apply('_'.join, axis=1).set_axis(adata.obs_names)
        else:
            raise TypeError("stratum should be either .obs columns or pandas.Series/DataFrame to stratify the data.")    
        grouping = pd.concat([ grouping, stratum ], axis=1)
        grouping['__group__stratum__'] = grouping[['__group__','__stratum__']].apply('_'.join, axis=1)
    
    df_group_meta = pd.DataFrame(grouping.value_counts().reset_index().set_axis(list(grouping.columns)+['ncells'], axis=1))
    df_group_meta = df_group_meta.set_axis(df_group_meta['__group__stratum__'], axis=0)
    _log(f"Grouping prepared. Stratification: {'yes' if stratum is not None else 'no'}.")

    idx_pseudobulk = list()
    meta_pseudobulk = list()
    for grp in grouping['__group__'].unique():
        idx = np.where(grouping['__group__'] == grp)[0]
        grouping_grp = grouping.iloc[idx,:]
        ncells_grp = len(idx)
        ncells_strata = grouping_grp['__stratum__'].value_counts()
        
        # decide whether to generate pseudobulk per stratum
        is_good_stratum = ncells_strata > (n_per_stratum * min_ncells)
        do_stratification = np.sum(is_good_stratum) * n_per_stratum >= min_n_per_group
        
        _log(f"  Group '{grp}': {ncells_grp} cells, stratified={do_stratification}.")
        if do_stratification: # generate pseudobulks per stratum
            for stra in grouping_grp['__group__stratum__'].unique():
                idx_stra = np.where(grouping['__group__stratum__'] == stra)[0]
                n_per_stratum_this = n_per_stratum
                if len(idx_stra) < n_per_stratum_this * min_ncells:
                    n_per_stratum_this = int(np.floor(len(idx_stra) * 1.0 / min_ncells))
                if n_per_stratum_this < 1:
                    n_per_stratum_this = 1
                meta = pd.DataFrame(df_group_meta.loc[[stra],:]).set_axis(df_group_meta.columns, axis=1)
                
                np.random.seed(seed)
                rand = pd.DataFrame({ 'idx' : idx_stra,
                                      'rand' : np.random.uniform(0, n_per_stratum_this, len(idx_stra)) })
                rand['rank'] = rand['rand'].rank().astype('int')
                rand['set'] = np.ceil(rand['rand']).astype('int')
                for i in range(1, n_per_stratum_this+1):
                    rand.loc[rand['rank']==i,'set'] = i
                for i in rand['set'].unique():
                    idx_pseudobulk.append(list(rand['idx'][rand['set']==i]))
                    meta_pseudobulk.append(meta)
        else: # generate pseudobulk without considering the strata
            n_per_group_this = np.max([min_n_per_group, n_per_stratum])
            if len(idx) < n_per_group_this * min_ncells:
                n_per_group_this = int(np.floor(len(idx) * 1.0 / min_ncells))
            if n_per_group_this < 1:
                n_per_group_this = 1
            meta = pd.DataFrame({ '__group__' : [grp]})
            meta['__group__stratum__'] = meta['__group__'].copy()
            
            np.random.seed(seed)
            rand = pd.DataFrame({ 'idx' : idx,
                                  'rand' : np.random.uniform(0, n_per_group_this, len(idx)) })
            rand['rank'] = rand['rand'].rank().astype('int')
            rand['set'] = np.ceil(rand['rand']).astype('int')
            for i in range(1, n_per_group_this+1):
                rand.loc[rand['rank']==i,'set'] = i
            for i in rand['set'].unique():
                idx_pseudobulk.append(list(rand['idx'][rand['set']==i]))
                meta_pseudobulk.append(meta)
    
    idx_pseudobulk = pd.DataFrame({
        'idx_bulk' : np.repeat(list(range(len(idx_pseudobulk))), [len(x) for x in idx_pseudobulk]),
        'id_bulk' : np.repeat([ 'PB_'+str(i) for i in list(range(len(idx_pseudobulk))) ], [len(x) for x in idx_pseudobulk]),
        'idx_cell' : list(itertools.chain.from_iterable(idx_pseudobulk))
    }).sort_values('idx_cell')
    idx_pseudobulk['id_cell'] = adata.obs_names[idx_pseudobulk['idx_cell']]
    meta_pseudobulk = pd.concat(meta_pseudobulk, axis=0).set_axis([ 'PB_'+str(i) for i in list(range(len(meta_pseudobulk))) ])
    meta_pseudobulk['ncells'] = idx_pseudobulk['id_bulk'].value_counts()[meta_pseudobulk.index]
    _log(f"{len(meta_pseudobulk)} pseudobulks generated (avg {meta_pseudobulk['ncells'].mean():.1f} cells each).")
    
    from .py_util import summarize_numeric_matrix
    if layer is None or layer not in adata.layers.keys():
        counts_sc = adata.X
    else:
        counts_sc = adata.layers[layer]
    _log(f"Aggregating counts (method='{use}')...")
    counts = summarize_numeric_matrix(counts_sc, group = idx_pseudobulk['id_bulk'], use_mean=(use=='mean'))
    if force_int:
        from scipy.sparse import spmatrix
        from scipy.sparse import csr_matrix
        if isinstance(counts, spmatrix):
            counts = csr_matrix(counts, dtype='int')
        else:
            counts = np.array(counts).astype(int)
    bulk_id = idx_pseudobulk['id_bulk'].astype("category").cat.categories
    adata_bulk = anndata.AnnData(X = counts, obs = meta_pseudobulk.loc[bulk_id,:], var = adata.var.copy())
    _log(f"Done. AnnData pseudobulk: {adata_bulk.shape[0]} x {adata_bulk.shape[1]}.")
    
    if return_idx:
        res = {'idx' : idx_pseudobulk, 'adata' : adata_bulk}
        return res
    else:
        return adata_bulk

def merge_clusters_topdown_anova(
    adata,
    groupby,
    linkage = None,
    stratum = None,
    covar = None,
    features = None,
    layer = None,
    min_ncells = 10,
    n_DEGs = 3,
    p_method: Literal['anova','fres'] = 'fres',
    padj_max = 0.01,
    fc_min = 1.2,
    verbose=False,
    **kwargs
):
    import scanpy as sc
    import numpy as np
    import anndata
    from statsmodels.stats.multitest import multipletests
    from scipy.cluster.hierarchy import cut_tree
    
    if groupby not in adata.obs.columns:
        raise ValueError('The provided cl_base='+cl_base+' is not a column in .obs.')

    if linkage is None:
        import warnings
        if 'dendrogram_'+groupby not in adata.uns.keys():
            warnings.warn('A pre-computed dendrogram is not found. Run `sc.tl.dendrogram` with the default setting.')
            sc.tl.dendrogram(adata, groupby=groupby)
        linkage = adata.uns['dendrogram_'+groupby]['linkage']
    elif linkage.shape[0] != len(adata.obs[groupby].unique())-1:
        raise TypeError('The provided lineage doesn\'t fit with the grouping result.')
    
    hcl_groups = pd.DataFrame(np.array([cut_tree(linkage, i).flatten() for i in range(1, linkage.shape[0]+2)]).transpose(), index = adata.obs[groupby].astype('category').cat.categories)
    hcls2compare = list()
    for i in range(1, hcl_groups.shape[1]):
        crosstab = pd.crosstab(hcl_groups.iloc[:,i-1], hcl_groups.iloc[:,i])
        grp2split = crosstab.apply(lambda x: np.sum(x>0)==2, axis=1)
        grp1, grp2 = crosstab.loc[grp2split,:].iloc[0,:][lambda x: x>0].index
        hcls2compare.append((grp1, grp2))
    
    pseudobulks = make_pseudobulk(adata,
                                  group = groupby,
                                  stratum = stratum,
                                  layer = layer,
                                  min_ncells = min_ncells,
                                  force_int = False,
                                  **kwargs
                                 )
    pseudobulks = pseudobulks[pseudobulks.obs['ncells'] >= min_ncells,:]
    
    num_fail = pd.Series(np.repeat(0, hcl_groups.shape[0]), index=hcl_groups.index)
    last_success = pd.Series(np.repeat(-1, hcl_groups.shape[0]), index=hcl_groups.index)
    traj = dict(zip(list(hcl_groups.index), 
                    [ list() for i in range(hcl_groups.shape[0]) ]
                   ))
    markers = list()
    for i in range(len(hcls2compare)):
        hcl1, hcl2 = hcls2compare[i]
        grps1 = hcl_groups.iloc[:,i+1][lambda x: x==hcl1].index
        grps2 = hcl_groups.iloc[:,i+1][lambda x: x==hcl2].index
        if num_fail[grps1].max() > 1 or num_fail[grps2].max() > 1:
            markers.append((list(), list()))
            continue
        else:
            for x in list(grps1) + list(grps2):
                traj[x].append(i)
        
        d1 = pseudobulks[np.isin(pseudobulks.obs['__group__'], grps1),:].copy()
        d1.obs['__hclgroup__'] = hcl1
        d1 = d1[:,np.array((d1.X>0).sum(0)).flatten()>0]
        d2 = pseudobulks[np.isin(pseudobulks.obs['__group__'], grps2),:].copy()
        d2.obs['__hclgroup__'] = hcl2
        d2 = d2[:,np.array((d2.X>0).sum(0)).flatten()>0]
        
        ds = anndata.concat([d1, d2])
        ds.obs['__hclgroup__'] = ds.obs['__hclgroup__'].astype('category').cat.reorder_categories([hcl1, hcl2])
        ds.X = pd.DataFrame(np.array(ds.X.todense())).set_axis(ds.obs_names, axis=0).set_axis(ds.var_names, axis=1)
        design_col = [x for x in ds.obs.columns if not(x.startswith('__') and x.endswith('__')) and ds.obs[x].isnull().sum()==0 ]

        sc.pp.normalize_total(ds, target_sum=1e4)
        sc.pp.log1p(ds)
        if features is None:
            features2check = ds.var_names
        else:
            features2check = np.intersect1d(ds.var_names, features)
        res = ancova_group_test(expr = ds[:,features2check].X,
                                group = ds.obs['__hclgroup__'],
                                covar = ds.obs[design_col] if len(design_col)>0 else None,
                                return_coef_group=ds.obs['__hclgroup__'].cat.categories[0],
                                var_names = features2check)
        res['padj_Resi'] = multipletests(res['p_Resi'], method='fdr_bh')[1]
        res['padj_ANCOVA'] = multipletests(res['p_ANCOVA'], method='holm')[1]
        if p_method == 'anova':
            idx_DE = np.where((res['padj_ANCOVA'] < padj_max) & (res['coef'].abs() > np.log(fc_min)))[0]
        else:
            idx_DE = np.where((res['padj_Resi'] < padj_max) & (res['coef'].abs() > np.log(fc_min)))[0]
        markers_hcl1 = res.iloc[idx_DE,:]['coef'][lambda x: x>0].sort_values(ascending=False).index
        markers_hcl2 = res.iloc[idx_DE,:]['coef'][lambda x: x<0].sort_values(ascending=True).index
        markers.append((markers_hcl1, markers_hcl2))
        
        if len(idx_DE) > n_DEGs:
            num_fail[grps1] = 0
            num_fail[grps2] = 0
            last_success[grps1] = i
            last_success[grps2] = i
        else:
            num_fail[grps1] = num_fail[grps1]+1
            num_fail[grps2] = num_fail[grps2]+1
            if num_fail.min() > 0:
                break

    hcl2take = last_success.copy()
    cl_merged = pd.DataFrame(
        {'iter_hcl' : hcl2take+1,
         'idx_hcl' : [ hcl_groups.iloc[i,hcl2take.iloc[i]+1] for i in range(len(hcl2take)) ]
        },
        index = hcl_groups.index.copy()
    )
    cl_merged['merged_cl'] = cl_merged.apply(lambda x: '_'.join([str(y) for y in x]), axis=1)
    
    res = {
        'merged_cluster' : cl_merged,
        'splitting_trajectory' : traj,
        'splitting_markers' : markers
    }
    return res