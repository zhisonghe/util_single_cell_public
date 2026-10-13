import datetime
import numpy as np
import pandas as pd

def _counts_to_weighted_long(
        counts_df,
        meta_df,
        ) -> tuple:
    """
    Convert sample x cell-type counts into long format suitable for sklearn.

    Multinomial logistic regression in sklearn expects one observation per row.
    The natural representation of cell type count data is to treat each cell as
    an independent observation: a sample with 50 T cells and 100 B cells would
    ideally contribute 150 rows. Instead, we keep one row per (sample, cell_type)
    pair and pass the count as a sample_weight, which is mathematically equivalent
    but far more memory-efficient.

    Zero-count (sample, cell_type) pairs are omitted since they carry no
    information for the weighted likelihood.

    Returns
    -------
    X_long:
        metadata rows repeated once per sample-celltype combination.

    y_long:
        cell-type label for each row.

    sample_weight:
        count for each sample-celltype combination.
    """
    rows = []

    for sample_id in counts_df.index:
        meta = meta_df.loc[sample_id]
        counts = counts_df.loc[sample_id]

        for cell_type, count in counts.items():
            if count > 0:
                row = meta.to_dict()
                row["sample_id"] = sample_id
                row["cell_type"] = cell_type
                row["weight"] = count
                rows.append(row)

    long_df = pd.DataFrame(rows)

    X_long = long_df.drop(columns=["cell_type", "weight"])
    y_long = long_df["cell_type"].values
    sample_weight = long_df["weight"].values.astype(float)

    return X_long, y_long, sample_weight, long_df

def _fit_sklearn_multinomial_model(
    counts_df,
    metadata_df,
    covariate_columns,
    C=1e6,
    max_iter=5000,
    l1_ratio=0.0
):
    """
    Fit multinomial logistic regression using sklearn.

    The model learns P(cell_type | covariates), i.e. how much of the observed
    variation in cell type proportions can be explained by the specified
    covariates (e.g. batch, sequencing depth, donor age).

    C is set large by default to approximate unregularized MLE; set it smaller
    to apply L2 regularization, or adjust l1_ratio for elastic-net mixing.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    X_long, y_long, weights, long_df = _counts_to_weighted_long(
        counts_df,
        metadata_df[covariate_columns]
    )

    # Separate categorical and numeric covariates for appropriate preprocessing.
    # Categorical variables are one-hot encoded (first level dropped to avoid
    # perfect multicollinearity); numeric variables are passed through as-is.
    categorical_cols = [ col for col in covariate_columns if metadata_df[col].dtype.name in ["object", "category", "bool"] ]
    numeric_cols = [ col for col in covariate_columns if col not in categorical_cols ]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_cols
            ),
            (
                "numeric",
                "passthrough",
                numeric_cols
            )
        ],
        remainder="drop"
    )

    clf = LogisticRegression(
        solver="lbfgs",
        C=C,
        max_iter=max_iter,
        l1_ratio=l1_ratio,
    )
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf)
        ]
    )
    # Pass cell counts as sample weights so each cell contributes equally to
    # the likelihood, regardless of how many cells were sequenced per sample.
    pipe.fit(
        X_long[covariate_columns],
        y_long,
        model__sample_weight=weights
    )

    return pipe

def multinomial_pearson_residuals(
        counts_df,
        metadata_df,
        covariate_columns,
        model = None,
        C=1e6,
        max_iter=5000,
        l1_ratio=0.0,
        return_model=False,
        pseudocount=0.5,
        verbose=True
    ):
    """
    Compute Pearson residuals from a multinomial logistic regression model to
    obtain covariate-adjusted cell type composition scores.

    The procedure is:
      1. (Optionally) add pseudocounts to each sample x cell-type entry for
         robustness against zero-count cell types.
      2. Fit (or reuse) a multinomial logistic regression of cell type ~ covariates.
      3. For each sample, obtain covariate-predicted cell type probabilities P.
      4. Compute expected counts E = n * P, where n is the total cell count per sample.
      5. Compute marginal binomial variance Var = n * P * (1 - P).
      6. Return Pearson residuals (Y - E) / sqrt(Var).

    Positive residuals indicate a cell type is more abundant than expected given
    the covariates; negative residuals indicate depletion. These residuals can be
    used directly for downstream comparisons across conditions of interest.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Sample x cell-type matrix of raw cell counts.
    metadata_df : pd.DataFrame
        Sample-level metadata; index must match counts_df.
    covariate_columns : list of str
        Columns in metadata_df to regress out (e.g. batch, donor).
    model : fitted sklearn Pipeline, optional
        Pre-fitted model from a previous call (with return_model=True).
        If None, a new model is fitted on counts_df and metadata_df.
    C : float
        Inverse regularization strength for LogisticRegression.
    max_iter : int
        Maximum number of solver iterations.
    l1_ratio : float
        Elastic-net mixing parameter (0 = pure L2, 1 = pure L1).
    return_model : bool
        If True, also return the fitted model for reuse.
    pseudocount : float
        Value added to every entry of counts_df before model fitting and
        residual computation. Smooths probability estimates and prevents
        division-by-zero for cell types absent in some samples. Default 0.5
        (no smoothing).
    verbose : bool
        If True, print timestamped progress messages to stdout.

    Returns
    -------
    residuals : pd.DataFrame
        Sample x cell-type Pearson residual matrix.
    model : sklearn Pipeline (only when return_model=True)
        The fitted multinomial logistic regression pipeline.
    """
    def _ts():
        """Return a bracketed timestamp string for verbose messages."""
        return datetime.datetime.now().strftime("[%H:%M:%S]")

    n_samples, n_celltypes = counts_df.shape

    # Apply pseudocounts before fitting and residual computation so that
    # zero-count (sample, cell_type) pairs receive a small positive weight,
    # preventing degenerate probability estimates for rare cell types.
    if pseudocount != 0:
        counts_df = counts_df + pseudocount
        if verbose:
            print(f"{_ts()} Applied pseudocount of {pseudocount} to all entries.")

    if model is None:
        if verbose:
            print(
                f"{_ts()} Fitting multinomial logistic regression "
                f"({n_samples} samples, {n_celltypes} cell types, "
                f"covariates: {covariate_columns})..."
            )
        model = _fit_sklearn_multinomial_model(
            counts_df,
            metadata_df,
            covariate_columns,
            C=C,
            max_iter=max_iter,
            l1_ratio=l1_ratio
        )
        if verbose:
            print(f"{_ts()} Model fitted.")
    else:
        if verbose:
            print(f"{_ts()} Using provided pre-fitted model.")

    if verbose:
        print(f"{_ts()} Computing covariate-predicted cell type probabilities...")

    # Predict P(cell_type | covariates) for each sample using the fitted model.
    # predict_proba returns an (n_samples, n_celltypes) matrix; columns are
    # ordered by model.classes_.
    probs = model.predict_proba(metadata_df[covariate_columns])
    cell_types = model.named_steps["model"].classes_

    predicted_cov_props = pd.DataFrame(
        probs,
        index=metadata_df.index,
        columns=cell_types
    )
    # Align counts_df columns to the order returned by the model.
    counts_df = counts_df[predicted_cov_props.columns]

    if verbose:
        print(f"{_ts()} Computing Pearson residuals...")

    Y = counts_df.values.astype(float)           # observed counts (samples x cell types)
    P = predicted_cov_props.values.astype(float) # covariate-predicted probabilities

    # Total cell count per sample, used to scale probabilities to expected counts.
    n = Y.sum(axis=1, keepdims=True)

    # Expected count under the covariate model: E[Y_ij] = n_i * P_ij
    expected = n * P

    # Marginal binomial variance of the j-th cell type count in sample i.
    # Under a multinomial model, Var[Y_ij] = n_i * P_ij * (1 - P_ij).
    # A small floor is applied to avoid division by zero for rare cell types.
    var = n * P * (1 - P)
    var = np.maximum(var, 1e-8)

    # Pearson residual: standardized deviation of observed from expected.
    # Values > 0 indicate enrichment beyond covariate expectation;
    # values < 0 indicate depletion.
    resid = (Y - expected) / np.sqrt(var)

    if verbose:
        print(
            f"{_ts()} Done. Residual matrix shape: "
            f"{resid.shape[0]} samples x {resid.shape[1]} cell types."
        )

    if return_model:
        return pd.DataFrame(
            resid,
            index=counts_df.index,
            columns=counts_df.columns
        ), model
    else:
        return pd.DataFrame(
            resid,
            index=counts_df.index,
            columns=counts_df.columns
        )
