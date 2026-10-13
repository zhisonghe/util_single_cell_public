# Single-cell analysis utilities

This repository is a collection of Python helpers and analysis scripts for
single-cell genomic data. Most routines use `AnnData`/Scanpy objects or
NumPy/SciPy matrices. The files are standalone modules rather than a packaged
library; import a function from its file or run a script where noted.

## Contents

| File | Purpose |
| --- | --- |
| [`py_util.py`](./py_util.py) | General matrix, cell selection, pseudocell, QC, and plotting helpers |
| [`py_wknn.py`](./py_wknn.py) | Weighted nearest-neighbor graphs, reference-presence scores, and label transfer |
| [`differential_expression.py`](./differential_expression.py) | Feature-wise statistical tests, variance decomposition, pseudobulk, and cluster merging |
| [`cell_type_composition.py`](./cell_type_composition.py) | Covariate-adjusted cell-type composition residuals |
| [`build_VAE_models.py`](./build_VAE_models.py) | Train SCVI, SCANVI, and scPoli models and create synthetic sample labels |
| [`mapping_scarches.py`](./mapping_scarches.py) | Load and prepare query data for reference mapping with scVI/scArches models |
| [`run_scvelo_pt.py`](./run_scvelo_pt.py) | Run scVelo velocity and pseudotime inference |
| [`displacement_field.py`](./displacement_field.py) | Train a smooth neural displacement field for embedding alignment |
| [`__.py`](./__.py) | Empty Python file; no functions or classes are currently defined |

## File and method guide

### `py_util.py`

General-purpose single-cell data helpers. Sparse matrices are supported by
several matrix operations.

| Method | What it does |
| --- | --- |
| `run_NMF(X, ...)` | Fits or applies a scikit-learn NMF model, or factorizes using supplied `W`/`H`; returns factors or the model. |
| `run_PAGA_only(path, rep, k, grp, ...)` | Reads an H5AD file, computes neighbors and PAGA connectivity, optionally overwrites the file, and returns the updated `AnnData`. |
| `corSparse(X, Y=None)` | Computes column-wise Pearson correlations between two matrices (or among columns of `X` when `Y` is omitted). |
| `rankMatrix(X)` | Ranks values by column; its sparse path preserves sparse storage and ranks observed nonzero entries with an offset for implicit zeros. |
| `rankMatrix_dense(X)` | Sparse-aware ranking variant using dense ranks for nonzero entries; dense inputs use average ranks. |
| `rankMatrix_nonzero(X)` | Ranks nonzero sparse entries while leaving implicit zeros as zero; dense inputs use average ranks. |
| `select_subset_cells(adata, ...)` | Selects a Boolean cell mask using random sampling, geometric sketching, or a mixture of both. |
| `generate_aggrecell_idx(adata, ...)` | Builds a Scanpy kNN graph, optionally constrains edges by PAGA-connected groups, and assigns cells to graph-based pseudocells. |
| `group_vec_to_ident_mat(group, ...)` | Builds a sparse group-membership matrix, optionally normalizing each group column. |
| `summarize_numeric_matrix(mat, group, ...)` | Aggregates rows by group using means by default, or sums when requested. |
| `summarize_categorical_vector(fac, group)` | Assigns each group the most frequent category in a vector. |
| `summarize_dataframe(df, group)` | Summarizes numeric columns by mean, Boolean columns by majority, and other columns by most frequent category. |
| `generate_aggrecells(adata, aggrecell_idx, ...)` | Creates an aggregated `AnnData` object from pseudocell assignments, including summarized observation and variable metadata. |
| `find_valley_between_bimodal(x, ...)` | Uses a kernel density estimate to locate the valley between two sufficiently prominent peaks; can plot the density. |
| `find_maximums_in_distribution(x, ...)` | Finds prominent local maxima in a kernel density estimate; can plot on a supplied or current axes. |
| `is_not_outliers(x, min_p, max_p)` | Returns a Boolean mask for observations within quantile-derived standardized-normal cutoffs. |
| `do_QC(ad, ...)` | Filters cells by a detected low-gene-count valley (optionally) and outlier thresholds for gene count and mitochondrial percentage. |
| `is_logged(mat, ...)` | Heuristically checks whether matrix column maxima contain non-integer values, a hint that data may be log-transformed. |
| `subsample(x, size, N, ...)` | Returns an item-by-subsample Boolean matrix; can make repeated random samples or distribute selections to cover items across subsamples. |
| `split_embedding(adata, split_by, ...)` | Draws the same embedding in separate panels for each category of an observation column. |

### `py_wknn.py`

Constructs sparse neighbor graphs between reference and query representations
and uses them for matching, presence estimation, and annotation transfer.

| Method | What it does |
| --- | --- |
| `gaussian_kernel(d, sigma=None)` | Converts distances to Gaussian weights. |
| `nn2adj(nn, ...)` | Converts neighbor indices and distances into a sparse adjacency matrix with unweighted, distance, or Gaussian weights. |
| `build_nn(ref, query=None, ...)` | Builds a query-to-reference kNN graph, using cuML when available and enabled or PyNNDescent otherwise. |
| `build_mutual_nn(dat1, dat2=None, ...)` | Returns reciprocal nearest-neighbor links within one dataset or between two datasets. |
| `get_transition_prob_mat(dat, k, symm)` | Constructs a kNN graph and returns its row-normalized transition-probability matrix. |
| `random_walk_with_restart(init, transition_prob, ...)` | Smooths an initial per-cell score vector by repeatedly propagating it through a graph with restart. |
| `get_wknn(ref, query, ...)` | Builds a weighted query-reference neighbor matrix. Supports directed/bidirectional links and neighbor-count, Jaccard, Gaussian, or distance weighting; optionally returns intermediate adjacency matrices. |
| `estimate_presence_score(ref_adata, query_adata, ...)` | Scores reference-cell presence from weighted query-reference links, optionally by a query metadata group and smoothed over a reference graph. Returns maximum and per-group normalized scores, the weighted graph, and transition matrix. |
| `get_matched_transcriptome(adata, adata_ref, wknn, ...)` | Computes a weighted reference-expression profile for each query observation, with optional use of `.raw` and reference expression rescaling. |
| `calculate_pairwise_correlation(adata1, adata2, ...)` | Calculates Pearson correlation between matched cell expression profiles across shared (or selected) genes; supports parallel execution. |
| `transfer_labels(ref_adata, query_adata, wknn, label_key)` | Transfers reference labels by weighted voting; returns per-label scores, each query's best label, and its best score. |
| `hierarchical_region_lab_transfer(...)` | Aggregates transferred labels into a supplied region hierarchy and assigns a hierarchical label; optionally smooths scores on a query graph. |
| `hierarchical_region_lab_transfer_braun(...)` | Convenience wrapper that transfers `Subregion` labels using a predefined brain-region hierarchy. |
| `cmd_interface()` | Parses command-line options for the script interface. When run as a script, reads reference/query H5AD files, saves a weighted graph and presence-score table, and optionally writes transferred labels. |

The command-line interface requires query and reference H5AD paths. It can
select representations and neighbor counts, smooth presence scores, split
scores by a query observation column, and transfer a reference label column.

### `differential_expression.py`

Feature-wise statistical testing and variance analysis. Many tests accept a
cell-by-feature matrix; the parametric and variance-dissection routines accept
`AnnData` and use `.X` or a selected layer. Results are generally returned as
Pandas data frames indexed by feature.

| Method | What it does |
| --- | --- |
| `f_nonzero(e, covar=None)` | Fits a per-feature linear model and tests the intercept/nonzero signal, optionally adjusting for covariates; returns F statistic, intercept coefficient, and p-value. |
| `f_nonzero_test(expr, ...)` | Applies `f_nonzero` across features, optionally in parallel, and adjusts p-values for multiple testing. |
| `scipy_test(expr, method, ...)` | Applies a one-sample t-test, Wilcoxon signed-rank test, or Mann–Whitney test against zero to each feature; returns statistic, mean, raw p-value, and adjusted p-value. |
| `anova(e, group, covar=None, ...)` | Compares covariate-only and covariate-plus-group linear models for one feature; returns group/covariate/total variation, p-values, and an optional group coefficient. |
| `ancova_group_test(expr, group, ...)` | Runs `anova` across features, with optional covariates, parallelism, selected group coefficient, and multiple-testing correction. |
| `ancova_group_test_parametric(adata, ...)` | Uses full and reduced linear models with QR decomposition to test a group effect across features, optionally including group-by-covariate interactions. Returns variance components and three F-test p-values (adjusted), and optionally model coefficients. |
| `dissect_variance_parametric(adata, sources, ...)` | Decomposes feature variance into marginal contributions from multiple observation sources, optional source interactions, and an optional combined nuisance-covariate block. |
| `dissect_variance_sequential_parametric(adata, sources, ...)` | Sequential counterpart to the marginal decomposition: adds covariates, sources, and interactions in order. Attribution is order-dependent, and the components sum to explained variance. |
| `ancova_group_test_ridge(adata, ...)` | ANCOVA variant that ridge-penalizes interaction coefficients only; selects the penalty by generalized cross-validation when not supplied and reports it and the effective degrees of freedom in result attributes. |
| `make_pseudobulk(adata, group, ...)` | Aggregates cells into sum- or mean-based pseudobulk profiles, optionally stratifying by metadata and controlling desired replicate/cell counts; can also return cell-to-pseudobulk indices. |
| `merge_clusters_topdown_anova(adata, groupby, ...)` | Traverses a hierarchical cluster linkage, tests candidate splits using pseudobulk profiles and ANCOVA, and returns merged-cluster labels, split trajectories, and marker sets. |

The variance-dissection routines require nonempty `sources` observation
columns. Categorical observation columns are modeled as factors; other columns
are treated as continuous. The marginal routine attributes shared variance
using reduced models, while the sequential routine attributes it according to
the specified term order.

### `cell_type_composition.py`

Provides covariate-adjusted composition scores from sample-by-cell-type count
tables.

| Method | What it does |
| --- | --- |
| `_counts_to_weighted_long(counts_df, meta_df)` | Converts positive sample/cell-type counts into long-form observations and uses counts as sample weights, avoiding expansion to one row per cell. |
| `_fit_sklearn_multinomial_model(counts_df, metadata_df, covariate_columns, ...)` | Fits a multinomial logistic-regression pipeline for cell type as a function of metadata; one-hot encodes categorical predictors and passes numeric predictors through. |
| `multinomial_pearson_residuals(counts_df, metadata_df, covariate_columns, ...)` | Fits or reuses that model, estimates expected cell-type proportions, and returns Pearson residuals per sample and cell type. Positive/negative residuals indicate enrichment/depletion relative to covariate-predicted composition. Optionally returns the fitted model. |

The public input convention is rows = samples in both tables, columns =
cell types in `counts_df`, and metadata indexed to the same samples. The
function adds a pseudocount of `0.5` to each count by default.

### `build_VAE_models.py`

Convenience wrappers for training reference models and generating artificial
sample/batch labels.

| Method | What it does |
| --- | --- |
| `run_scvi(adata, ...)` | Registers the `AnnData` batch/layer with SCVI, trains a variational autoencoder, optionally saves it, and returns the model. |
| `run_scanvi_from_scvi(adata, vae_scvi, labels_key, unlabeled_category, ...)` | Initializes SCANVI from a trained SCVI model and trains it with labeled and unlabeled cell categories. |
| `run_scpoli(adata, condition_keys, cell_type_keys, ...)` | Constructs and trains a scPoli model; optionally saves it and its conditional sample embeddings. |
| `plot_scvi_scanvi_lost(vae)` | Plots training reconstruction loss and validation reconstruction loss when available. |
| `plot_scpoli_lost(scpoli_model)` | Plots scPoli epoch and validation losses. |
| `make_artificial_sample_labels(adata, dataset_key, sample_key, new_sample_key, ...)` | Splits each sample into fixed/unfixed cells and pairs unfixed portions across datasets to create synthetic mixed-sample labels. Supports supplied pairings and pairing priorities/weights; can return labels or a modified object. |

With default options, `make_artificial_sample_labels` writes the new labels into
the supplied object's `.obs` and returns `None`. Set `return_lab=True` to
return the label series instead, or `copy=True` to work on and return a copy.

### `mapping_scarches.py`

Helpers for checking and adapting query `AnnData` objects before mapping them
to a trained SCVI, SCANVI, or scPoli reference model.

| Method | What it does |
| --- | --- |
| `is_lognorm(mat, ...)` | Uses non-integer column maxima as a heuristic for whether a matrix is log-normalized. |
| `load_vae(model_path, model_type, ref, **kwargs)` | Loads an SCVI, SCANVI, or scPoli model from a saved path and reference object; rejects unsupported model-type strings. |
| `prepare_query_anndata(adata, ref, vae, ...)` | Copies the query, optionally checks expression scale, maps batch/condition labels, prepares scPoli unknown cell-type labels, and aligns query features to the reference model. |
| `train_scarches(adata, vae, ...)` | Loads query data into the supported reference model and trains the query-adapted SCVI/SCANVI/scPoli model; optionally saves it. |
| `get_latent_space(adata, vae, ref_adata=None, col_batch=None)` | Obtains latent representations for query cells, optionally together with reference cells, and returns one matrix or separate `ref`/`query` matrices. |

`prepare_query_anndata` checks data scale unless `skip_validation=True`.
Features and observation metadata must be compatible with the selected model,
including any batch or condition keys it expects.

### `run_scvelo_pt.py`

| Method | What it does |
| --- | --- |
| `run_scvelo_pt(filename, use_rep, ...)` | Reads a dataset with scVelo, optionally filters genes, normalizes and log-transforms, computes moments, RNA velocity, velocity graph, recovered dynamics, latent time, and velocity pseudotime. Optionally writes the processed H5AD and returns the two time columns from `.obs`. |

### `displacement_field.py`

Learns a smooth mapping between source and target representations from
mutual-nearest-neighbor (MNN) pairs. The example at the bottom of the file
trains on synthetic data when the file is executed directly.

| Class | Methods and role |
| --- | --- |
| `LipschitzConstrainedLinear` | Spectrally normalized linear layer scaled by a configured Lipschitz bound; exposes `forward` and `get_spectral_norm()`. |
| `SmoothDisplacementNet` | Neural network predicting displacement vectors with normalized/dropout hidden layers and a residual connection. Exposes `forward`, weight initialization, and theoretical/per-layer Lipschitz-bound reporting. |
| `SimpleProgressBar` | PyTorch Lightning callback that reports epoch, batch, and validation progress. |
| `SmoothDisplacementField` | Lightning module combining displacement MSE with neighbor-consistency, total-variation, gradient, and manifold-preservation penalties; defines training/validation steps and optimizer scheduling. |
| `SmoothDisplacementFieldTrainer` | High-level API: `fit(X_source, X_target, mnn_pairs, ...)` trains from index pairs; `predict(X)` returns displacements; `transform(X)` applies them and returns the data in the original scale; `analyze_smoothness(X_sample, ...)` reports local smoothness and displacement metrics. |

The file lists its core dependencies in a comment: PyTorch, PyTorch Lightning,
scikit-learn, and tqdm. Its trainer currently configures the Lightning trainer
with a GPU accelerator, so fitting requires a compatible GPU setup even
though the helper also detects a CPU device.

## Dependencies and usage notes

There is no dependency manifest or project-level test suite in the repository.
The modules import scientific Python packages including NumPy, pandas, SciPy,
scikit-learn, Scanpy, AnnData, igraph, PyNNDescent, scVelo, scvi-tools,
scArches, and (for the displacement model) PyTorch and PyTorch Lightning.
Some optional paths use geosketch, cuML, joblib, formulaic, statsmodels, and
tqdm. Install the packages needed by the specific module and workflow you
intend to use.

Several functions operate on or update `AnnData` objects and assume that
metadata columns, layers, representations, and category levels named in their
arguments already exist. Check each function's arguments and the relevant
input conventions above before applying it to a dataset.
