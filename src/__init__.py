"""
BOD/COD surrogate-modelling package.

Notebooks import from the submodules directly (e.g. `from src.evaluation import
evaluate_model_oof`); the names below are re-exported for convenience.
Importing `src` does not import xgboost or lightgbm — those load lazily inside
models.fit_xgboost / models.fit_lightgbm.
"""
from .config import *
from .preprocessing import (
    target_forward, target_inverse, target_inverse_smeared, assert_target_consistent,
    FoldSafePCA, build_preprocessor, pca_block_feature_names, inspect_pca_block,
)
from .data_loading import (
    load_locked_dataset, build_datasets, run_integrity_checks, print_checks, check_status,
    observation_key, add_calendar_columns, as_bool, load_phase2_lock,
    find_cross_river_duplicates, exclude_flagged_cross_river_duplicates,
    apply_do_substitution, do_reporting_floor, whole_number_share,
)
from .folds import (
    build_loso_river, build_loso_station, build_fc,
    build_stratified_kfold, build_random_splits,
    make_inner_cv, inner_scheme_for, inner_cv_for_folds, ForwardChainGroupSplit,
    save_all_folds, load_folds, build_fold_keys, save_fold_keys, verify_fold_keys,
)
from .evaluation import (
    evaluate_model, evaluate_model_oof, evaluate_lookup_baseline, evaluate_mixedlm,
    compute_metrics, summarize_cv, summarize_with_pooled, smeared_metrics,
    pooled_threshold_metrics, skill_vs_reference, protocol_comparison, exceeds,
)
from .models import (
    make_pipeline, tune,
    fit_ridge_scaled, fit_ols, fit_ridge_cv, fit_lasso_cv, fit_elasticnet_cv,
    fit_random_forest, fit_xgboost, fit_lightgbm,
    fit_mixedlm_river, fit_mixedlm_random_slope_DO,
    predict_mixedlm_new_group, predict_mixedlm_known_group,
    GrandMeanBaseline, GroupMeanBaseline,
)
from .stats import (
    true_icc, nested_variance_decomposition, partial_corr,
    chow_test, clustered_break_test, vif, cooks_distance,
    cluster_bootstrap, bootstrap_summary,
)
from .provenance import provenance
