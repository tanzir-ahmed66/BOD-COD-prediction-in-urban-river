"""
Model fitting functions.

Every tuned estimator is built as

    GridSearchCV( Pipeline([preprocessing..., estimator]), grid, cv=inner_cv )

not

    Pipeline([ preprocessing..., GridSearchCV(estimator, grid) ])

The second shape would fit the StandardScaler (and, for FS-E, the PCA) once on
the whole outer training set and then run the inner CV on already-transformed
data, leaking the inner validation folds' means, SDs and covariance structure
into every candidate hyperparameter. With this nesting, preprocessing is refit
from scratch inside each inner fold.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .config import (
    SEED, RIDGE_ALPHAS, RIDGE_GRID, LASSO_GRID, ENET_GRID, RF_GRID, XGB_GRID, LGBM_GRID,
)
from .preprocessing import build_preprocessor


# ═══════════════════════════════════════════════════════════════
# PIPELINE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════

def make_pipeline(estimator, feature_cols: list, pca_block: str | None = None) -> Pipeline:
    """
    Preprocessing + estimator as ONE Pipeline, named 'est' at the tail.

    Grids in config.py use the 'est__' prefix, so any estimator can be swapped
    in without touching the grid.
    """
    pre = build_preprocessor(feature_cols, pca_block=pca_block)
    steps = list(pre.steps) + [("est", estimator)]
    return Pipeline(steps)


def tune(estimator, grid: Dict, feature_cols: list, inner_cv,
         pca_block: str | None = None, scoring: str = "r2",
         n_jobs: int = -1, refit: bool = True) -> GridSearchCV:
    """
    Wrap the FULL pipeline in GridSearchCV.

    inner_cv : splitter from folds.make_inner_cv(). If it is a grouped or
               forward-chaining splitter, `groups=` must be supplied at fit time
               -- evaluation.evaluate_model does this for you.
    """
    return GridSearchCV(
        make_pipeline(estimator, feature_cols, pca_block=pca_block),
        param_grid=grid,
        cv=inner_cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=refit,
        error_score="raise",   # a failed fit must stop the run, not score NaN like a bad alpha
    )


# ═══════════════════════════════════════════════════════════════
# LINEAR MODELS (Phase 3, section 3.3)
# ═══════════════════════════════════════════════════════════════

def fit_ridge_scaled(feature_cols: list, alpha: float = 1.0,
                     pca_block: str | None = None) -> Pipeline:
    """Fixed-alpha ridge. No tuning, so no inner CV needed."""
    # NB: Ridge's random_state only affects the 'sag'/'saga' solvers; the
    # default solver is deterministic, so none is set.
    return make_pipeline(Ridge(alpha=alpha), feature_cols, pca_block)


def fit_ols(feature_cols: list, pca_block: str | None = None) -> Pipeline:
    return make_pipeline(LinearRegression(), feature_cols, pca_block)


def _as_grid(values, default) -> Dict:
    """
    Accept a list of alphas, a numpy array, or a full param-grid dict.

    Avoids `list(alphas or RIDGE_ALPHAS)`, which raises "truth value of an
    array is ambiguous" for a numpy array (e.g. np.logspace(...)).
    """
    if values is None:
        values = default
    if isinstance(values, dict):
        return values
    return {"est__alpha": [float(a) for a in values]}


def fit_ridge_cv(feature_cols: list, inner_cv, alphas=None,
                 pca_block: str | None = None) -> GridSearchCV:
    return tune(Ridge(), _as_grid(alphas, RIDGE_ALPHAS), feature_cols, inner_cv, pca_block)


def fit_lasso_cv(feature_cols: list, inner_cv, alphas=None,
                 pca_block: str | None = None) -> GridSearchCV:
    # Lasso uses LASSO_GRID: lasso and ridge alphas are on different scales
    # (lasso's penalty is not squared and is divided by n).
    return tune(Lasso(random_state=SEED, max_iter=10000), _as_grid(alphas, LASSO_GRID),
                feature_cols, inner_cv, pca_block)


def fit_elasticnet_cv(feature_cols: list, inner_cv, grid: Dict = None,
                      pca_block: str | None = None) -> GridSearchCV:
    return tune(ElasticNet(random_state=SEED, max_iter=10000), grid or ENET_GRID,
                feature_cols, inner_cv, pca_block)


# ═══════════════════════════════════════════════════════════════
# TREE-BASED MODELS (Phase 3, section 3.4)
# ═══════════════════════════════════════════════════════════════

def fit_random_forest(feature_cols: list, inner_cv, grid: Dict = None,
                      pca_block: str | None = None) -> GridSearchCV:
    from sklearn.ensemble import RandomForestRegressor
    return tune(RandomForestRegressor(random_state=SEED, n_jobs=1),
                grid or RF_GRID, feature_cols, inner_cv, pca_block)


def fit_xgboost(feature_cols: list, inner_cv, grid: Dict = None,
                pca_block: str | None = None) -> GridSearchCV:
    from xgboost import XGBRegressor
    return tune(XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0),
                grid or XGB_GRID, feature_cols, inner_cv, pca_block)


def fit_lightgbm(feature_cols: list, inner_cv, grid: Dict = None,
                 pca_block: str | None = None) -> GridSearchCV:
    # config.LGBM_GRID mirrors XGB_GRID; LightGBM accepts those names as aliases.
    # num_leaves / min_child_samples stay at defaults unless a grid is passed.
    from lightgbm import LGBMRegressor
    return tune(LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1),
                grid or LGBM_GRID, feature_cols, inner_cv, pca_block)


# ═══════════════════════════════════════════════════════════════
# HIERARCHICAL / MIXED-EFFECTS (not used by the notebooks)
# ═══════════════════════════════════════════════════════════════
# These are NOT sklearn estimators. Run them through
# evaluation.evaluate_mixedlm(), which handles fold-safe scaling, the
# new-river vs known-river prediction split, and convergence failures.

def fit_mixedlm_river(df_train: pd.DataFrame, feature_cols: list,
                      target_col: str = "log_BOD", group_col: str = "River",
                      reml: bool = True, re_formula: str | None = None) -> Tuple:
    """
    Frequentist mixed-effects model with group random intercepts
    (plus random slopes if re_formula is given, e.g. "~log1p_DO").

    Returns (result, formula). Raises on non-convergence -- callers that need
    to survive a bad fold should catch, not this function.
    """
    import statsmodels.formula.api as smf

    formula = f"{target_col} ~ " + " + ".join(feature_cols)
    model = smf.mixedlm(formula, data=df_train, groups=df_train[group_col],
                        re_formula=re_formula)
    result = model.fit(reml=reml)
    return result, formula


def fit_mixedlm_random_slope_DO(df_train: pd.DataFrame, feature_cols: list,
                                target_col: str = "log_BOD",
                                group_col: str = "River",
                                slope_var: str = "log1p_DO") -> Tuple:
    """Random intercept + random slope on DO."""
    return fit_mixedlm_river(df_train, feature_cols, target_col, group_col,
                             re_formula=f"~{slope_var}")


def _fixed_effects_vector(result, feature_cols: list) -> Tuple[float, np.ndarray]:
    fe = result.fe_params
    missing = [c for c in feature_cols if c not in fe.index]
    if missing:
        raise KeyError(
            f"Fixed effects missing {missing}. Patsy renamed them (categoricals?) "
            f"or the formula differs from feature_cols. Have: {list(fe.index)}"
        )
    return float(fe["Intercept"]), np.array([fe[c] for c in feature_cols], dtype=float)


def predict_mixedlm_new_group(result, df_test: pd.DataFrame,
                              feature_cols: list) -> np.ndarray:
    """
    Held-out group: FIXED EFFECTS ONLY (u_j = 0).
    This is the honest LOSO-River deployment scenario -- a new river has no
    estimable random intercept.
    """
    b0, betas = _fixed_effects_vector(result, feature_cols)
    return b0 + df_test[feature_cols].to_numpy(dtype=float) @ betas


def predict_mixedlm_known_group(result, df_test: pd.DataFrame, feature_cols: list,
                                group_name, slope_var: str | None = None) -> np.ndarray:
    """
    Known group: fixed effects + that group's BLUP (intercept, and slope if the
    model has one). LOSO-Station scenario.

    statsmodels names the random intercept "Group" for a plain random-intercept
    model and "Intercept" once re_formula is supplied, and the entry is a Series
    not a dict -- so `.get(name, {}).get("Group", 0.0)` would silently return
    0.0 for every random-slope model.
    """
    b0, betas = _fixed_effects_vector(result, feature_cols)
    X = df_test[feature_cols].to_numpy(dtype=float)
    pred = b0 + X @ betas

    re = result.random_effects
    if group_name not in re:
        return pred                       # unseen group -> fixed effects only
    u = re[group_name]

    icept_key = next((k for k in ("Group", "Intercept", "group") if k in u.index), None)
    if icept_key is None and len(u):
        icept_key = u.index[0]
    if icept_key is not None:
        pred = pred + float(u[icept_key])

    if slope_var is not None and slope_var in u.index:
        pred = pred + float(u[slope_var]) * df_test[slope_var].to_numpy(dtype=float)
    return pred


# ═══════════════════════════════════════════════════════════════
# QUANTILE REGRESSION (not used by the notebooks)
# ═══════════════════════════════════════════════════════════════

def fit_quantile_regression(X_train: np.ndarray, y_train: np.ndarray, q: float = 0.50):
    """Linear quantile regression at quantile q. Returns QuantRegResults."""
    import statsmodels.api as sm
    return sm.QuantReg(y_train, sm.add_constant(X_train)).fit(q=q)


# ═══════════════════════════════════════════════════════════════
# NAIVE BASELINES (Phase 3, section 3.1)
# ═══════════════════════════════════════════════════════════════

# BaseEstimator supplies get_params/set_params/__sklearn_clone__ and the tags
# newer scikit-learn versions query (the hand-written get_params stubs worked
# with clone() but not with every sklearn utility).

class GrandMeanBaseline(RegressorMixin, BaseEstimator):
    """B0: predict overall mean of the transformed target."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class GroupMeanBaseline(RegressorMixin, BaseEstimator):
    """
    B1/B3: predict the group mean. Held-out groups fall back to the grand mean.

    CAUTION: under LOSO-River (group=River) or LOSO-Station (group=Station_ID)
    the held-out group is never in training, so this IS the grand mean in every
    fold. Use evaluation.evaluate_lookup_baseline(), which reports how each row
    was served, for Phase 3 baselines.

    Takes group labels as a SEPARATE argument rather than a column index into X,
    because X is a float array and river names are strings.
    """

    def fit(self, X, y, groups=None):
        if groups is None:
            raise ValueError("GroupMeanBaseline.fit requires `groups`.")
        groups = np.asarray(groups)
        y = np.asarray(y, dtype=float)
        self.grand_mean_ = float(y.mean())
        self.group_means_ = {g: float(y[groups == g].mean()) for g in np.unique(groups)}
        return self

    def predict(self, X, groups=None):
        if groups is None:
            return np.full(len(X), self.grand_mean_)
        return np.array([self.group_means_.get(g, self.grand_mean_)
                         for g in np.asarray(groups)], dtype=float)
