"""
Target transform contract and fold-safe feature preprocessing.

Two things live here:

1. The forward/inverse target transform, in ONE place. The raw CSV stores
   log_BOD = log1p(BOD), so the inverse is expm1, not exp. Using exp with a
   log1p target would shift every back-transformed concentration by +1 mg/L,
   silently turning a "BOD > 6" exceedance rule into "BOD > 5".

2. FoldSafePCA -- a PCA that is fitted inside a sklearn Pipeline, so it is
   refit on every training fold instead of once on the full dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import TARGET_TRANSFORM, PCA_BLOCKS


# ═══════════════════════════════════════════════════════════════
# TARGET TRANSFORM
# ═══════════════════════════════════════════════════════════════

def target_forward(y: np.ndarray) -> np.ndarray:
    """mg/L -> model space. Must match the raw CSV's log_BOD definition."""
    if TARGET_TRANSFORM == "log1p":
        return np.log1p(y)
    if TARGET_TRANSFORM == "log":
        return np.log(y)
    raise ValueError(f"Unknown TARGET_TRANSFORM={TARGET_TRANSFORM!r}")


def target_inverse(y_t: np.ndarray, clip_at: float | None = 0.0) -> np.ndarray:
    """
    Model space -> mg/L.

    clip_at : predictions below this are clipped. expm1 of a negative prediction
              gives a negative concentration, which is nonphysical and poisons
              MAE_mgL / R2_orig. Set to None to see the raw values.
    """
    if TARGET_TRANSFORM == "log1p":
        out = np.expm1(y_t)
    elif TARGET_TRANSFORM == "log":
        out = np.exp(y_t)
    else:
        raise ValueError(f"Unknown TARGET_TRANSFORM={TARGET_TRANSFORM!r}")
    return out if clip_at is None else np.clip(out, clip_at, None)


def target_inverse_smeared(y_t: np.ndarray, smear_factor, clip_at: float | None = 0.0) -> np.ndarray:
    """
    Model space -> mg/L, estimating the conditional MEAN (Duan smearing).

    target_inverse() of a log-scale prediction estimates the conditional
    median, which is biased low for the mean. With D = mean(exp(training
    residuals)) on the log scale:
        log1p target : E[Y] = D * exp(yhat) - 1
        log target   : E[Y] = D * exp(yhat)

    Use for mg/L error metrics. Threshold classification should stay on
    target_inverse(): P(Y > c) > 0.5 exactly when the median exceeds c.
    """
    y_t = np.asarray(y_t, dtype=float)
    D = np.asarray(smear_factor, dtype=float)
    if TARGET_TRANSFORM == "log1p":
        out = D * np.exp(y_t) - 1.0
    elif TARGET_TRANSFORM == "log":
        out = D * np.exp(y_t)
    else:
        raise ValueError(f"Unknown TARGET_TRANSFORM={TARGET_TRANSFORM!r}")
    return out if clip_at is None else np.clip(out, clip_at, None)


def assert_target_consistent(df: pd.DataFrame, raw_col: str, log_col: str,
                             tol: float = 1e-8) -> None:
    """
    Fail loudly if a stored log column disagrees with TARGET_TRANSFORM.
    Guards against an input file that defines the log columns differently.
    """
    ok = df[raw_col].notna() & df[log_col].notna()
    if not ok.any():
        return
    dev = float((df.loc[ok, log_col] - target_forward(df.loc[ok, raw_col].values)).abs().max())
    if dev > tol:
        raise AssertionError(
            f"{log_col} does not match TARGET_TRANSFORM={TARGET_TRANSFORM!r} "
            f"(max deviation {dev:.3g}). Either the CSV changed definition or "
            f"config.TARGET_TRANSFORM is wrong. Do not proceed."
        )


# ═══════════════════════════════════════════════════════════════
# FOLD-SAFE PCA
# ═══════════════════════════════════════════════════════════════

class FoldSafePCA(BaseEstimator, TransformerMixin):
    """
    Standardise -> PCA -> (optionally) sign-stabilise.

    Why this exists instead of a precomputed PC1_EC_TDS column:
    a PCA fitted on the full dataset has seen the test fold's covariance
    structure. Under LOSO-River that is a real leak -- the rotation is partly
    determined by the held-out river. Fitted inside a Pipeline, this is refit
    per fold and the test fold is only ever transformed.

    Sign stabilisation: PCA component signs are arbitrary and flip between
    folds. Without pinning them, averaged coefficients across folds are
    meaningless (a +0.8 in one fold cancels a -0.8 in the next). We force the
    loading on `sign_ref_idx` to be positive, so for [log_EC, log_TDS] the
    component always reads "higher = more dissolved ionic load".
    """

    def __init__(self, n_components: int = 1, sign_ref_idx: int = 0,
                 standardize: bool = True):
        self.n_components = n_components
        self.sign_ref_idx = sign_ref_idx
        self.standardize = standardize

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.scaler_ = StandardScaler().fit(X) if self.standardize else None
        Xs = self.scaler_.transform(X) if self.standardize else X

        self.pca_ = PCA(n_components=self.n_components).fit(Xs)
        comp = self.pca_.components_.copy()

        # pin the sign of every component against the reference feature
        self.signs_ = np.where(comp[:, self.sign_ref_idx] < 0, -1.0, 1.0)
        self.components_ = comp * self.signs_[:, None]
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Xs = self.scaler_.transform(X) if self.standardize else X
        return self.pca_.transform(Xs) * self.signs_[None, :]

    def get_feature_names_out(self, input_features=None):
        return np.array([f"PC{i + 1}" for i in range(self.n_components)])


def build_preprocessor(feature_cols: list[str],
                       pca_block: str | None = None,
                       scale_rest: bool = True) -> Pipeline:
    """
    Build the preprocessing half of a modelling Pipeline.

    pca_block : key into config.PCA_BLOCKS, e.g. "PC1_EC_TDS". The block's
                member columns are collapsed into components; every other
                column in `feature_cols` passes through (scaled if scale_rest).
                None -> plain StandardScaler on everything.

    Returns a Pipeline, so it can be nested inside GridSearchCV and refit per
    inner fold. Never call .fit() on it with the full dataset.
    """
    if pca_block is None:
        return Pipeline([("scaler", StandardScaler())])

    if pca_block not in PCA_BLOCKS:
        raise ValueError(f"Unknown PCA block {pca_block!r}. Have: {list(PCA_BLOCKS)}")

    spec = PCA_BLOCKS[pca_block]
    block_cols = spec["cols"]
    missing = [c for c in block_cols if c not in feature_cols]
    if missing:
        raise ValueError(
            f"PCA block {pca_block!r} needs {missing}, absent from feature_cols={feature_cols}"
        )

    block_idx = [feature_cols.index(c) for c in block_cols]
    rest_idx = [i for i in range(len(feature_cols)) if i not in block_idx]
    sign_ref = block_cols.index(spec.get("sign_ref", block_cols[0]))

    ct = ColumnTransformer(
        transformers=[
            ("pca_block", FoldSafePCA(n_components=spec.get("n_components", 1),
                                      sign_ref_idx=sign_ref), block_idx),
            ("rest", StandardScaler() if scale_rest else "passthrough", rest_idx),
        ],
        remainder="drop",
    )
    # The PCA scores are already on a comparable scale post-standardisation, but
    # a second scaler makes the ridge penalty isotropic across the two blocks.
    return Pipeline([("block", ct), ("scaler", StandardScaler())])


def pca_block_feature_names(feature_cols: list[str], pca_block: str) -> list[str]:
    """Column names produced by build_preprocessor, in order."""
    if pca_block is None:
        return list(feature_cols)
    spec = PCA_BLOCKS[pca_block]
    k = spec.get("n_components", 1)
    names = [f"{pca_block}" if k == 1 else f"{pca_block}_{i + 1}" for i in range(k)]
    return names + [c for c in feature_cols if c not in spec["cols"]]


def inspect_pca_block(df: pd.DataFrame, feature_cols: list[str],
                      pca_block: str = "PC1_EC_TDS") -> pd.DataFrame:
    """
    Diagnostic ONLY -- fits on whatever you hand it. Use on a training fold to
    report loadings and explained variance for the manuscript. Do not use its
    output as a feature.
    """
    spec = PCA_BLOCKS[pca_block]
    cols = spec["cols"]
    sub = df[cols].dropna()
    ref = cols.index(spec.get("sign_ref", cols[0]))
    fs = FoldSafePCA(n_components=spec.get("n_components", 1), sign_ref_idx=ref).fit(sub.values)
    return pd.DataFrame(
        fs.components_, columns=cols,
        index=[f"PC{i + 1} (EVR={r:.3f})" for i, r in enumerate(fs.explained_variance_ratio_)],
    ).round(4)
