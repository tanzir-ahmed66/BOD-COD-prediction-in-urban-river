"""
Model evaluation engine.

evaluate_model()   -- any sklearn-compatible estimator through any CV scheme.
evaluate_mixedlm() -- MixedLM, which is not sklearn-compatible and needs
                      different handling at every stage.
compute_metrics()  -- regression + threshold-exceedance metrics.

Back-transform note: metrics in mg/L go through preprocessing.target_inverse(),
which respects config.TARGET_TRANSFORM. Never apply np.exp to the log1p target:
it inflates every back-transformed concentration by +1 mg/L and turns the
"BOD > 6" exceedance rule into "BOD > 5".
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix,
)

from .config import PRIMARY_THRESHOLD, THRESHOLDS, EXCEEDANCE_RULE, EXCEEDANCE_RULES
from .preprocessing import target_inverse, target_inverse_smeared

Fold = Tuple[np.ndarray, np.ndarray, str]

_NAN_CLS = ["sensitivity", "specificity", "F1", "kappa", "AUC", "NPV"]


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

def exceeds(values, threshold: float, rule: Optional[str] = None,
            rtol: float = 1e-9, atol: float = 1e-9) -> np.ndarray:
    """
    Boolean exceedance of a mg/L limit under `rule`.
    ">"  : value > limit (default, config.EXCEEDANCE_RULE)
    ">=" : value >= limit (inclusive reading; sensitivity for values reported at the limit)

    Values within floating-point tolerance of the limit count as AT the limit.
    Concentrations reach this function after a log1p/expm1 round trip, which is
    not exact: 7 COD values reported as 100 mg/L come back as 100.00000000000001
    (counted as exceedances under a bare ">"), and 44 BOD values of 6 mg/L come
    back as 5.999999999999999 (missed under a bare ">=").
    """
    rule = EXCEEDANCE_RULE if rule is None else rule
    if rule not in EXCEEDANCE_RULES:
        raise ValueError(f"rule must be one of {EXCEEDANCE_RULES}, got {rule!r}")
    v = np.asarray(values, dtype=float)
    at = np.isclose(v, threshold, rtol=rtol, atol=atol)
    return (v >= threshold) | at if rule == ">=" else (v > threshold) & ~at


def compute_metrics(y_true_t: np.ndarray, y_pred_t: np.ndarray,
                    threshold: float = None, rule: Optional[str] = None) -> Dict[str, float]:
    """
    y_true_t / y_pred_t : values in TRANSFORMED space (log1p by default).
    threshold           : mg/L cut-off; applied after back-transform.
    rule                : ">" (default, config.EXCEEDANCE_RULE) or ">="; see exceeds().
    """
    y_true_t = np.asarray(y_true_t, dtype=float)
    y_pred_t = np.asarray(y_pred_t, dtype=float)

    y_true_orig = target_inverse(y_true_t)
    y_pred_orig = target_inverse(y_pred_t)

    m = {
        "R2_log":   r2_score(y_true_t, y_pred_t),
        "MAE_log":  mean_absolute_error(y_true_t, y_pred_t),
        "RMSE_log": np.sqrt(mean_squared_error(y_true_t, y_pred_t)),
        "R2_orig":  r2_score(y_true_orig, y_pred_orig),
        "MAE_mgL":  mean_absolute_error(y_true_orig, y_pred_orig),
        "RMSE_mgL": np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)),
        "n":        len(y_true_t),
    }

    if threshold is None:
        return m

    y_true_bin = exceeds(y_true_orig, threshold, rule).astype(int)
    y_pred_bin = exceeds(y_pred_orig, threshold, rule).astype(int)

    if len(np.unique(y_true_bin)) < 2:
        # A fold where nothing (or everything) exceeds the threshold. Reporting
        # 0 or 1 here would be a fabricated number; NaN drops out of the mean.
        for k in _NAN_CLS:
            m[k] = np.nan
        m["exceedance_rate"] = float(y_true_bin.mean())
        return m

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    m["sensitivity"] = tp / (tp + fn) if (tp + fn) else np.nan
    m["specificity"] = tn / (tn + fp) if (tn + fp) else np.nan
    m["NPV"] = tn / (tn + fn) if (tn + fn) else np.nan
    m["F1"] = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    m["kappa"] = cohen_kappa_score(y_true_bin, y_pred_bin)
    # AUC ranks on the continuous prediction. Use the transformed value: it is a
    # monotone function of mg/L, but unlike y_pred_orig it is not clipped at 0,
    # so clipped predictions do not become artificial ties.
    m["AUC"] = roc_auc_score(y_true_bin, y_pred_t)
    m["exceedance_rate"] = float(y_true_bin.mean())
    return m


# ═══════════════════════════════════════════════════════════════
# SKLEARN MODELS
# ═══════════════════════════════════════════════════════════════

def _accepts_groups(estimator) -> bool:
    """
    True if fit() should be given groups= -- i.e. this is a search object whose
    cv is a grouped or forward-chaining splitter rather than an int/None.

    Duck-typed rather than isinstance(BaseSearchCV): that class only lives at a
    private import path, which breaks across sklearn versions.
    """
    if not (hasattr(estimator, "cv") and hasattr(estimator, "estimator")):
        return False
    return _cv_requires_groups(getattr(estimator, "cv", None))


def _cv_requires_groups(cv) -> bool:
    """
    True for splitters whose split() consumes `groups`. KFold and ShuffleSplit
    accept the argument and ignore it, so 'has a split method' is too loose --
    it would demand a group column for the deliberately-unstructured protocols.
    """
    if cv is None or isinstance(cv, (int, np.integer)) or not hasattr(cv, "split"):
        return False
    if getattr(cv, "requires_groups", False):
        return True
    name = type(cv).__name__
    return name.startswith(("Group", "StratifiedGroup", "LeaveOneGroup", "LeavePGroups"))


def evaluate_model_oof(model, df: pd.DataFrame, feature_cols: List[str], target_col: str,
                       folds: List[Fold], threshold: float = None,
                       inner_group_col: Optional[str] = None,
                       fit_fn: Optional[Callable] = None,
                       predict_fn: Optional[Callable] = None,
                       scale: bool = False,
                       smearing: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a model through a CV scheme; return (per-fold metrics, out-of-fold predictions).

    inner_group_col : column whose values are passed as `groups=` to the inner
                      search (e.g. "River" for LOSO-River, "Year" for
                      forward-chain). Required whenever `model` is a
                      GridSearchCV built with a grouped/forward-chaining cv.
    smearing        : also record the fold's Duan smearing factor,
                      D = mean(exp(training residual)) on the transformed scale.
                      In-sample residuals understate spread for flexible models
                      (trees), so D is biased toward 1 for them.

    scale : default False; leave it off. Scaling belongs inside the pipeline so
            it is refit within each inner fold. Scaling here happens once on the
            outer training set and defeats the point of nesting.

    The out-of-fold frame has one row per test observation:
    row (df index label), fold, y_true, y_pred, smear_factor.
    """
    if scale:
        warnings.warn(
            "evaluate_model(scale=True) rescales on the OUTER training set, "
            "outside the inner CV. Put StandardScaler in the pipeline instead "
            "(models.make_pipeline does this).", UserWarning, stacklevel=2,
        )

    needs_groups = _accepts_groups(model) if fit_fn is None else False
    if needs_groups and inner_group_col is None:
        raise ValueError(
            "This estimator's inner CV is grouped or forward-chaining, so "
            "evaluate_model needs inner_group_col (e.g. 'River' or 'Year'). "
            "Passing no groups would silently fall back to an unstructured split."
        )

    results, oof = [], []
    for train_idx, test_idx, fold_label in folds:
        X_train = df.loc[train_idx, feature_cols].to_numpy(dtype=float)
        X_test = df.loc[test_idx, feature_cols].to_numpy(dtype=float)
        y_train = df.loc[train_idx, target_col].to_numpy(dtype=float)
        y_test = df.loc[test_idx, target_col].to_numpy(dtype=float)

        if scale:
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(X_train)
            X_train, X_test = sc.transform(X_train), sc.transform(X_test)

        if fit_fn is not None:
            fitted = fit_fn(X_train, y_train)
        else:
            fitted = clone(model)
            if needs_groups:
                g = df.loc[train_idx, inner_group_col].to_numpy()
                fitted.fit(X_train, y_train, groups=g)
            else:
                fitted.fit(X_train, y_train)

        predict = (lambda X: predict_fn(fitted, X)) if predict_fn else fitted.predict
        y_pred = np.asarray(predict(X_test), dtype=float)
        smear = (float(np.mean(np.exp(y_train - np.asarray(predict(X_train), dtype=float))))
                 if smearing else np.nan)

        m = compute_metrics(y_test, y_pred, threshold=threshold)
        m["fold"] = fold_label
        m["n_train"] = len(train_idx)
        m["smear_factor"] = smear
        if hasattr(fitted, "best_params_"):
            m["best_params"] = str(fitted.best_params_)       # CSV-friendly
            m["best_params_dict"] = dict(fitted.best_params_)  # for edge audits
        results.append(m)
        oof.append(pd.DataFrame({"row": np.asarray(test_idx), "fold": fold_label,
                                 "y_true": y_test, "y_pred": y_pred, "smear_factor": smear}))

    return pd.DataFrame(results), pd.concat(oof, ignore_index=True)


def evaluate_model(model, df: pd.DataFrame, feature_cols: List[str], target_col: str,
                   folds: List[Fold], threshold: float = None,
                   inner_group_col: Optional[str] = None,
                   fit_fn: Optional[Callable] = None,
                   predict_fn: Optional[Callable] = None,
                   scale: bool = False) -> pd.DataFrame:
    """
    Per-fold metrics only. Same computation as evaluate_model_oof(), which also
    returns the out-of-fold predictions needed for pooled metrics.
    """
    fold_df, _ = evaluate_model_oof(model, df, feature_cols, target_col, folds,
                                    threshold=threshold, inner_group_col=inner_group_col,
                                    fit_fn=fit_fn, predict_fn=predict_fn, scale=scale)
    return fold_df


# ═══════════════════════════════════════════════════════════════
# NAIVE LOOKUP BASELINES
# ═══════════════════════════════════════════════════════════════

def evaluate_lookup_baseline(df: pd.DataFrame, folds: List[Fold], target_col: str,
                             levels: List[List[str]], threshold: float = None
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hierarchical lookup of training-fold means on the transformed scale.

    levels : most specific first, e.g. [["River", "Month_num"], ["Month_num"], []];
             [] is the grand mean. Each test row is served by the first level whose
             key exists in the training fold.

    Returns (per-fold metrics, out-of-fold predictions). Per fold, `pct_first_level`
    and `pct_grand_mean` record how rows were served, so a baseline that silently
    degenerates to the grand mean (e.g. river mean under LOSO-River) is visible.
    The OOF frame carries `served_by` per row.
    """
    names = [" × ".join(lev) if lev else "grand mean" for lev in levels]
    fold_rows, oof = [], []
    for train_idx, test_idx, fold_label in folds:
        trd, ted = df.loc[train_idx], df.loc[test_idx]
        pred = np.full(len(ted), np.nan)
        served = np.full(len(ted), "", dtype=object)
        for lev, name in zip(levels, names):
            todo = np.isnan(pred)
            if not todo.any():
                break
            if lev:
                means = trd.groupby(lev[0] if len(lev) == 1 else lev)[target_col].mean()
                idx = (pd.Index(ted[lev[0]]) if len(lev) == 1
                       else pd.MultiIndex.from_frame(ted[lev]))
                vals = means.reindex(idx).to_numpy(dtype=float)
            else:
                vals = np.full(len(ted), float(trd[target_col].mean()))
            fill = todo & ~np.isnan(vals)
            pred[fill], served[fill] = vals[fill], name
        if np.isnan(pred).any():
            raise ValueError("Lookup left rows unserved; end `levels` with [] (grand mean).")
        y_test = ted[target_col].to_numpy(dtype=float)
        m = compute_metrics(y_test, pred, threshold=threshold)
        m.update(fold=fold_label, n_train=len(train_idx), smear_factor=np.nan,
                 pct_first_level=float(np.mean(served == names[0])),
                 pct_grand_mean=float(np.mean(served == "grand mean")))
        fold_rows.append(m)
        oof.append(pd.DataFrame({"row": np.asarray(test_idx), "fold": fold_label,
                                 "y_true": y_test, "y_pred": pred,
                                 "smear_factor": np.nan, "served_by": served}))
    return pd.DataFrame(fold_rows), pd.concat(oof, ignore_index=True)


# ═══════════════════════════════════════════════════════════════
# MIXEDLM EVALUATOR
# ═══════════════════════════════════════════════════════════════

def evaluate_mixedlm(df: pd.DataFrame, feature_cols: List[str], folds: List[Fold],
                     target_col: str = "log_BOD", group_col: str = "River",
                     re_formula: Optional[str] = None,
                     slope_var: Optional[str] = None,
                     mode: str = "auto", threshold: float = None,
                     reml: bool = True, standardize: bool = True,
                     verbose: bool = False) -> pd.DataFrame:
    """
    Cross-validate a MixedLM. Separate from evaluate_model because MixedLM
    differs at every stage:

      - it needs a DataFrame and a formula, not an (X, y) array pair;
      - prediction depends on whether the test fold's group was in training
        (fixed effects only vs fixed + BLUP) -- a distinction sklearn has no
        concept of, and the single biggest source of accidentally optimistic
        hierarchical results;
      - it can fail to converge, and a crashed fold must be recorded rather
        than aborting the whole protocol.

    mode : "auto"   -- FE-only for held-out groups, FE+BLUP for seen ones.
                       Correct for a fold set that mixes both (e.g. forward-chain).
           "new"    -- force FE-only. The honest LOSO-River deployment scenario.
           "known"  -- force FE+BLUP. Upper bound; only valid when the group
                       really is in training.

    Scaling is fitted on the TRAINING fold only and applied to the test fold.

    Returns one row per fold, plus `converged`, `mode_used`, `group_seen`,
    and the fitted variance components.
    """
    from .models import (fit_mixedlm_river, predict_mixedlm_new_group,
                         predict_mixedlm_known_group)

    if mode not in {"auto", "new", "known"}:
        raise ValueError(f"mode must be auto|new|known, got {mode!r}")
    if re_formula is None and slope_var is not None:
        re_formula = f"~{slope_var}"
    if re_formula is not None and slope_var is None:
        slope_var = re_formula.lstrip("~ ").strip() or None

    rows = []
    for train_idx, test_idx, fold_label in folds:
        tr = df.loc[train_idx, feature_cols + [target_col, group_col]].dropna().copy()
        te = df.loc[test_idx, feature_cols + [target_col, group_col]].dropna().copy()

        rec = {"fold": fold_label, "n_train": len(tr), "n": len(te)}

        if len(te) == 0 or len(tr) == 0:
            rec.update({"converged": False, "fail_reason": "empty fold"})
            rows.append(rec)
            continue

        # ---- fold-safe standardisation -----------------------------------
        if standardize:
            mu = tr[feature_cols].mean()
            sd = tr[feature_cols].std(ddof=0).replace(0.0, 1.0)
            tr[feature_cols] = (tr[feature_cols] - mu) / sd
            te[feature_cols] = (te[feature_cols] - mu) / sd

        # ---- which prediction path? --------------------------------------
        test_groups = te[group_col].unique()
        train_groups = set(tr[group_col].unique())
        seen = [g for g in test_groups if g in train_groups]
        all_seen = len(seen) == len(test_groups)

        if mode == "new":
            use_known = False
        elif mode == "known":
            if not all_seen:
                raise ValueError(
                    f"Fold {fold_label!r}: mode='known' but "
                    f"{set(test_groups) - train_groups} are absent from training. "
                    "Use mode='auto'."
                )
            use_known = True
        else:
            use_known = all_seen

        rec["group_seen"] = all_seen
        rec["mode_used"] = "known" if use_known else "new"

        # ---- fit ----------------------------------------------------------
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result, _ = fit_mixedlm_river(
                    tr, feature_cols, target_col=target_col,
                    group_col=group_col, reml=reml, re_formula=re_formula,
                )
            converged = bool(getattr(result, "converged", True))
        except Exception as exc:
            rec.update({"converged": False, "fail_reason": f"{type(exc).__name__}: {exc}"})
            for k in ["R2_log", "MAE_log", "RMSE_log", "R2_orig", "MAE_mgL", "RMSE_mgL"]:
                rec[k] = np.nan
            rows.append(rec)
            if verbose:
                print(f"  {fold_label}: FIT FAILED — {exc}")
            continue

        # ---- predict ------------------------------------------------------
        if use_known:
            preds = np.empty(len(te), dtype=float)
            for g in test_groups:
                m = (te[group_col] == g).to_numpy()
                preds[m] = predict_mixedlm_known_group(
                    result, te.loc[m], feature_cols, g, slope_var=slope_var
                )
            y_pred = preds
        else:
            y_pred = predict_mixedlm_new_group(result, te, feature_cols)

        y_true = te[target_col].to_numpy(dtype=float)
        rec.update(compute_metrics(y_true, y_pred, threshold=threshold))
        rec["converged"] = converged

        # variance components: how much of the signal is river-level?
        try:
            sigma2_resid = float(result.scale)
            sigma2_group = float(np.asarray(result.cov_re)[0, 0])
            rec["sigma2_group"] = round(sigma2_group, 5)
            rec["sigma2_resid"] = round(sigma2_resid, 5)
            denom = sigma2_group + sigma2_resid
            rec["icc_fitted"] = round(sigma2_group / denom, 4) if denom > 0 else np.nan
        except Exception:
            rec["sigma2_group"] = rec["sigma2_resid"] = rec["icc_fitted"] = np.nan

        rows.append(rec)
        if verbose:
            print(f"  {fold_label}: R2_log={rec.get('R2_log', np.nan):.3f} "
                  f"mode={rec['mode_used']} converged={converged}")

    out = pd.DataFrame(rows)
    if "converged" in out and not out["converged"].all():
        bad = out.loc[~out["converged"].fillna(False), "fold"].tolist()
        warnings.warn(f"MixedLM did not converge on folds {bad}; their metrics are NaN.",
                      UserWarning, stacklevel=2)
    return out


# ═══════════════════════════════════════════════════════════════
# SUMMARIES
# ═══════════════════════════════════════════════════════════════

def summarize_cv(results_df: pd.DataFrame, label: str = "",
                 weight_by_n: bool = False) -> pd.Series:
    """
    Mean ± SD across folds.

    weight_by_n : LOSO-Station folds range widely in size; an unweighted mean
                  lets a 12-row station count as much as a 60-row one. Set True
                  for a pooled-observation view. Report which you used.
    """
    summary = pd.Series(dtype=object)
    summary["label"] = label
    summary["n_folds"] = len(results_df)
    if "converged" in results_df:
        summary["n_converged"] = int(results_df["converged"].fillna(False).sum())

    w = results_df["n"] if (weight_by_n and "n" in results_df) else None

    for col in ["R2_log", "MAE_log", "RMSE_log", "R2_orig", "MAE_mgL", "RMSE_mgL"]:
        if col not in results_df:
            continue
        v = pd.to_numeric(results_df[col], errors="coerce")
        ok = v.notna()
        if not ok.any():
            summary[f"{col}_mean"] = summary[f"{col}_sd"] = np.nan
            continue
        summary[f"{col}_mean"] = (np.average(v[ok], weights=w[ok]) if w is not None
                                  else v[ok].mean())
        summary[f"{col}_sd"] = v[ok].std()

    for col in ["sensitivity", "specificity", "F1", "kappa", "AUC", "NPV"]:
        if col in results_df:
            v = pd.to_numeric(results_df[col], errors="coerce")
            summary[f"{col}_mean"] = v.mean()
            summary[f"{col}_n"] = int(v.notna().sum())   # folds that had both classes
    return summary


def pooled_threshold_metrics(pooled: Dict[str, float]) -> Dict[str, float]:
    """
    Balanced accuracy, MCC and the always-exceed F1 reference, derived from the
    rates compute_metrics() returns on pooled predictions (same exceedance rule).
    MCC is undefined (NaN) when a predicted or observed class is empty; it is
    never reported as 0.0, which would be a fabricated value.
    F1_always_exceed = 2p/(1+p) is what predicting "exceeds" for every row
    scores; with ~70% prevalence it is already high, so report it beside F1.
    """
    sens, spec, p = pooled.get("sensitivity"), pooled.get("specificity"), pooled.get("exceedance_rate")
    out = {"prevalence": p, "sensitivity_pooled": sens, "specificity_pooled": spec,
           "kappa_pooled": pooled.get("kappa"), "F1_pooled": pooled.get("F1"),
           "AUC_pooled": pooled.get("AUC")}
    if any(v is None for v in (sens, spec, p)) or not np.isfinite([sens, spec, p]).all():
        return out
    tp, fn, tn, fp = sens * p, (1 - sens) * p, spec * (1 - p), (1 - spec) * (1 - p)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    out.update(balanced_accuracy_pooled=(sens + spec) / 2,
               MCC_pooled=float((tp * tn - fp * fn) / denom) if denom > 0 else np.nan,
               F1_always_exceed=2 * p / (1 + p))
    return out


def smeared_metrics(oof: pd.DataFrame) -> Dict[str, float]:
    """
    Pooled mg/L bias and, when smearing factors are present, smeared MAE/RMSE/R2.
    Both back-transforms are clipped at 0 mg/L, as in compute_metrics().
    """
    y_true = target_inverse(oof["y_true"].to_numpy(dtype=float))
    y_hat_t = oof["y_pred"].to_numpy(dtype=float)
    out = {"bias_mgL_pooled": float(np.mean(target_inverse(y_hat_t) - y_true))}
    D = oof["smear_factor"].to_numpy(dtype=float)
    if np.isfinite(D).all():
        sm = target_inverse_smeared(y_hat_t, D)
        sst = float(np.sum((y_true - y_true.mean()) ** 2))
        out.update(
            MAE_mgL_pooled_smear=float(np.mean(np.abs(sm - y_true))),
            RMSE_mgL_pooled_smear=float(np.sqrt(np.mean((sm - y_true) ** 2))),
            R2_orig_pooled_smear=float(1 - np.sum((y_true - sm) ** 2) / sst) if sst > 0 else np.nan,
            bias_mgL_pooled_smear=float(np.mean(sm - y_true)),
            smear_factor_mean=float(np.mean(D)),
        )
    return out


def summarize_with_pooled(fold_df: pd.DataFrame, oof: pd.DataFrame, label: str = "",
                          threshold: float = None, rule: Optional[str] = None) -> Dict[str, float]:
    """
    summarize_cv() (unweighted fold means — the pre-specified metric) plus:
      R2_log_wmean        per-fold R2 weighted by fold size
      *_pooled            computed once over all out-of-fold predictions; pooled
                          MAE equals the row-weighted fold MAE; pooled RMSE equals
                          sqrt(row-weighted fold MSE), not the weighted fold RMSE
      smeared mg/L metrics and bias (see smeared_metrics)
      threshold extras    when `threshold` is given (see pooled_threshold_metrics)
    """
    s = summarize_cv(fold_df, label=label).to_dict()
    w = fold_df["n"] / fold_df["n"].sum()
    s["R2_log_wmean"] = float((pd.to_numeric(fold_df["R2_log"]) * w).sum())
    pooled = compute_metrics(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy(),
                             threshold=threshold, rule=rule)
    for k in ["R2_log", "R2_orig", "MAE_mgL", "RMSE_mgL"]:
        s[f"{k}_pooled"] = pooled[k]
    s.update(smeared_metrics(oof))
    if threshold is not None:
        s.update(pooled_threshold_metrics(pooled))
    return s


def skill_vs_reference(oof_model: pd.DataFrame, oof_ref: pd.DataFrame,
                       by: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Skill relative to a reference forecast, on the transformed (log) scale:

        SS = 1 − MSE(model) / MSE(reference)

    With a month-of-year climatology as the reference this is the seasonal
    skill score used throughout the pipeline (the hydrological-forecasting
    convention of scoring against climatology rather than a grand mean).
    SS = 0: no better than the reference; SS < 0: worse; SS = 1: perfect.

    Rows are matched on (row, fold); only rows present in both frames count.
    by : optional columns present in BOTH frames (e.g. ["fold"] or ["River"]).
    Returns one row per group (or a single row) with SS, both MSEs and n.
    """
    keys = ["row", "fold"] + [c for c in (by or []) if c not in ("row", "fold")]
    a = oof_model[keys + ["y_true", "y_pred"]].rename(columns={"y_pred": "pred_model"})
    b = oof_ref[["row", "fold", "y_pred"]].rename(columns={"y_pred": "pred_ref"})
    m = a.merge(b, on=["row", "fold"], how="inner")
    m["se_model"] = (m["y_true"] - m["pred_model"]) ** 2
    m["se_ref"] = (m["y_true"] - m["pred_ref"]) ** 2

    def _one(d: pd.DataFrame) -> pd.Series:
        mse_m, mse_r = float(d["se_model"].mean()), float(d["se_ref"].mean())
        return pd.Series({"n": len(d), "mse_model": mse_m, "mse_ref": mse_r,
                          "skill": 1 - mse_m / mse_r if mse_r > 0 else np.nan})

    if not by:
        return _one(m).to_frame().T
    return pd.DataFrame([{**dict(zip(by, k if isinstance(k, tuple) else (k,))), **_one(g)}
                         for k, g in m.groupby(by, sort=True)])


def protocol_comparison(estimator_factory: Callable, df: pd.DataFrame,
                        feature_cols: List[str], target_col: str,
                        threshold: float = None) -> pd.DataFrame:
    """
    The SAME model family under 4 validation protocols, each with
    its OWN matching inner CV.

    estimator_factory : callable(feature_cols, inner_cv) -> estimator.
                        e.g. lambda fc, cv: fit_ridge_cv(fc, cv)
                        A factory, not an instance, because the inner splitter
                        differs per protocol -- reusing one estimator would tune
                        every protocol against the same inner scheme.
    """
    from .folds import (build_loso_river, build_random_splits,
                        build_stratified_kfold, build_fc, inner_cv_for_folds)

    protocols = {
        "Random 80/20 (x20)": (build_random_splits(df), "random"),
        "Stratified 5-fold":  (build_stratified_kfold(df), "stratified_5f"),
        "LOSO-River":         (build_loso_river(df), "loso_river"),
        "Forward-Chain":      (build_fc(df), "forward_chain"),
    }

    rows = []
    for name, (folds, outer_scheme) in protocols.items():
        inner_cv, group_col = inner_cv_for_folds(df, folds, outer_scheme)
        est = estimator_factory(feature_cols, inner_cv)
        res = evaluate_model(est, df, feature_cols, target_col, folds,
                             threshold=threshold, inner_group_col=group_col)
        rows.append(summarize_cv(res, label=name))

    return pd.DataFrame(rows).set_index("label")
