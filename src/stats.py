"""
Statistical analysis functions used across EDA and robustness phases.
Each function is self-contained — no hidden state.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def true_icc(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
) -> Dict[str, float]:
    """
    Compute intraclass correlation coefficient (ICC) for a grouping variable.
    Uses one-way random effects ANOVA decomposition.

    ICC_river, ICC_station (Phase 1, sections 1.3 and 1.5).

    Returns
    -------
    dict with keys: icc, sigma2_between, sigma2_within, deff, n_eff
    """
    d = df[[target_col, group_col]].dropna()
    groups = d[group_col]
    y = d[target_col].to_numpy(dtype=float)
    if groups.nunique() < 2:
        raise ValueError(f"true_icc needs >= 2 groups in {group_col!r}, got {groups.nunique()}.")

    # Group statistics
    group_labels = groups.unique()
    k = len(group_labels)
    N = len(y)
    grand_mean = y.mean()

    # Between-group and within-group sums of squares
    ss_between = 0.0
    ss_within = 0.0
    n_per_group = []

    for g in group_labels:
        mask = (groups == g).to_numpy()
        y_g = y[mask]
        n_g = len(y_g)
        n_per_group.append(n_g)
        group_mean = y_g.mean()
        ss_between += n_g * (group_mean - grand_mean) ** 2
        ss_within += ((y_g - group_mean) ** 2).sum()

    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (N - k)

    # Average group size (harmonic mean for unbalanced data)
    n_per = np.array(n_per_group)
    n0 = (N - (n_per ** 2).sum() / N) / (k - 1)

    # Variance components
    sigma2_within = ms_within
    sigma2_between = max(0, (ms_between - ms_within) / n0)

    icc = sigma2_between / (sigma2_between + sigma2_within) if (sigma2_between + sigma2_within) > 0 else 0.0

    # Design effect and effective n
    m_bar = N / k
    deff = 1 + (m_bar - 1) * icc
    n_eff = N / deff

    return {
        "icc": round(icc, 4),
        "sigma2_between": round(sigma2_between, 4),
        "sigma2_within": round(sigma2_within, 4),
        "n_groups": k,
        "n_total": N,
        "mean_group_size": round(m_bar, 1),
        "deff": round(deff, 2),
        "n_eff": int(round(n_eff)),
    }


def nested_variance_decomposition(
    df: pd.DataFrame,
    target_col: str,
    level1_col: str = "River",
    level2_col: str = "Station_ID",
) -> Dict[str, float]:
    """
    Nested ANOVA variance decomposition (Phase 1, section 1.3).
    Station nested within River.

    Returns σ²_river, σ²_station(river), σ²_residual, and % each.
    """
    from scipy import stats as sp_stats

    y = df[target_col].values
    grand_mean = y.mean()
    N = len(y)

    # Level 1: River
    rivers = df[level1_col].unique()
    k1 = len(rivers)

    river_means = df.groupby(level1_col)[target_col].mean()
    river_sizes = df.groupby(level1_col)[target_col].count()

    ss_river = sum(
        river_sizes[r] * (river_means[r] - grand_mean) ** 2 for r in rivers
    )

    # Level 2: Station within river
    stations = df[level2_col].unique()
    k2 = len(stations)

    station_means = df.groupby(level2_col)[target_col].mean()
    station_sizes = df.groupby(level2_col)[target_col].count()
    station_river = df.groupby(level2_col)[level1_col].first()

    ss_station_within = sum(
        station_sizes[s] * (station_means[s] - river_means[station_river[s]]) ** 2
        for s in stations
    )

    # Residual
    ss_resid = sum(
        ((df.loc[df[level2_col] == s, target_col] - station_means[s]) ** 2).sum()
        for s in stations
    )

    ss_total = ((y - grand_mean) ** 2).sum()

    pct_river = 100 * ss_river / ss_total
    pct_station = 100 * ss_station_within / ss_total
    pct_resid = 100 * ss_resid / ss_total

    return {
        "SS_river": round(ss_river, 3),
        "SS_station_within_river": round(ss_station_within, 3),
        "SS_residual": round(ss_resid, 3),
        "SS_total": round(ss_total, 3),
        "pct_river": round(pct_river, 1),
        "pct_station": round(pct_station, 1),
        "pct_residual": round(pct_resid, 1),
        "eta2_river": round(ss_river / ss_total, 4),
    }


def partial_corr(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    control_cols: list,
) -> Dict[str, float]:
    """
    Partial correlation between x and y, controlling for a set of variables.
    "Unique contribution after collinearity" (Phase 1, section 1.2).

    Uses OLS residualization.
    """
    from sklearn.linear_model import LinearRegression

    sub = df[[x_col, y_col] + control_cols].dropna()

    X_controls = sub[control_cols].values

    # Residualize x
    reg_x = LinearRegression().fit(X_controls, sub[x_col].values)
    resid_x = sub[x_col].values - reg_x.predict(X_controls)

    # Residualize y
    reg_y = LinearRegression().fit(X_controls, sub[y_col].values)
    resid_y = sub[y_col].values - reg_y.predict(X_controls)

    r = np.corrcoef(resid_x, resid_y)[0, 1]

    return {
        "partial_r": round(r, 4),
        "partial_r2": round(r ** 2, 4),
        "x": x_col,
        "y": y_col,
        "controls": control_cols,
        "n": len(sub),
    }


def chow_test(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    break_col: str = "Year",
    break_value: int = 2020,
) -> Dict[str, float]:
    """
    Chow test for a structural break at a given year (Phase 1, section 1.4).
    Tests whether the regression coefficients differ before vs after the break.

    Returns F-statistic, p-value, and degrees of freedom.
    """
    from scipy.stats import f as f_dist
    from sklearn.linear_model import LinearRegression

    sub = df[feature_cols + [target_col, break_col]].dropna()

    X_all = sub[feature_cols].values
    y_all = sub[target_col].values

    mask_before = sub[break_col] < break_value
    mask_after = sub[break_col] >= break_value

    X1, y1 = X_all[mask_before], y_all[mask_before]
    X2, y2 = X_all[mask_after], y_all[mask_after]

    n1, n2 = len(y1), len(y2)
    k = X_all.shape[1] + 1  # features + intercept

    # Pooled RSS
    reg_all = LinearRegression().fit(X_all, y_all)
    rss_pooled = np.sum((y_all - reg_all.predict(X_all)) ** 2)

    # Separate RSS
    reg1 = LinearRegression().fit(X1, y1)
    rss1 = np.sum((y1 - reg1.predict(X1)) ** 2)

    reg2 = LinearRegression().fit(X2, y2)
    rss2 = np.sum((y2 - reg2.predict(X2)) ** 2)

    rss_separate = rss1 + rss2

    # F-statistic. NB: assumes independent rows. Observations share station and
    # sampling-campaign effects, so this p-value is anti-conservative; see
    # clustered_break_test() for a cluster-robust version.
    F = ((rss_pooled - rss_separate) / k) / (rss_separate / (n1 + n2 - 2 * k))
    # sf() rather than 1 - cdf(): the latter underflows to exactly 0 for large F.
    p_value = float(f_dist.sf(F, k, n1 + n2 - 2 * k))

    return {
        "F_stat": round(F, 3),
        "p_value": p_value,
        "df_num": k,
        "df_den": n1 + n2 - 2 * k,
        "n_before": n1,
        "n_after": n2,
        "break_at": break_value,
    }


def _cluster_cov(X: np.ndarray, resid: np.ndarray, codes: np.ndarray,
                 bread: np.ndarray) -> np.ndarray:
    """One-way cluster-robust (CR1) sandwich, same small-sample factor as statsmodels."""
    n, k = X.shape
    G = int(codes.max()) + 1
    scores = np.zeros((G, k))
    np.add.at(scores, codes, X * resid[:, None])
    meat = scores.T @ scores
    return (G / (G - 1.0)) * ((n - 1.0) / (n - k)) * (bread @ meat @ bread)


def clustered_break_test(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    break_col: str = "Year",
    break_value: int = 2020,
    cluster_cols: tuple = ("Station_ID", "Campaign"),
    label: str = "",
) -> Dict[str, float]:
    """
    Cluster-robust joint Wald F-test of a post-break shift in the
    intercept and every slope, with covariance clustered by `cluster_cols`
    (two-way, Cameron–Gelbach–Miller, when two columns are given — e.g. station
    and year-month campaign).

    Same null as chow_test(), but valid when rows share station and campaign
    effects. Denominator df = (number of clusters − 1) of the smaller clustering.

    numpy/scipy only — no statsmodels dependency. Numerically identical to
    statsmodels OLS(...).fit(cov_type="cluster", use_t=True).wald_test(use_f=True).
    """
    from scipy.stats import f as f_dist

    cluster_cols = list(cluster_cols)
    if len(cluster_cols) not in (1, 2):
        raise ValueError("cluster_cols must name one or two columns.")
    d = df.dropna(subset=list(feature_cols) + [target_col, break_col] + cluster_cols)

    post = (d[break_col] >= break_value).to_numpy(dtype=float)
    X0 = d[list(feature_cols)].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(d)), X0, post, X0 * post[:, None]])
    y = d[target_col].to_numpy(dtype=float)
    k0 = len(feature_cols)
    tested = np.arange(k0 + 1, 2 * k0 + 2)      # post + every feature×post term

    pinv_X = np.linalg.pinv(X)
    beta = pinv_X @ y
    resid = y - X @ beta
    bread = pinv_X @ pinv_X.T                    # (X'X)^-1

    codes = [pd.factorize(d[c])[0] for c in cluster_cols]
    n_groups = [int(c.max()) + 1 for c in codes]
    V = _cluster_cov(X, resid, codes[0], bread)
    if len(codes) == 2:
        both = pd.factorize(pd.Series(list(zip(codes[0], codes[1]))))[0]
        V = V + _cluster_cov(X, resid, codes[1], bread) - _cluster_cov(X, resid, both, bread)

    rb = beta[tested]
    J = len(tested)
    F = float(rb @ np.linalg.pinv(V[np.ix_(tested, tested)]) @ rb / J)
    df_denom = min(n_groups) - 1
    out = {
        "test": label or f"Clustered Wald ({' + '.join(cluster_cols)})",
        "n": int(len(d)),
        "clusters": " + ".join(cluster_cols),
        "F": F,
        "df_num": float(J),
        "df_denom": float(df_denom),
        "p_value": float(f_dist.sf(F, J, df_denom)),
        "post_intercept_shift": float(beta[k0 + 1]),
        "break_at": break_value,
    }
    for c, g in zip(cluster_cols, n_groups):
        out[f"n_{c}"] = g
    return out


def vif(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Variance Inflation Factor for each feature.
    VIF > 10 signals problematic collinearity.
    """
    from sklearn.linear_model import LinearRegression

    X = df[feature_cols].dropna().values
    vifs = []
    for i in range(X.shape[1]):
        y_i = X[:, i]
        X_others = np.delete(X, i, axis=1)
        reg = LinearRegression().fit(X_others, y_i)
        r2 = reg.score(X_others, y_i)
        vif_val = 1 / (1 - r2) if r2 < 1 else np.inf
        vifs.append({"feature": feature_cols[i], "VIF": round(vif_val, 1)})

    return pd.DataFrame(vifs)


def cooks_distance(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Cook's distance for each observation.
    Flag influential points with D > 4/n.
    """
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    residuals = y - y_pred
    n, p = X.shape
    p += 1  # intercept

    mse = np.sum(residuals ** 2) / (n - p)

    # Hat matrix diagonal
    X_aug = np.column_stack([np.ones(n), X])
    H = X_aug @ np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T
    h = np.diag(H)

    D = (residuals ** 2 / (p * mse)) * (h / (1 - h) ** 2)
    return D


# ═══════════════════════════════════════════════════════════════
# CLUSTER BOOTSTRAP
# ═══════════════════════════════════════════════════════════════

def cluster_bootstrap(
    df: pd.DataFrame,
    stat_fn,
    cluster_col: str = "Campaign",
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Nonparametric cluster bootstrap: resample whole clusters (default: sampling
    campaigns, i.e. year-month) with replacement and recompute `stat_fn`.

    Rows in the same campaign share hydrological conditions, so resampling rows
    would understate uncertainty.

    stat_fn : callable(DataFrame) -> dict or Series of named scalars. It receives
              a resampled frame with a fresh RangeIndex (duplicated clusters
              appear as repeated rows), so anything estimated inside it — e.g.
              seasonal cell means — is re-estimated in every draw.
    Returns one row per draw.
    """
    codes, uniques = pd.factorize(df[cluster_col])
    G = len(uniques)
    if G < 2:
        raise ValueError(f"cluster_bootstrap needs >= 2 clusters in {cluster_col!r}, got {G}.")
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=G)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    members = [order[starts[c]:starts[c] + counts[c]] for c in range(G)]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, G, size=G)
        pos = np.concatenate([members[c] for c in pick])
        draws.append(pd.Series(stat_fn(df.iloc[pos].reset_index(drop=True)), dtype=float))
    return pd.DataFrame(draws).reset_index(drop=True)


def bootstrap_summary(point, draws: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    """
    Percentile intervals for every statistic.

    point : dict/Series of estimates on the original data (same keys as draws).
    Returns estimate, ci_low, ci_high, se and the number of finite draws.
    """
    point = pd.Series(point, dtype=float)
    a = (1 - level) / 2
    q = draws.quantile([a, 1 - a])
    out = pd.DataFrame({
        "estimate": point,
        "ci_low": q.iloc[0].reindex(point.index),
        "ci_high": q.iloc[1].reindex(point.index),
        "se": draws.std().reindex(point.index),
        "n_boot_valid": draws.notna().sum().reindex(point.index),
    })
    out.index.name = "statistic"
    return out
