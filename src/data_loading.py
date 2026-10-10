"""
Loading, deriving and quality-checking the locked analysis datasets.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    RAW_CSV, DS3F_PATH, DS4F_PATH, DS4F_ALK_PATH, DS5F_PATH, DATA_DIR, COLUMN_RENAME,
    EXPECTED_N, EXPECTED_STATIONS, EXPECTED_RIVERS, RIVER_ORDER,
    BOD_COD_RATIO_RANGE, PH_RANGE, TARGET_TRANSFORM,
    MONTH_DUMMY_COLS, PHASE2_MANIFEST,
    DO_ZERO_IS_CENSORED, DO_CENSOR_SUBSTITUTE, DO_CENSOR_INDICATOR, rel,
)
from .preprocessing import target_forward, assert_target_consistent

DATASET_PATHS = {
    "DS-3F": DS3F_PATH,
    "DS-4F": DS4F_PATH,
    "DS-4F-ALK": DS4F_ALK_PATH,
    "DS-5F": DS5F_PATH,
}


# ═══════════════════════════════════════════════════════════════
# KEYS AND FLAGS
# ═══════════════════════════════════════════════════════════════

def _key_part(s: pd.Series) -> pd.Series:
    """Whole numbers as '2017' (not '2017.0'); everything else as stripped text."""
    num = pd.to_numeric(s, errors="coerce")
    whole = num.notna() & (num == num.round())
    txt = s.astype(str).str.strip().fillna("NA")
    return txt.mask(whole, num.round().astype("Int64").astype(str))


def observation_key(df: pd.DataFrame) -> pd.Series:
    """
    River||Place||Year||Month -- the observational unit.

    Single definition used by the loaders, the cross-river duplicate check, and
    the keyed fold record (folds.build_fold_keys). Year/Month are written as
    integers so '2017' and '2017.0' cannot produce different keys.
    """
    required = ["River", "Place", "Year", "Month_num"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Cannot create Observation_Key; missing columns: {missing}")
    return (
        df["River"].astype(str).str.strip() + "||"
        + df["Place"].astype(str).str.strip() + "||"
        + _key_part(df["Year"]) + "||"
        + _key_part(df["Month_num"])
    )


def as_bool(s: pd.Series) -> pd.Series:
    """
    Robust boolean coercion for flag columns that went through a CSV.
    `~df[flag].fillna(False)` breaks when the column is object/str dtype
    ('True'/'False' strings, or bools mixed with NaN).
    """
    if s.dtype == bool:
        return s
    return s.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0", "yes"))


def _stable_sort_and_key(df: pd.DataFrame) -> pd.DataFrame:
    """Create a stable observation key and deterministic row order."""
    df = df.copy()
    df["Observation_Key"] = observation_key(df)

    if df["Observation_Key"].duplicated().any():
        dup = df.loc[df["Observation_Key"].duplicated(keep=False), "Observation_Key"].tolist()
        raise ValueError(
            "Duplicate River/Place/Year/Month observations found: "
            + ", ".join(map(str, dup[:10]))
        )

    return df.sort_values(
        ["River", "Place", "Year", "Month_num", "ID"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def find_cross_river_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify potential duplicate observations occurring at different rivers.

    These records are NOT automatically removed because the source agency
    could not confirm whether they are erroneous. The returned table is used
    only for QC flagging and sensitivity analysis.
    """
    core = ["Year", "Month_num", "pH", "DO", "BOD", "COD", "TDS", "EC"]
    required = core + ["River", "Place", "Observation_Key"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Cross-river duplicate check missing columns: {missing}")

    valid = df.dropna(subset=core).copy()
    dup_mask = valid.duplicated(subset=core, keep=False)
    dup = valid.loc[dup_mask].copy()

    if dup.empty:
        return pd.DataFrame(columns=["Observation_Key"] + core + ["River", "Place"])

    river_counts = dup.groupby(core, dropna=False)["River"].transform("nunique")
    dup = dup.loc[river_counts > 1].copy()

    return dup[["Observation_Key"] + core + ["River", "Place"]].sort_values(
        core + ["River", "Place"], kind="mergesort"
    ).reset_index(drop=True)


def exclude_flagged_cross_river_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy excluding flagged observations for sensitivity analysis.
    The index is reset, because folds.py builds folds from df.index.
    """
    flag = "Potential_CrossRiver_Duplicate"
    if flag not in df.columns:
        raise KeyError(f"Required QC flag '{flag}' is missing.")
    return df.loc[~as_bool(df[flag])].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# DERIVED COLUMNS
# ═══════════════════════════════════════════════════════════════

def add_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar predictors known at prediction time.

      Month_01..Month_11 : month dummies, December is the reference level
      sin_month, cos_month : cyclic month
      IsDry              : 1 if Season is 'Dry' (case/whitespace-insensitive)

    Called by _add_derived, so every locked dataset carries these columns and
    Phase 2 / Phase 3 use exactly the same definitions.
    """
    df = df.copy()
    month = pd.to_numeric(df["Month_num"], errors="raise")
    bad = month.notna() & ~month.between(1, 12)
    if bad.any():
        raise ValueError(f"Month_num outside 1–12 in {int(bad.sum())} rows.")
    if month.notna().all():
        df["Month_num"] = month.astype(int)
    for m, col in zip(range(1, 12), MONTH_DUMMY_COLS):
        df[col] = (month == m).astype(float)
    df["sin_month"] = np.sin(2 * np.pi * month / 12)
    df["cos_month"] = np.cos(2 * np.pi * month / 12)
    if "Season" in df.columns:
        df["IsDry"] = df["Season"].astype(str).str.strip().str.lower().eq("dry").astype(int)
    return df

def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all derived columns. Called once at load time.

    Targets use config.TARGET_TRANSFORM (log1p), matching the raw CSV. Log
    columns already present in the CSV are first checked against this contract
    (assert_target_consistent), so stored and recomputed values cannot disagree.
    """
    df = df.copy()

    # If the CSV already carries log targets, they must agree with our contract.
    for raw, log in (("BOD", "log_BOD"), ("COD", "log_COD")):
        if log in df.columns and raw in df.columns:
            assert_target_consistent(df, raw, log)

    df["log_BOD"] = target_forward(df["BOD"].to_numpy(dtype=float))
    df["log_COD"] = target_forward(df["COD"].to_numpy(dtype=float))

    # Predictor transforms.
    # EC and TDS are strictly positive here (min 65.2 and 35.4), so plain log is
    # safe and keeps the multiplicative interpretation. DO has exact zeros (reported 0.0),
    # which is why it uses log1p -- np.log would give -inf on 13% of rows.
    if "EC" in df.columns:
        df["log_EC"] = np.log(df["EC"].where(df["EC"] > 0))
    if "DO" in df.columns:
        # A reported 0.0 mg/L is a below-reporting-limit value, not a measured
        # zero (see config.DO_ZERO_IS_CENSORED). The flag is always
        # created; the substitute defaults to 0.0, which reproduces the lock.
        do = pd.to_numeric(df["DO"], errors="coerce")
        df[DO_CENSOR_INDICATOR] = (do == 0).astype(float).where(do.notna())
        df["log1p_DO"] = np.log1p(_substitute_do(do, DO_CENSOR_SUBSTITUTE).clip(lower=0))
    if "TDS" in df.columns:
        df["log_TDS"] = np.log(df["TDS"].where(df["TDS"] > 0))
    # Log representation of the optional predictors, defined here so every
    # phase uses the same definition.
    for raw, col in (("Alkalinity", "log1p_Alkalinity"), ("Turbidity", "log1p_Turbidity")):
        if raw in df.columns:
            v = pd.to_numeric(df[raw], errors="coerce")
            df[col] = np.log1p(v.where(v >= 0))

    # Calendar predictors: month dummies, cyclic month, dry-season flag.
    if "Month_num" in df.columns:
        df = add_calendar_columns(df)

    if "River" in df.columns and "Place" in df.columns:
        df["Station_ID"] = df["River"] + "__" + df["Place"]

    return df


def _substitute_do(do: pd.Series, value: float) -> pd.Series:
    """Replace reported 0.0 mg/L DO by `value` (left-censored substitution)."""
    if not DO_ZERO_IS_CENSORED or value == 0:
        return do
    return do.where(do != 0, float(value))


def apply_do_substitution(df: pd.DataFrame, value: float) -> pd.DataFrame:
    """
    Copy of `df` with log1p_DO recomputed after replacing reported 0.0 mg/L DO
    by `value`. `value` should not exceed the smallest positive DO
    the laboratory reported (see do_reporting_floor); 0.0 returns the lock.
    """
    if "DO" not in df.columns:
        raise KeyError("apply_do_substitution needs the raw DO column")
    if value < 0:
        raise ValueError("DO substitute must be >= 0")
    out = df.copy()
    do = pd.to_numeric(out["DO"], errors="coerce")
    out["log1p_DO"] = np.log1p(_substitute_do(do, value).clip(lower=0))
    return out


def do_reporting_floor(df: pd.DataFrame, by: str = "Year") -> pd.DataFrame:
    """
    Per group: rows, reported zeros, their share, and the smallest positive DO
    reported. The smallest positive value is an empirical upper bound on the
    reporting limit when the laboratory does not disclose it.
    """
    do = pd.to_numeric(df["DO"], errors="coerce")
    g = pd.DataFrame({by: df[by], "DO": do}).dropna(subset=["DO"]).groupby(by)["DO"]
    out = pd.DataFrame({
        "n": g.size(),
        "n_reported_zero": g.apply(lambda s: int((s == 0).sum())),
        "smallest_positive_mgL": g.apply(lambda s: float(s[s > 0].min()) if (s > 0).any() else np.nan),
    })
    out["share_reported_zero"] = out["n_reported_zero"] / out["n"]
    total = pd.DataFrame({"n": [int(do.notna().sum())], "n_reported_zero": [int((do == 0).sum())],
                          "smallest_positive_mgL": [float(do[do > 0].min())],
                          "share_reported_zero": [float((do == 0).sum() / do.notna().sum())]},
                         index=pd.Index(["all"], name=by))
    return pd.concat([out, total]).reset_index()


def whole_number_share(df: pd.DataFrame, cols, by: str = "Year", atol: float = 1e-9) -> pd.DataFrame:
    """
    Share of values reported as whole numbers, per group. A rising
    share at the same stations signals a change in laboratory reporting
    resolution, not in the water.
    """
    rows = {}
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        ok = v.notna()
        whole = pd.Series(np.isclose(v[ok], np.round(v[ok]), atol=atol), index=v[ok].index)
        rows[c] = whole.groupby(df.loc[ok, by]).mean()
    return pd.DataFrame(rows).reset_index()


def load_locked_dataset(name: str = "DS-3F", add_derived: bool = True) -> pd.DataFrame:
    """
    Load a pre-built analysis dataset with hard size assertions.

    Returns a DataFrame with a RESET index -- folds.py builds fold indices from
    df.index and evaluation.py uses .loc, so a non-contiguous index silently
    misaligns the two.
    """
    paths = DATASET_PATHS
    if name not in paths:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(paths.keys())}")

    path = paths[name]
    if not path.exists():
        raise FileNotFoundError(
            f"{rel(path)} not found. Run notebooks/phase0-1_data_qc_eda.ipynb first."
        )

    df = pd.read_csv(path).reset_index(drop=True)
    df = _stable_sort_and_key(df)

    # ── Hard assertions ──────────────────────────────────────
    n_expected = EXPECTED_N[name]
    assert len(df) == n_expected, (
        f"{name}: expected {n_expected} rows, got {len(df)}. "
        f"Data may have been modified since lock."
    )

    n_rivers = df["River"].nunique()
    if name == "DS-5F":
        assert n_rivers == 4, f"{name}: expected 4 rivers, got {n_rivers}"
    else:
        assert n_rivers == EXPECTED_RIVERS, (
            f"{name}: expected {EXPECTED_RIVERS} rivers, got {n_rivers}"
        )

    if name != "DS-5F":
        n_stations = df.groupby(["River", "Place"]).ngroups
        assert n_stations == EXPECTED_STATIONS, (
            f"{name}: expected {EXPECTED_STATIONS} stations, got {n_stations}"
        )

    # ── Soft checks (warn, don't crash) ──────────────────────
    if (df["BOD"] <= 0).any():
        print(f"WARNING: {name} has {(df['BOD'] <= 0).sum()} non-positive BOD values")
    if (df["COD"] <= 0).any():
        print(f"WARNING: {name} has {(df['COD'] <= 0).sum()} non-positive COD values")

    if add_derived:
        df = _add_derived(df)

    return df


def build_datasets(raw_path: Path = None) -> dict:
    """
    Phase 0.2: build locked analysis datasets from the raw CSV and save to data/.
    Run ONCE; afterwards use load_locked_dataset().
    """
    raw_path = raw_path or RAW_CSV
    try:
        df = pd.read_csv(raw_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(raw_path, encoding="latin-1")
    df = df.rename(columns=COLUMN_RENAME)
    print(f"Raw: {len(df)} rows, {len(df.columns)} cols")

    df = _stable_sort_and_key(df)

    # Potential cross-river duplicates are retained because the DoE source
    # could not confirm whether they are erroneous. Flag them for QC and
    # sensitivity analysis instead of silently deleting them.
    cross_river_dupes = find_cross_river_duplicates(df)
    df["Potential_CrossRiver_Duplicate"] = False
    if not cross_river_dupes.empty:
        flagged_keys = set(cross_river_dupes["Observation_Key"].astype(str))
        df["Potential_CrossRiver_Duplicate"] = (
            df["Observation_Key"].astype(str).isin(flagged_keys)
        )
        print(
            f"WARNING: {len(flagged_keys)} observations flagged as potential "
            "cross-river duplicates; retained in all primary datasets."
        )

    for raw, log in (("BOD", "log_BOD"), ("COD", "log_COD")):
        if log in df.columns:
            assert_target_consistent(df, raw, log)
    print(f"Target transform contract OK: {TARGET_TRANSFORM}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {}
    # Keep FS-D separate from DS-4F because Alkalinity has a different
    # missing-data pattern. If FS-D were allowed to read DS-4F, rows with
    # missing Alkalinity would survive into the feature matrix and could fail
    # during model fitting. This dataset is complete-case specifically for
    # EC + pH + DO + Alkalinity + BOD + COD.
    dataset_specs = [
        ("DS-3F", ["EC", "pH", "DO"]),
        ("DS-4F", ["EC", "pH", "DO", "TDS"]),
        ("DS-4F-ALK", ["EC", "pH", "DO", "Alkalinity"]),
        ("DS-5F", ["EC", "pH", "DO", "TDS", "Turbidity"]),
    ]
    for name, req_features in dataset_specs:
        sub = df.dropna(subset=req_features + ["BOD", "COD"]).copy().reset_index(drop=True)
        out_path = DATA_DIR / f"{name.replace('-', '_')}.csv"
        sub.to_csv(out_path, index=False)
        datasets[name] = sub
        print(f"  {name}: {len(sub)} rows → {out_path.name}")

    # Explicit lock check for the dedicated FS-D dataset.
    assert len(datasets["DS-4F-ALK"]) == EXPECTED_N["DS-4F-ALK"], (
        f"DS-4F-ALK: expected {EXPECTED_N['DS-4F-ALK']} rows, "
        f"got {len(datasets['DS-4F-ALK'])}"
    )

    return datasets


def run_integrity_checks(df: pd.DataFrame) -> dict:
    """
    Phase 0, section 0.1: all pre-flight checks.

    Each entry has `ok` (False = hard failure) and, for plausibility checks,
    `flagged` (True = rows retained but listed for the record).
    """
    checks = {}

    n_stations = df.groupby(["River", "Place"]).ngroups
    checks["0.1a_stations"] = {"value": n_stations, "expected": EXPECTED_STATIONS,
                               "ok": n_stations == EXPECTED_STATIONS}

    if "Month_num" in df.columns:
        n_dup = df.duplicated(subset=["River", "Place", "Year", "Month_num"]).sum()
        checks["0.1b_duplicates"] = {"value": int(n_dup), "expected": 0, "ok": n_dup == 0}

    checks["0.1c_positive"] = {
        "BOD_min": df["BOD"].min(), "COD_min": df["COD"].min(),
        "ok": (df["BOD"].min() > 0) and (df["COD"].min() > 0),
    }

    if "Observation_Key" in df.columns:
        cross_dupes = find_cross_river_duplicates(df)
        n_flag = int(cross_dupes["Observation_Key"].nunique())
        checks["0.1c_cross_river_duplicates"] = {
            "n_flagged": n_flag,
            "ok": True,               # unresolved source provenance: retained, not failed
            "flagged": n_flag > 0,    # ... but reported
        }

    ratio = df["BOD"] / df["COD"]
    lo, hi = BOD_COD_RATIO_RANGE
    outside = int(((ratio < lo) | (ratio > hi)).sum())
    checks["0.1d_ratio"] = {
        "n_outside": outside, "pct": f"{100 * outside / len(ratio):.1f}%",
        "mean": f"{ratio.mean():.2f}",
        "n_bod_equals_cod": int(np.isclose(ratio, 1.0).sum()),
        "ok": True,               # plausibility range: retained, not failed
        "flagged": outside > 0,
    }
    # BOD5 is a fraction of COD; BOD > COD is physically impossible.
    n_gt = int((ratio > 1 + 1e-12).sum())
    checks["0.1d_bod_le_cod"] = {"n_bod_gt_cod": n_gt, "ok": n_gt == 0}

    plo, phi = PH_RANGE
    ph_out = int(((df["pH"] < plo) | (df["pH"] > phi)).sum())
    checks["0.1e_pH"] = {"n_outside": ph_out,
                         "range": f"{df['pH'].min():.1f}–{df['pH'].max():.1f}",
                         "ok": ph_out == 0}

    do_neg = int((df["DO"] < 0).sum())
    checks["0.1f_DO"] = {"n_negative": do_neg, "min": df["DO"].min(), "ok": do_neg == 0}

    # 0.1g: the log columns still match the declared transform
    try:
        assert_target_consistent(df, "BOD", "log_BOD")
        assert_target_consistent(df, "COD", "log_COD")
        checks["0.1g_log_transform"] = {"transform": TARGET_TRANSFORM, "ok": True}
    except AssertionError as e:
        checks["0.1g_log_transform"] = {"transform": TARGET_TRANSFORM,
                                        "error": str(e), "ok": False}
    return checks


def check_status(v: dict) -> str:
    """PASS / FLAG / FAIL for one entry of run_integrity_checks()."""
    if not v.get("ok", True):
        return "FAIL"
    return "FLAG" if v.get("flagged", False) else "PASS"


def print_checks(checks: dict) -> None:
    symbol = {"PASS": "✓", "FLAG": "⚠", "FAIL": "✗"}
    for k, v in checks.items():
        print(f"  {symbol[check_status(v)]} {k}: {v}")


# ═══════════════════════════════════════════════════════════════
# PHASE 2 LOCK
# ═══════════════════════════════════════════════════════════════

LOCK_KEYS = (
    "locked_dataset", "locked_features", "locked_chemistry_features",
    "locked_temporal_features", "locked_pca_block", "ridge_alphas",
)


def load_phase2_lock(path: Path = None) -> dict:
    """
    Read the feature/population lock written by Phase 2.

    Phase 3 onward must take features and dataset from here rather than from
    config.PRIMARY_FEATURES. Raises if the manifest is missing or incomplete.
    """
    path = Path(path or PHASE2_MANIFEST)
    if not path.exists():
        raise FileNotFoundError(f"{rel(path)} not found. Run the Phase 2 notebook first.")
    lock = json.loads(path.read_text())
    missing = [k for k in LOCK_KEYS if k not in lock]
    if missing:
        raise KeyError(
            f"{path.name} has no {missing}: rerun the Phase 2 notebook."
        )
    if lock["locked_dataset"] not in DATASET_PATHS:
        raise ValueError(f"Unknown locked_dataset {lock['locked_dataset']!r}.")
    lock["locked_pca_block"] = lock["locked_pca_block"] or None
    return lock
