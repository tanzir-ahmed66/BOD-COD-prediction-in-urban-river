"""
Project-wide constants. No logic lives here except path construction.
"""
from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────
SEED = 42

# ── Paths ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGURES_DIR  = PROJECT_ROOT / "figures"

RAW_CSV = DATA_DIR / "DOE_Data_clean_audited_v4.csv"
DS3F_PATH = DATA_DIR / "DS_3F.csv"
DS4F_PATH = DATA_DIR / "DS_4F.csv"
DS4F_ALK_PATH = DATA_DIR / "DS_4F_ALK.csv"
DS5F_PATH = DATA_DIR / "DS_5F.csv"
FOLDS_PATH = DATA_DIR / "fold_indices.json"            # DS-3F row POSITIONS (DS-3F row order only)
FOLD_KEYS_PATH = DATA_DIR / "fold_keys_by_dataset.json"  # observation keys, every dataset
PHASE2_MANIFEST = RESULTS_DIR / "phase2_manifest.json"   # the feature/population lock


def rel(path) -> str:
    """`path` relative to the project root, for printed messages and manifests."""
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)

# Raw file ships with units in the header; everything downstream uses bare names.
COLUMN_RENAME = {
    "DO (mg/L)":         "DO",
    "BOD (mg/L)":        "BOD",
    "COD (mg/L)":        "COD",
    "TDS (mg/l)":        "TDS",
    "Turbidity (NTU)":   "Turbidity",
    "Chloride (mg/l)":   "Chloride",
    "SS (mg/l)":         "SS",
    "Alkalinity (mg/l)": "Alkalinity",
    "EC (umohs/cm)":     "EC",
}

# ── Target transform ──────────────────────────────────────────
# The raw CSV defines log_BOD = log1p(BOD) and log_COD = log1p(COD).
# Verified: max |stored - log1p(x)| = 5e-10 over 1154 rows; plain log would be
# off by 1.79. Changing this constant changes forward AND inverse together
# (see preprocessing.target_forward / target_inverse). Never hard-code np.exp.
TARGET_TRANSFORM = "log1p"          # {"log1p", "log"}

# ── Rivers (fixed order for all plots and tables) ─────────────
RIVER_ORDER = ["Buriganga", "Turag", "Shitalakhya", "Balu River", "Tongi Khal"]

RIVER_COLORS = {
    "Buriganga":    "#1f77b4",
    "Turag":        "#ff7f0e",
    "Shitalakhya":  "#2ca02c",
    "Balu River":   "#d62728",
    "Tongi Khal":   "#9467bd",
}

# Feature set configurations.
# `pca_block` names a block collapsed by FoldSafePCA *inside* the pipeline.
FEATURE_CONFIGS = {
    "FS-A": {"cols": ["EC", "pH", "DO"],                     "dataset": "DS-3F"},
    "FS-B": {"cols": ["TDS", "pH", "DO"],                    "dataset": "DS-4F"},
    "FS-C": {"cols": ["EC", "pH", "DO", "TDS"],              "dataset": "DS-4F"},
    "FS-D": {"cols": ["EC", "pH", "DO", "Alkalinity"],       "dataset": "DS-4F-ALK"},
    "FS-E": {"cols": ["log_EC", "log_TDS", "pH", "log1p_DO"],
             "dataset": "DS-4F", "pca_block": "PC1_EC_TDS"},
    "FS-F": {"cols": ["EC", "pH", "DO", "TDS", "Turbidity"], "dataset": "DS-5F"},
    "FS-G": {"cols": ["EC", "DO"],                           "dataset": "DS-3F"},
    "FS-H": {"cols": ["pH", "DO"],                           "dataset": "DS-3F"},
}

# ── Log representation of the feature sets ───────────────────
# FEATURE_CONFIGS above is the RAW representation, reported as a descriptive
# check only. Feature-set decisions (Phase 2) are taken on the LOG
# representation below, which is the one that gets locked (PRIMARY_FEATURES).
# Columns are created in data_loading._add_derived.
FEATURE_CONFIGS_LOG = {
    "FS-A": {"cols": ["log_EC", "pH", "log1p_DO"],                       "dataset": "DS-3F"},
    "FS-B": {"cols": ["log_TDS", "pH", "log1p_DO"],                      "dataset": "DS-4F"},
    "FS-C": {"cols": ["log_EC", "log_TDS", "pH", "log1p_DO"],            "dataset": "DS-4F"},
    "FS-D": {"cols": ["log_EC", "pH", "log1p_DO", "log1p_Alkalinity"],   "dataset": "DS-4F-ALK"},
    "FS-E": {"cols": ["log_EC", "log_TDS", "pH", "log1p_DO"],
             "dataset": "DS-4F", "pca_block": "PC1_EC_TDS"},
    "FS-F": {"cols": ["log_EC", "log_TDS", "pH", "log1p_DO", "log1p_Turbidity"], "dataset": "DS-5F"},
}

# ── PCA blocks (fold-safe; see preprocessing.build_preprocessor) ──
# EC and TDS are near-collinear (both track dissolved ionic load). Collapsing
# them to one component is a collinearity fix, not feature engineering.
# sign_ref pins the component sign so loadings are comparable across folds:
# PC1 always loads positively on log_EC.
PCA_BLOCKS = {
    "PC1_EC_TDS": {
        "cols": ["log_EC", "log_TDS"],
        "n_components": 1,
        "sign_ref": "log_EC",
    },
}

# ── Calendar columns (added by data_loading._add_derived) ──────
# Month dummies use December as the reference level.
MONTH_DUMMY_COLS = [f"Month_{m:02d}" for m in range(1, 12)]
CYCLIC_MONTH_COLS = ["sin_month", "cos_month"]
SEASON_COLS = ["IsDry"]

# ── Primary feature set ───────────────────────────────────────
# Mirrors the Phase 2 lock in PHASE2_MANIFEST (EC chosen over TDS; no calendar
# term passed the joint LOSO-River + Forward-Chain rule). Phase 3 onward read
# the lock with data_loading.load_phase2_lock(), not from here; keep this in
# sync if Phase 2 is ever rerun with a different outcome.
PRIMARY_FEATURES = ["log_EC", "pH", "log1p_DO"]
PRIMARY_TARGET   = "log_BOD"

# ── Water-quality standards: ECR 2023 ────────────────────────
# Environment Conservation Rules 2023, Schedule 2 (Rule 31), Part (a)(1):
# inland surface-water quality standards by designated use.
# Bangladesh Gazette Extraordinary, 5 March 2023, p. 3061.
# Upper limits in mg/L (DO_min is a lower limit). Exceedance = value > limit.
# Applied AFTER target_inverse(); never compare against log space.
ECR2023_INLAND_SURFACE = {
    "drinking-water source, disinfection only":      {"BOD": 2,  "COD": 10,  "DO_min": 6},
    "recreation":                                    {"BOD": 3,  "COD": 10,  "DO_min": 5},
    "drinking-water source, conventional treatment": {"BOD": 3,  "COD": 25,  "DO_min": 5},
    "fisheries":                                     {"BOD": 6,  "COD": 50,  "DO_min": 5},
    "industrial process and cooling":                {"BOD": 12, "COD": 100, "DO_min": 1},
    "irrigation":                                    {"BOD": 12, "COD": 100, "DO_min": None},
}
PRIMARY_THRESHOLD = 6          # BOD, fisheries use class — PRIMARY
COD_PRIMARY_THRESHOLD = 50     # COD, fisheries use class
THRESHOLDS = {                 # named thresholds (evaluation.compute_metrics)
    "BOD > 6 (fisheries)": 6,
    "BOD > 12 (industrial / irrigation)": 12,
    "COD > 50 (fisheries)": 50,
    "COD > 100 (industrial / irrigation)": 100,
}

# ── Hyperparameter grids ─────────────────────────────────────
# Keys are pipeline-qualified: GridSearchCV wraps the WHOLE pipeline
# (preprocessing included), so parameter names carry the 'est__' prefix.
#
# The grids extend beyond the values that get selected in practice. Phase 3
# writes phase3_hyperparameter_edge_audit.csv, listing every parameter chosen at
# a grid edge — check it after each run.
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
RIDGE_GRID = {"est__alpha": RIDGE_ALPHAS}

# Lasso's penalty is on |beta| and scaled by 1/(2n); ridge-scale alphas (>= 1)
# zero every coefficient on standardised features, so lasso has its own grid.
LASSO_ALPHAS = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0]
LASSO_GRID = {"est__alpha": LASSO_ALPHAS}
ENET_GRID = {
    "est__alpha":    [1e-4, 1e-3, 0.01, 0.03, 0.1, 0.3, 1.0],
    "est__l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9],
}

RF_GRID = {
    "est__n_estimators":     [300],          # more trees never hurt RF; not a tuning knob
    "est__max_depth":        [2, 3, 5, 10, None],
    "est__min_samples_leaf": [5, 10, 20, 40, 80],
    "est__max_features":     [1.0, 0.5],
}

XGB_GRID = {
    "est__learning_rate": [0.02, 0.05, 0.1],
    "est__max_depth":     [1, 2, 3, 5],
    "est__n_estimators":  [100, 300, 600],
    "est__reg_lambda":    [1, 10, 100],
}
# LightGBM accepts these names as aliases. Its own complexity controls
# (num_leaves, min_child_samples) stay at defaults; num_leaves is capped by
# max_depth anyway (2**depth leaves).
LGBM_GRID = dict(XGB_GRID)

# ── Inner CV (hyperparameter selection) ──────────────────────
# The inner loop must mirror the outer protocol. Otherwise alpha is tuned on a
# random split (optimistic) and evaluated on a spatial split (honest), and the
# selected penalty is systematically too weak.
INNER_CV = {
    "loso_river":    {"kind": "group",         "group_col": "River",      "n_splits": 4},
    "loso_station":  {"kind": "group",         "group_col": "Station_ID", "n_splits": 5},
    "forward_chain": {"kind": "forward_chain", "group_col": "Year",       "min_train": 2},
    # Row-wise outer protocol -> row-wise inner split, so tuning matches what
    # is evaluated.
    "stratified_5f": {"kind": "kfold",         "group_col": None,         "n_splits": 5},
    "random":        {"kind": "kfold",         "group_col": None,         "n_splits": 5},
}
DEFAULT_INNER_SCHEME = "loso_river"

# ── Exceedance rule ──────────────────────────────────────────
# ECR 2023 limits are upper limits; "exceedance" is read as value > limit,
# the rule used since Phase 3. Laboratory BOD is increasingly reported as whole
# numbers (15% of values in 2017, 95% in 2021), so 44 rows sit exactly at
# 6 mg/L and 52 at 12 mg/L. Every threshold function takes `rule` so the
# inclusive reading (>=) can be reported as a sensitivity. ">" is the primary
# rule.
EXCEEDANCE_RULE = ">"
EXCEEDANCE_RULES = (">", ">=")

# ── Dissolved oxygen reported as 0.0 mg/L ─────────────────────
# DO is a modified-Winkler titration (DoE methods table). 143 DS-3F rows report
# exactly 0.0 mg/L; the smallest positive value reported is 0.1 mg/L overall but
# 0.2 in 2018 and 0.28 in 2020, and the share of zeros peaks in 2020 (26.5%).
# DoE does not disclose its reporting limit, so a reported zero is treated as
# LEFT-CENSORED below an unknown limit. The primary analysis keeps the reported
# 0.0; a censoring indicator column is always created.
DO_ZERO_IS_CENSORED = True
DO_CENSOR_SUBSTITUTE = 0.0
DO_CENSOR_INDICATOR = "DO_censored"

# ── Forward-chain splits (Phase 0, section 0.3) ──────────────
# Test-set composition depends on the dataset:
#   * Tongi Khal has no observations after 2020, so FC-3 and FC-4 never test it.
#   * Nine stations start in 2021, so roughly half of FC-3's test rows come from
#     stations absent from its training set.
#   * Turag has no 2020 rows in DS-4F (TDS missing), so FC-2 lacks Turag THERE
#     ONLY. In DS-3F, 2020 contains all five rivers.
# Phase 3 writes the exact composition to phase3_fold_composition.csv.
FC_SPLITS = {
    "FC-1": {"train": [2017, 2018],             "test": 2019},
    "FC-2": {"train": [2017, 2018, 2019],       "test": 2020},
    "FC-3": {"train": [2017, 2018, 2019, 2020], "test": 2021},
    "FC-4": {"train": list(range(2017, 2022)),  "test": 2022},
}

# ── Data quality flags (Phase 0, section 0.1) ─────────────────
BOD_COD_RATIO_RANGE = (0.05, 0.80)
PH_RANGE = (5.5, 9.5)

# ── Decision rules ───────────────────────────────────────────
PARSIMONY_DR2 = 0.02    # ΔR² < 0.02 → simpler model wins
TEMPORAL_DR2  = 0.01    # temporal features must improve both tracks by ≥ 0.01

# ── Expected dataset sizes (for assertions) ──────────────────
EXPECTED_N = {"DS-3F": 1131, "DS-4F": 1083, "DS-4F-ALK": 1129, "DS-5F": 710}
EXPECTED_STATIONS = 30
EXPECTED_RIVERS = 5
