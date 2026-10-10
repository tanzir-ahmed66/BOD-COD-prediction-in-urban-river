"""
Cross-validation fold builders.

OUTER folds: list of (train_idx, test_idx, fold_label) tuples.
INNER CV:    sklearn-compatible splitter objects for hyperparameter search.

The inner splitters are the important addition. GridSearchCV's default cv=5 is
random KFold, which puts observations from the same river (and often the same
station-month) on both sides of the inner split. Tuning under that regime and
then reporting under LOSO-River is not nested CV -- it selects hyperparameters
for a task the model is never evaluated on.
"""
import json
from pathlib import Path
from typing import List, Tuple, Iterator, Optional

import numpy as np
import pandas as pd

from .config import SEED, FC_SPLITS, FOLDS_PATH, FOLD_KEYS_PATH, INNER_CV, rel


Fold = Tuple[np.ndarray, np.ndarray, str]


# ═══════════════════════════════════════════════════════════════
# OUTER FOLDS
# ═══════════════════════════════════════════════════════════════

def build_loso_river(df: pd.DataFrame, river_col: str = "River") -> List[Fold]:
    """Leave-One-River-Out: 5 folds. PRIMARY spatial evaluation protocol."""
    folds = []
    for river in sorted(df[river_col].unique()):
        mask = df[river_col] == river
        folds.append((df.index[~mask].values, df.index[mask].values, river))
    return folds


def build_loso_station(df: pd.DataFrame, station_col: str = "Station_ID") -> List[Fold]:
    """Leave-One-Station-Out: 30 folds. Within-river generalisation."""
    folds = []
    for station in sorted(df[station_col].unique()):
        mask = df[station_col] == station
        folds.append((df.index[~mask].values, df.index[mask].values, station))
    return folds


def build_fc(df: pd.DataFrame, year_col: str = "Year") -> List[Fold]:
    """
    Forward-Chain temporal validation: 4 splits.

    FC-1: train 2017–2018 → test 2019
    FC-2: train 2017–2019 → test 2020
    FC-3: train 2017–2020 → test 2021
    FC-4: train 2017–2021 → test 2022

    Composition depends on the dataset (see config.FC_SPLITS): Tongi Khal is
    never in FC-3/FC-4 tests; about half of FC-3's test rows come from stations
    that start in 2021; Turag is absent from FC-2 in DS-4F only.
    """
    folds = []
    for name, spec in FC_SPLITS.items():
        train_mask = df[year_col].isin(spec["train"])
        test_mask = df[year_col] == spec["test"]
        folds.append((df.index[train_mask].values, df.index[test_mask].values, name))
    return folds


def build_stratified_kfold(df: pd.DataFrame, n_folds: int = 5,
                           river_col: str = "River") -> List[Fold]:
    """Stratified K-Fold by river. FOR LITERATURE COMPARISON ONLY — inflates."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    return [(df.index[tr].values, df.index[te].values, f"fold_{i}")
            for i, (tr, te) in enumerate(skf.split(df, df[river_col]))]


def build_random_splits(df: pd.DataFrame, n_repeats: int = 20,
                        test_frac: float = 0.20) -> List[Fold]:
    """Random 80/20 splits. Upper bound of inflation."""
    from sklearn.model_selection import ShuffleSplit

    ss = ShuffleSplit(n_splits=n_repeats, test_size=test_frac, random_state=SEED)
    return [(df.index[tr].values, df.index[te].values, f"split_{i}")
            for i, (tr, te) in enumerate(ss.split(df))]


# ═══════════════════════════════════════════════════════════════
# INNER CV SPLITTERS
# ═══════════════════════════════════════════════════════════════

class ForwardChainGroupSplit:
    """
    Expanding-window splitter over an ordered grouping variable (Year).

    With years [2017, 2018, 2019, 2020] and min_train=2:
        train {2017, 2018}             -> test 2019
        train {2017, 2018, 2019}       -> test 2020

    sklearn's TimeSeriesSplit cannot be used here: it slices on row position and
    would cut a year in half. Groups must stay intact or the "no future
    information" guarantee is void.
    """

    requires_groups = True

    def __init__(self, min_train: int = 2, max_splits: Optional[int] = None):
        self.min_train = min_train
        self.max_splits = max_splits

    def _effective_min_train(self, n_groups: int) -> int:
        """
        Shrink the warm-up window when the outer training set is short.
        FC-1 trains on 2017-2018 only -- 2 year-groups. With min_train=2 there
        is nothing left to validate on, so the inner loop would crash on the
        first outer fold. One year of warm-up still respects the ordering.
        """
        eff = min(self.min_train, n_groups - 1)
        if eff < 1:
            raise ValueError(
                f"Need >= 2 distinct groups to forward-chain, got {n_groups}. "
                "Widen the outer training window."
            )
        return eff

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("ForwardChainGroupSplit requires `groups` (e.g. Year).")
        groups = np.asarray(groups)
        order = np.sort(np.unique(groups))
        min_train = self._effective_min_train(len(order))
        splits = []
        for i in range(min_train, len(order)):
            tr = np.flatnonzero(np.isin(groups, order[:i]))
            te = np.flatnonzero(groups == order[i])
            if len(tr) and len(te):
                splits.append((tr, te))
        if self.max_splits is not None:
            splits = splits[-self.max_splits:]
        yield from splits

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if groups is None:
            raise ValueError("ForwardChainGroupSplit requires `groups`.")
        n = len(np.unique(np.asarray(groups)))
        return max(0, n - self._effective_min_train(n))


def make_inner_cv(scheme: str = None, n_groups_available: int = None):
    """
    Build the inner splitter that mirrors an outer protocol.

    scheme : key into config.INNER_CV -- "loso_river", "loso_station",
             "forward_chain", "stratified_5f", "random".
    n_groups_available : number of distinct groups present in the OUTER training
             set. GroupKFold cannot make more splits than it has groups; under
             LOSO-River the outer train has 4 rivers, so n_splits is clamped.

    Returns (splitter, group_col). group_col is the dataframe column to pass as
    `groups=` at fit time; None for plain KFold.
    """
    from sklearn.model_selection import GroupKFold, KFold

    from .config import DEFAULT_INNER_SCHEME
    scheme = scheme or DEFAULT_INNER_SCHEME
    if scheme not in INNER_CV:
        raise ValueError(f"Unknown inner scheme {scheme!r}. Have: {list(INNER_CV)}")
    spec = INNER_CV[scheme]

    if spec["kind"] == "forward_chain":
        return ForwardChainGroupSplit(min_train=spec.get("min_train", 2)), spec["group_col"]

    if spec["kind"] == "group":
        n = spec["n_splits"]
        if n_groups_available is not None:
            n = min(n, n_groups_available)
        if n < 2:
            raise ValueError(
                f"Inner scheme {scheme!r} needs >= 2 groups in the outer training set, "
                f"got {n_groups_available}. Fall back to scheme='random' and say so "
                f"in the manuscript."
            )
        return GroupKFold(n_splits=n), spec["group_col"]

    return KFold(n_splits=spec["n_splits"], shuffle=True, random_state=SEED), None


def inner_cv_for_folds(df: pd.DataFrame, folds: List[Fold], outer_scheme: str):
    """
    Inner splitter for a given outer protocol, sized to the outer training sets.

    Returns (splitter, group_col) exactly like make_inner_cv(). The group column
    is read from config.INNER_CV, and n_groups_available is the smallest number
    of distinct groups in any outer training set (e.g. 4 rivers under
    LOSO-River), so GroupKFold never asks for more splits than it has groups.
    """
    inner_scheme = inner_scheme_for(outer_scheme)
    group_col = INNER_CV[inner_scheme]["group_col"]
    n_groups = (min(df.loc[tr, group_col].nunique() for tr, _, _ in folds)
                if group_col else None)
    return make_inner_cv(inner_scheme, n_groups_available=n_groups)


def inner_scheme_for(outer_scheme: str) -> str:
    """Map an outer protocol name to its matching inner scheme."""
    mapping = {
        "loso_river": "loso_river", "LOSO-River": "loso_river",
        "loso_station": "loso_station", "LOSO-Station": "loso_station",
        "forward_chain": "forward_chain", "Forward-Chain": "forward_chain",
        "stratified_5f": "stratified_5f", "Stratified 5-fold": "stratified_5f",
    }
    return mapping.get(outer_scheme, "random")


# ═══════════════════════════════════════════════════════════════
# PERSISTENCE
# ═══════════════════════════════════════════════════════════════

def save_all_folds(df: pd.DataFrame, path: Path = None) -> None:
    """
    Phase 0, section 0.3: pre-compute and save all OUTER fold indices.

    NB: these are row POSITIONS in `df`'s order, valid for that dataset only.
    Use save_fold_keys()/verify_fold_keys() to pin folds across datasets.
    """
    path = path or FOLDS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    all_folds = {
        "loso_river":    {lab: {"train": tr.tolist(), "test": te.tolist()}
                          for tr, te, lab in build_loso_river(df)},
        "loso_station":  {lab: {"train": tr.tolist(), "test": te.tolist()}
                          for tr, te, lab in build_loso_station(df)},
        "forward_chain": {lab: {"train": tr.tolist(), "test": te.tolist()}
                          for tr, te, lab in build_fc(df)},
        "stratified_5f": {lab: {"train": tr.tolist(), "test": te.tolist()}
                          for tr, te, lab in build_stratified_kfold(df)},
    }

    with open(path, "w") as f:
        json.dump(all_folds, f)
    print(f"Saved fold indices → {rel(path)}")
    for scheme, folds in all_folds.items():
        sizes = [len(v["test"]) for v in folds.values()]
        print(f"  {scheme}: {len(folds)} folds, test sizes {min(sizes)}–{max(sizes)}")


OUTER_BUILDERS = {
    "loso_river": build_loso_river,
    "loso_station": build_loso_station,
    "forward_chain": build_fc,
    "stratified_5f": build_stratified_kfold,
    "random_8020": build_random_splits,
}


def build_fold_keys(df: pd.DataFrame) -> dict:
    """
    Observation keys in every test fold of every outer protocol.

    fold_indices.json stores row POSITIONS, which are only valid for one
    dataset's row order. Keys identify the observations themselves, so any
    later notebook can rebuild folds from its own dataset and prove they match.
    """
    from .data_loading import observation_key

    d = df.reset_index(drop=True)
    keys = observation_key(d)
    if not keys.is_unique:
        raise ValueError("Observation keys are not unique; fold keys would be ambiguous.")
    out = {}
    for proto, builder in OUTER_BUILDERS.items():
        out[proto] = [
            {"label": str(label), "n_test": int(len(te)), "test_keys": sorted(keys.loc[te].tolist())}
            for _, te, label in builder(d)
        ]
    return {"n_rows": int(len(d)), "folds": out}


def save_fold_keys(datasets: dict, path: Path = None) -> Path:
    """datasets: {name: DataFrame}. Writes FOLD_KEYS_PATH (or `path`)."""
    path = Path(path or FOLD_KEYS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({name: build_fold_keys(d) for name, d in datasets.items()}))
    print(f"Keyed folds for {list(datasets)} → {rel(path)}")
    return path


def verify_fold_keys(df: pd.DataFrame, dataset_name: str, folds_by_protocol: dict,
                     path: Path = None) -> Optional[pd.DataFrame]:
    """
    Compare rebuilt folds with the keys frozen in Phase 0.

    folds_by_protocol : {"loso_river": folds, ...}, keys as in OUTER_BUILDERS.
    Returns a per-fold table with `matches_frozen`, or None when no frozen
    record exists (the caller decides whether that is acceptable). Raises if the
    dataset's row count changed since the folds were frozen.
    """
    from .data_loading import observation_key

    path = Path(path or FOLD_KEYS_PATH)
    if not path.exists():
        return None
    frozen = json.loads(path.read_text()).get(dataset_name)
    if frozen is None:
        return None
    if frozen["n_rows"] != len(df):
        raise AssertionError(
            f"{dataset_name} has {len(df)} rows now but {frozen['n_rows']} when the folds were frozen."
        )
    keys = observation_key(df)
    rows = []
    for proto, folds in folds_by_protocol.items():
        ref = {f["label"]: set(f["test_keys"]) for f in frozen["folds"].get(proto, [])}
        for _, te, label in folds:
            rows.append({"protocol": proto, "fold": str(label),
                         "matches_frozen": set(keys.loc[te]) == ref.get(str(label))})
    return pd.DataFrame(rows)


def load_folds(scheme: str, path: Path = None) -> List[Fold]:
    """Load pre-saved OUTER fold indices from JSON."""
    path = path or FOLDS_PATH
    with open(path) as f:
        data = json.load(f)
    if scheme not in data:
        raise ValueError(f"Unknown scheme '{scheme}'. Available: {list(data.keys())}")
    return [(np.array(idx["train"]), np.array(idx["test"]), label)
            for label, idx in data[scheme].items()]
