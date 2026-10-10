# Data

The monitoring data are **not included** in this repository. They come from the surface-water quality monitoring programme of the Department of Environment (DoE), Bangladesh, and were obtained from DoE for this study. Requests for the data should go to DoE.

## Expected input

One CSV file, `DOE_Data_clean_audited_v4.csv` (the name is set in `src/config.py`, `RAW_CSV`):

- **1,184 rows**: one row per station and month
- **5 rivers**: Buriganga, Turag, Shitalakhya, Balu River, Tongi Khal
- **30 stations**, sampled monthly from **2017 to 2022**

| Column(s) | Content |
|---|---|
| `ID`, `Source` | Row identifier; data source (`DoE`) |
| `River`, `Place` | River and station name; together they identify a station |
| `Lat`, `Long`, `Lat_decimal`, `Long_decimal` | Station coordinates (DMS text and decimal degrees) |
| `Year`, `Month`, `Month_num`, `Season` | Sampling year and month; `Season` is `Dry` (Nov–Apr) or `Wet` (May–Oct) |
| `pH`, `DO (mg/L)`, `EC (umohs/cm)` | Predictors |
| `BOD (mg/L)`, `COD (mg/L)` | Targets |
| `TDS (mg/l)`, `Alkalinity (mg/l)`, `Turbidity (NTU)`, `Chloride (mg/l)`, `SS (mg/l)` | Additional parameters (TDS is a candidate predictor in Phase 2) |
| `log_BOD`, `log_COD`, `BOD_COD_ratio` | Optional. If present, `log_BOD`/`log_COD` must equal `log1p` of the raw values; the loader checks this |

Columns with units in the header are renamed on load (`COLUMN_RENAME` in `src/config.py`). The loader asserts the expected row and station counts, so a different extract of the data stops the Phase 0 notebook with a clear message.

## Files the notebooks write here

Phase 0 writes the analysis datasets (`DS_3F.csv`, `DS_4F.csv`, `DS_4F_ALK.csv`, `DS_5F.csv`) and the frozen validation folds (`fold_indices.json`, `fold_keys_by_dataset.json`).
