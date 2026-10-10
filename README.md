# Estimating BOD and COD from routine water-quality parameters in Dhaka's rivers

> **Status:** manuscript in preparation. This repository shares the first part of the analysis (data preparation, feature selection and baseline validation). The remaining analyses will be released with the publication. Please do not cite or reuse the results yet.

Biochemical and chemical oxygen demand (BOD, COD) are the standard indicators of organic pollution, but they need a 5-day laboratory test (BOD) or wet chemistry (COD). EC, pH and dissolved oxygen (DO) are cheap and routinely measured. The project asks whether these routine parameters can estimate BOD and COD **at a river the model has never seen** and **in a future year**, and whether they add anything **beyond the seasonal cycle**.

The data are monthly monitoring records (2017–2022) from the Department of Environment (DoE), Bangladesh, at 30 stations on five rivers around Dhaka. They are not included here (see `data/README.md`).

## What is in this repository

| Notebook | What it does |
|---|---|
| `phase0-1_data_qc_eda.ipynb` | Integrity checks, data-handling assumptions, complete-case datasets, frozen validation folds, exploratory analysis (clustering, effective sample size, structural-break tests) |
| `phase2_feature_selection.ipynb` | EC vs TDS, calendar terms and transformations under pre-specified decision rules; writes the feature lock used by later phases |
| `phase3_baselines.ipynb` | Naïve and seasonal baselines, linear and tree models, threshold metrics and paired fold comparisons |

`src/` holds the shared code: data loading and integrity checks, fold construction, fold-safe preprocessing, model fitting, evaluation and statistics.

## Design principles

- Leave-one-river-out tests transfer to an unmonitored river; forward-chaining by year tests prediction of a future year. Hyperparameters are tuned in an inner loop that mirrors the outer protocol, and scaling and PCA are refit inside every inner fold, so no information leaks from test data.
- Every model is compared with a month-of-year climatology and a month-only regression.


## Glossary

| Term | Meaning |
|---|---|
| DS-3F, DS-4F, DS-4F-ALK, DS-5F | Complete-case datasets: EC, pH, DO; + TDS; + alkalinity; + TDS and turbidity |
| FS-A … FS-F | Candidate feature sets (`src/config.py`) |
| LOSO-River / LOSO-Station | Leave one river / one station out |
| Forward-Chain (FC-1 … FC-4) | Train on all earlier years, test on the next year |
| Month climatology | Prediction = mean of the training data for that calendar month |
| Lock | The features and dataset fixed in Phase 2 and read by every later phase |

## Author

Tanzir Ahmed Khan. Shared for review; no license is granted (all rights reserved). The full analysis is available on request.
