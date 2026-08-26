# Cross-Fitted Risk Fusion for Bankruptcy Prediction

This repository is the reproducibility package for the manuscript
**“Cross-Fitted Risk Fusion With a Triad-Balanced Operating Policy for
Bankruptcy Prediction.”** It contains the code, redistributable data, released
intermediate outputs, final analysis tables, and figures needed to audit the
reported results.

The repository intentionally does **not** contain an old MDPI manuscript or its
superseded figures. The submission source and PDF are maintained separately;
the files here are the computational record cited by the manuscript.

## Start here

1. Read [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for commands.
2. Read [`docs/RESULTS_INDEX.md`](docs/RESULTS_INDEX.md) to map each manuscript
   result to a released file.
3. Read [`03_data/README.md`](03_data/README.md) before obtaining or using data.
4. Run `python tools/verify_release.py` with Python 3.10 or later.

## Repository layout

```text
02_code/
  01_core_polish_model/       Main HistGB–landmark-RBF/PBMSBAINGO implementation
  02_external_validation/     External calibration and evaluation backend
  03_pone_cross_region/       Cross-region preparation, execution, and aggregation
  04_manuscript_analysis/     Corrected matched baselines, tables, tests, and figures
  05_additional_internal_analysis/
                              Supporting/diagnostic analyses retained for audit
  06_environment/             Python dependency specification
03_data/
  01_raw/polish_bankruptcy/   Five UCI ARFF files (CC BY 4.0)
  01_raw/pone_global_2016/    Empty destination for third-party workbooks
  02_processed/pone_global_2016/
                              Empty destination for locally reconstructed data
04_results/
  01_polish_internal/         Original model runs and supporting diagnostics
  02_pone_cross_region/       Per-seed and consolidated regional-transfer outputs
  03_manuscript_tables/       Exact corrected tables used in the manuscript
05_figures/
  manuscript/                 Figures 1–7 used in the revised manuscript
  additional_diagnostics/     Supporting feature/operating-point diagnostics
docs/                         Reproduction, result-lineage, and schema documentation
tools/                        Release verification and checksum-manifest utilities
```

## Headline released results

Under the matched internal protocol, Triad risk fusion increases mean
sensitivity from 0.6131 to 0.6362 relative to HistGB, while MCC, specificity,
precision, and AP decrease and mean F2 is nearly unchanged. Under the tested
fixed operating points, its normalized cost becomes lower than HistGB only for
false-negative/false-positive cost ratios of 10 or more.

In the source-only regional stress test, balanced logistic regression leads AP,
MCC, and balanced accuracy; score fusion leads AUC; and the full model has the
lowest mean Brier score and log loss, but not the lowest ECE. Equal-budget
random search performs similarly to PBMSBAINGO on the main proper scores.

These values can be checked directly in
[`04_results/03_manuscript_tables`](04_results/03_manuscript_tables/).

## Data and redistribution

The five Polish Companies Bankruptcy ARFF files are redistributed under CC BY
4.0 with attribution to Sebastian Tomczak and the UCI Machine Learning
Repository: <https://doi.org/10.24432/C5F600>.

The PLOS ONE supporting workbooks used for the cross-regional experiment are
not redistributed. Their documentation identifies COMPUSTAT-derived firm-level
data. Users must obtain those files from the publisher and comply with the
applicable terms. See [`03_data/PONE_DATA_INSTRUCTIONS.md`](03_data/PONE_DATA_INSTRUCTIONS.md).

## Software environment

Python 3.10 or later is required.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r 02_code/06_environment/requirements.txt
```

## Release integrity

[`FILE_MANIFEST.csv`](FILE_MANIFEST.csv) records the byte size and SHA-256
checksum of every released file except the manifest itself. The largest
internal prediction file is stored as `fair_internal_predictions.csv.gz` to
avoid GitHub's large-file warning; Python, pandas, and common archive tools can
read it directly.

## Citation and license status

Citation metadata are in [`CITATION.cff`](CITATION.cff). Dataset and software
rights are explained in [`LICENSE_NOTICE.md`](LICENSE_NOTICE.md). No software
license has been granted by the authors at this stage; public visibility alone
does not grant permission to reuse the source code.
