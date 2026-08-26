# Code Index

| Directory | Purpose |
|---|---|
| `01_core_polish_model` | Main HistGB risk feature, landmark RBF learner, triad threshold policy, and NGO-family optimizers |
| `02_external_validation` | Shared regional-transfer calibration and evaluation backend |
| `03_pone_cross_region` | Publisher-file download helper, data reconstruction, regional execution, and aggregation |
| `04_manuscript_analysis` | Corrected matched internal baselines, final statistical tables, and manuscript figures |
| `05_additional_internal_analysis` | Earlier supporting scripts retained for diagnostic-output lineage |
| `06_environment` | Python dependency specification |

The manuscript's headline tables should be regenerated with
`04_manuscript_analysis`, not with the older scripts in
`05_additional_internal_analysis`. The latter are retained only because several
released feature-stability and supplementary diagnostic files were produced by
them.
