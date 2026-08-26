# Cross-Regional PLOS ONE Data Instructions

The cross-regional stress test uses supporting files associated with:

> Alaminos, D., del Castillo, A., and Fernández, M. Á. (2016). “A Global Model
> for Bankruptcy Prediction.” *PLOS ONE*, 11(11), e0166693.
> <https://doi.org/10.1371/journal.pone.0166693>.

The source documentation identifies Standard & Poor's COMPUSTAT as the source
of the underlying firm observations. The workbooks and the reconstructed
firm-level CSV are therefore not redistributed in this repository.

## Procedure

1. Review the publisher page, supporting-file documentation, and all applicable
   terms.
2. If those terms permit your use, explicitly acknowledge this responsibility:

   ```bash
   python 02_code/03_pone_cross_region/download_pone_supporting_files.py \
     --acknowledge-third-party-terms
   ```

3. The helper places the publisher-hosted files in
   `03_data/01_raw/pone_global_2016/`.
4. Build the raw and deduplicated long-form tables:

   ```bash
   python 02_code/03_pone_cross_region/prepare_pone_data.py
   ```

5. Confirm the preparation report records 1,296 raw rows, 1,288 deduplicated
   rows, and eight removed exact duplicates.

The experiment uses the region-specific S1, S2, S3, S5, S6, S7, S11, S12, and
S13 workbooks. Files described as global aggregates are excluded to prevent
source–target overlap.

The `.gitignore` rules exclude all downloaded source files and locally
reconstructed data while retaining the empty directory placeholders.
