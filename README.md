# Repurchase model choice under limited transaction information and screening capacity

Reproduction materials for the manuscript of the same name, submitted to the
*International Journal of Retail & Distribution Management*.

## Repository contents

```
code/                     Python source (7 files)
configs/                  Frozen experimental configuration
  frozen_selection.json   Frozen model settings and selection outcomes
  protocol_v1.md          Analysis protocol (recorded before test metrics)
  extension_v2.md         Supplementary CDNOW specification
  experiment_plan.csv     Experiment plan
data/
  Online Retail.xlsx      Raw UCI Online Retail transactions (CC BY 4.0)
  provenance.json         Source URL, SHA256 and licence of the raw data
  CDNOW_RETRIEVAL.md      How to obtain the CDNOW archive (not redistributed)
results/                  Key predictions and bootstrap replicates for comparison
  test_predictions.csv                 UCI test predictions
  paired_cluster_bootstrap.csv         UCI 1,000 paired cluster-bootstrap replicates
  cdnow_predictions.csv                CDNOW test predictions
  cdnow_paired_cluster_bootstrap.csv   CDNOW 1,000 replicates
requirements.txt          Python dependencies
requirements.lock.txt     Pinned Windows x64 CPython 3.12 wheels
variable_dictionary.md    Variable definitions and counting rules
```

## Requirements

- 64-bit Python 3.12 (the pipeline asserts Python 3.12).
- Dependencies: `pip install -r requirements.txt`

## Reproduction steps

1. Install 64-bit Python 3.12.
2. Install dependencies: `pip install -r requirements.txt`
3. Obtain the CDNOW archive (see `data/CDNOW_RETRIEVAL.md`) and save it as
   `data/CDNOW_master.zip` with SHA256
   `94081977a983e80dff8edfa58c83a97db33ec2bdd553151987f0dd2f63aaff49`.
   `Online Retail.xlsx` is already included under `data/`.
4. Run the pipeline:
   ```
   python code/pipeline.py all --run-id <reviewer_id>
   ```
5. Compare `results/<reviewer_id>/*.csv` with the reference CSVs in `results/`.

## Data

- **Online Retail**: UCI Machine Learning Repository, DOI 10.24432/C5BW33,
  licence CC BY 4.0. SHA256
  `43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d`.
- **CDNOW**: https://brucehardie.com/datasets/CDNOW_master.zip (not redistributed
  here; see `data/CDNOW_RETRIEVAL.md`).

## Notes

- The pipeline performs a deterministic, single-threaded computation.
- The UCI primary comparison is frozen; the CDNOW analysis is supplementary and
  exploratory.
- No treatment effect, causal sales uplift or marketing ROI is claimed.

## Citation

If you use these materials, please cite the associated manuscript and the data
sources (Chen, 2015; Fader and Hardie, 2013).
