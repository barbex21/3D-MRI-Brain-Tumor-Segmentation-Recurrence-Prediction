# RHUH-GBM ET and NECR segmentation

This Valencia research project compares MRI input modalities for enhancing tumor (ET) and non-enhancing core (NECR) segmentation using the Río Hortega University Hospital Glioblastoma (RHUH-GBM) dataset.

## Current experiment audit

The audit contains 13 experiment rows: 10 are verified against paired `config.json` and `metrics.json` files, and 3 NECR rows are reconstructed from the aggregate summary. All 13 selected thresholds are now verified: 10 from metrics JSON, two from individual NECR reports, and the remaining NECR T1ce+T2 threshold (0.80) from the ET-vs-NECR comparison report. Details and provenance are in [Results/experiment_audit.csv](Results/experiment_audit.csv) and [docs/data_dictionary.md](docs/data_dictionary.md).

The largest observed test Dice values in the current audit are ET: 0.6402 with T1ce+FLAIR, and NECR: 0.3770 with T1ce. These are descriptive results from the available runs, not confirmatory evidence that one modality generalizes better. See [docs/results_summary.md](docs/results_summary.md) for limitations.

## Evaluation

Dice is averaged across non-empty target volumes. The decision threshold is selected on validation data and applied unchanged to test evaluation. Recovered rows are identified in the audit.

## Repository contents

- `Code/03_monai_unet_controlled_v10.py`: MONAI/PyTorch 3D U-Net training and evaluation.
- `Results/experiment_audit.csv`: aggregate experiment audit.
- `docs/data_dictionary.md`: table fields, metric calculation, and recovery status.
- `docs/results_summary.md`: descriptive snapshot and limitations.
- `docs/reproducibility.md`: data preparation and example commands.
- `docs/dataset_and_citation.md`: dataset citation and access guidance.
- `requirements.txt`: Python package requirements.

## Data and privacy

This repository does not contain MRI volumes, masks, predictions, clinical tables, patient identifiers, model checkpoints, or credentials. Keep those files in approved, access-controlled storage and outside Git. See [docs/dataset_and_citation.md](docs/dataset_and_citation.md) for the official dataset source and required citation.

## Running the script

The script expects a locally prepared master dataset CSV and imaging data. These are intentionally not included. Example:

```powershell
python Code/03_monai_unet_controlled_v10.py --master-csv data/master_dataset.csv --modalities t1 --epochs 50
```

Review [docs/reproducibility.md](docs/reproducibility.md) and install dependencies in a private environment. The example does not reproduce the original metrics without matching data, preprocessing, environment, and hardware.

## License

No license is declared for this repository. Confirm code and dataset permissions before reuse or redistribution.
