# RHUH-GBM ET and NECR segmentation

This repository documents a Valencia research project comparing MRI input modalities for enhancing tumor (ET) and non-enhancing core (NECR) segmentation in the RHUH-GBM collection.

## Experiment audit

The current audit contains 13 experiment rows. Ten rows are verified directly against paired `config.json` and `metrics.json` files. Three NECR rows were recovered from the experiment summary; two thresholds were corroborated by individual reports, while the T1ce+T2 threshold remains unverified and is intentionally blank. See [Results/experiment_audit.csv](Results/experiment_audit.csv) and the [data dictionary](docs/data_dictionary.md) for definitions and limitations.

## Evaluation method

The audited Dice values are full-volume means over non-empty target volumes. The selected threshold is chosen on validation data by maximizing mean Dice; test Dice is evaluated using that selected threshold. Rows recovered from summaries or reports are marked explicitly in the audit.

## Repository contents

- `Code/03_monai_unet_controlled_v10.py`: controlled MONAI/PyTorch 3D U-Net training and evaluation script.
- `Results/experiment_audit.csv`: de-identified, aggregate experiment audit table.
- `docs/data_dictionary.md`: column definitions and provenance.
- `docs/reproducibility.md`: environment and reproducibility notes.
- `requirements.txt`: Python package requirements.

## Data and privacy

This repository does not contain MRI volumes, masks, predictions, clinical tables, patient identifiers, model checkpoints, or credentials. Keep those files in approved, access-controlled storage and outside Git. The project uses the RHUH-GBM dataset; consult the dataset's access and citation requirements before any use or redistribution.

## Running the script

The script expects a locally prepared master dataset CSV and imaging data. These are intentionally not included. Example:

```powershell
python Code/03_monai_unet_controlled_v10.py --master-csv data/master_dataset.csv --modalities t1 --epochs 50
```

Check the script's help output for the supported options and ensure your local environment matches [docs/reproducibility.md](docs/reproducibility.md).

## License

No license is declared for this repository. Confirm code and dataset permissions before reuse or redistribution.
