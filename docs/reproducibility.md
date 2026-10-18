# Reproducing an experiment

## Data preparation

Keep the dataset and all patient-level tables outside this repository. The trainer requires a local master CSV with patient_id, phase_id, phase_name, split, mask_path, and one <modality>_path column for each selected modality. The split values must be train, val, or test, and patient-level splits must not overlap. Paths in the CSV must point to local NIfTI images and masks. Never publish this CSV or the NIfTI files.

The supplied code uses ET label 3 and NECR label 1. Confirm these label conventions against the dataset documentation and preprocessing used for each run before retraining.

## Environment

Use a private environment and install compatible packages from requirements.txt. The original runs did not record Python, PyTorch, MONAI, CUDA, NumPy, pandas, or nibabel versions in their metrics files, so their exact software environment cannot be recovered from this audit. PyTorch, torchvision, and torchaudio are intentionally not pinned because the compatible installation depends on the selected CPU/GPU and CUDA build. Select and record a matching PyTorch build before retraining. The repository CI uses Python 3.11 for static validation; this is not evidence that the original runs used Python 3.11.

## Example commands

From the repository root, with a private master CSV at data/master_dataset.csv:

~~~bash
python Code/03_monai_unet_controlled_v10.py \
  --master-csv data/master_dataset.csv \
  --modalities t1ce,flair \
  --target-mask et \
  --epochs 100 \
  --controlled-training \
  --run-name et_t1ce_flair
~~~

For NECR, change --target-mask et to --target-mask necr and use a separate run name. Results are written under experiments/ by default; that directory is ignored by Git because it can contain patient-level manifests and predictions.

New runs write Python, PyTorch, CUDA runtime, MONAI, NumPy, pandas, and nibabel versions into metrics.json. These are example invocations, not a claim that the original runs can be exactly reproduced without the original data, preprocessing, environment, and hardware details.

