"""
Canonical MONAI + PyTorch 3D U-Net baseline for GBM ET/NECR segmentation.

Usage example:
    python Code/03_monai_unet_controlled_v10.py ^
      --master-csv data/master_dataset.csv ^
      --modalities t1 ^
      --epochs 50

Smoke test:
    python Code/03_monai_unet_controlled_v10.py ^
      --master-csv data/master_dataset.csv ^
      --modalities t1 ^
      --smoke-test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import warnings
from monai.data import DataLoader, Dataset, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
)
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings(
    "ignore",
    message="Num foregrounds 0.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing.*",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    master_csv: str
    output_root: str = "experiments"

    modalities: tuple[str, ...] = ("t1",)
    target_mask: str = "et"
    label_et: int = 3
    label_necr: int = 1

    seed: int = 42
    epochs: int = 50
    batch_size: int = 1
    num_workers: int = 0
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    amp: bool = True

    patch_size: tuple[int, int, int] = (128, 128, 64)
    samples_per_volume: int = 4
    positive_crop_ratio: float = 0.8
    min_foreground_voxels: int = 15

    roi_size: tuple[int, int, int] = (128, 128, 64)
    overlap: float = 0.5
    sw_batch_size: int = 2

    # None = no resampling. Orientation is still standardized to RAS.
    spacing: tuple[float, float, float] | None = None
    normalize_clip: float = 5.0

    threshold_candidates: tuple[float, ...] = (
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
        0.99,
    )

    early_stopping_patience: int = 20
    model_channels: tuple[int, ...] = (16, 32, 64, 128)
    model_strides: tuple[int, ...] = (2, 2, 2)
    num_res_units: int = 2
    dropout: float = 0.0

    run_name: str = ""
    smoke_test: bool = False
    overfit_volumes: int = 0
    overfit_learning_rate: float = 1e-3
    overfit_weight_decay: float = 0.0
    controlled_training: bool = False
    controlled_learning_rate: float = 1e-4
    single_batch_overfit_steps: int = 400


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ---------------------------------------------------------------------------
# CSV and metadata validation
# ---------------------------------------------------------------------------


def parse_csv_tuple(text: str, expected_length: int, cast: type) -> tuple:
    values = tuple(cast(item.strip()) for item in text.split(",") if item.strip())

    if len(values) != expected_length:
        raise ValueError(
            f"Expected {expected_length} values, received {text!r}."
        )

    return values


def read_master_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Master CSV not found: {path}")

    # Explicitly detect the malformed one-column header problem.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)

    if not header:
        raise ValueError(f"Master CSV is empty: {path}")

    if len(header) == 1 and "," in header[0]:
        raise ValueError(
            "Malformed master CSV: its header is stored as one quoted field. "
            "Run the corrected 01_dataset_index_run.py again."
        )

    dataframe = pd.read_csv(path, encoding="utf-8-sig")
    dataframe.columns = [
        str(column).replace("\ufeff", "").strip()
        for column in dataframe.columns
    ]

    required = {
        "patient_id",
        "phase_id",
        "phase_name",
        "split",
        "mask_path",
    }

    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise KeyError(f"Master CSV missing required columns: {missing}")

    dataframe["patient_id"] = (
        dataframe["patient_id"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )

    dataframe["split"] = (
        dataframe["split"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )

    if dataframe["patient_id"].eq("").any():
        raise ValueError("Master CSV contains empty patient_id values.")

    if not set(dataframe["split"].unique()).issubset({"train", "val", "test"}):
        raise ValueError(
            "The split column may contain only train, val, or test values."
        )

    return dataframe.reset_index(drop=True)


def validate_patient_splits(dataframe: pd.DataFrame) -> None:
    groups = {
        split: set(
            dataframe.loc[
                dataframe["split"] == split,
                "patient_id",
            ]
        )
        for split in ("train", "val", "test")
    }

    if not groups["train"] or not groups["val"] or not groups["test"]:
        raise ValueError(
            "Train, validation, and test splits must all contain patients."
        )

    overlaps = {
        "train-val": groups["train"] & groups["val"],
        "train-test": groups["train"] & groups["test"],
        "val-test": groups["val"] & groups["test"],
    }

    invalid = {
        name: patient_ids
        for name, patient_ids in overlaps.items()
        if patient_ids
    }

    if invalid:
        raise AssertionError(
            "Patient-level leakage detected between splits: "
            f"{ {name: len(ids) for name, ids in invalid.items()} }"
        )

    print(
        "Patient counts:",
        {name: len(patient_ids) for name, patient_ids in groups.items()},
    )

    print(
        "Volume counts:",
        dataframe["split"].value_counts().sort_index().to_dict(),
    )


def validate_requested_modalities(
    dataframe: pd.DataFrame,
    modalities: tuple[str, ...],
) -> None:
    required_columns = ["mask_path"]

    for modality in modalities:
        required_columns.append(f"{modality}_path")

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            "Requested modalities are not present in master CSV: "
            f"{missing_columns}"
        )

    missing_files: list[str] = []

    for _, row in dataframe.iterrows():
        for column in required_columns:
            path = Path(str(row[column]))

            if not path.is_file():
                missing_files.append(
                    f"{row['patient_id']} phase={row['phase_id']}: {path}"
                )

    if missing_files:
        raise FileNotFoundError(
            "Required NIfTI files are missing:\n"
            + "\n".join(missing_files[:30])
        )


def close_affine(
    affine_a: np.ndarray,
    affine_b: np.ndarray,
    tolerance: float = 1e-4,
) -> bool:
    return bool(np.allclose(affine_a, affine_b, atol=tolerance, rtol=0.0))


def validate_nifti_spatial_integrity(
    dataframe: pd.DataFrame,
    modalities: tuple[str, ...],
    allowed_labels: set[int],
) -> pd.DataFrame:
    """
    Validates every selected image/mask pair before training.

    This does not silently repair spatial disagreements. Such data require
    explicit preprocessing decisions.
    """
    reports: list[dict[str, Any]] = []
    problems: list[str] = []

    for _, row in dataframe.iterrows():
        patient_id = str(row["patient_id"])
        phase_id = str(row["phase_id"])

        mask_path = Path(str(row["mask_path"]))
        mask_nii = nib.load(str(mask_path))
        mask_data = np.asanyarray(mask_nii.dataobj)

        if mask_data.ndim != 3:
            problems.append(
                f"{patient_id}, phase {phase_id}: mask is not 3D."
            )
            continue

        mask_labels = set(np.unique(mask_data).astype(int).tolist())

        if not mask_labels.issubset(allowed_labels):
            problems.append(
                f"{patient_id}, phase {phase_id}: unsupported labels "
                f"{sorted(mask_labels - allowed_labels)}."
            )

        for modality in modalities:
            image_path = Path(str(row[f"{modality}_path"]))
            image_nii = nib.load(str(image_path))
            image_data = np.asanyarray(image_nii.dataobj)

            if image_data.ndim != 3:
                problems.append(
                    f"{patient_id}, phase {phase_id}, {modality}: image is not 3D."
                )
                continue

            if image_nii.shape != mask_nii.shape:
                problems.append(
                    f"{patient_id}, phase {phase_id}, {modality}: "
                    f"image shape {image_nii.shape} != mask shape {mask_nii.shape}."
                )

            if not close_affine(image_nii.affine, mask_nii.affine):
                problems.append(
                    f"{patient_id}, phase {phase_id}, {modality}: "
                    "image/mask affines differ."
                )

            if not np.isfinite(image_data).all():
                problems.append(
                    f"{patient_id}, phase {phase_id}, {modality}: "
                    "image contains NaN or Inf."
                )

            reports.append(
                {
                    "patient_id": patient_id,
                    "phase_id": phase_id,
                    "phase_name": row["phase_name"],
                    "split": row["split"],
                    "modality": modality,
                    "shape": str(tuple(image_nii.shape)),
                    "spacing": str(tuple(image_nii.header.get_zooms()[:3])),
                    "orientation": "".join(nib.aff2axcodes(image_nii.affine)),
                    "image_min": float(np.nanmin(image_data)),
                    "image_max": float(np.nanmax(image_data)),
                    "image_mean": float(np.nanmean(image_data)),
                    "image_std": float(np.nanstd(image_data)),
                    "mask_foreground_voxels": int(np.count_nonzero(mask_data)),
                    "mask_labels": str(sorted(mask_labels)),
                }
            )

    if problems:
        raise RuntimeError(
            "Dataset spatial-integrity validation failed:\n"
            + "\n".join(problems[:50])
        )

    return pd.DataFrame(reports)


# ---------------------------------------------------------------------------
# MONAI transforms
# ---------------------------------------------------------------------------


class FiniteNonzeroZScoreD(MapTransform):
    """Per-channel z-score on non-zero finite voxels while preserving background=0."""

    def __init__(
        self,
        keys: Iterable[str],
        clip_value: float,
    ) -> None:
        super().__init__(keys)
        self.clip_value = clip_value

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        output = dict(data)

        for key in self.keys:
            image = torch.as_tensor(output[key]).float().clone()

            # Images are channel-first after EnsureChannelFirstd.
            for channel_index in range(image.shape[0]):
                channel = image[channel_index]

                finite = torch.isfinite(channel)
                nonzero = channel != 0
                valid = finite & nonzero

                # Never propagate NaN/Inf.
                channel[~finite] = 0.0

                if valid.any():
                    values = channel[valid]
                    mean = values.mean()
                    std = values.std(unbiased=False)

                    if torch.isfinite(std) and std >= 1e-6:
                        channel[valid] = (values - mean) / std
                    else:
                        channel[valid] = values - mean

                # Critical: background stays exactly zero.
                channel[~nonzero] = 0.0

                channel = torch.clamp(
                    channel,
                    min=-self.clip_value,
                    max=self.clip_value,
                )

                # Re-assert background after clipping.
                channel[~nonzero] = 0.0

                image[channel_index] = channel

            output[key] = image

        return output


class BuildTargetMaskD(MapTransform):
    """Converts multiclass mask to the configured binary target."""

    def __init__(
        self,
        keys: Iterable[str],
        target_mask: str,
        label_et: int,
        label_necr: int,
    ) -> None:
        super().__init__(keys)
        self.target_mask = target_mask.lower()
        self.label_et = label_et
        self.label_necr = label_necr

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        output = dict(data)

        for key in self.keys:
            label = torch.as_tensor(output[key])

            if self.target_mask == "et":
                target = label == self.label_et
            elif self.target_mask == "necr":
                target = label == self.label_necr
            elif self.target_mask == "combined":
                target = (label == self.label_et) | (label == self.label_necr)
            else:
                raise ValueError(
                    f"Unsupported target mask: {self.target_mask}"
                )

            output[key] = target.float()

        return output


def build_transforms(cfg: Config, training: bool) -> Compose:
    keys = ("image", "label")
    interpolation = ("bilinear", "nearest")

    transforms: list[Any] = [
        LoadImaged(
            keys=keys,
            reader="NibabelReader",
            image_only=False,
        ),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
    ]

    if cfg.spacing is not None:
        transforms.append(
            Spacingd(
                keys=keys,
                pixdim=cfg.spacing,
                mode=interpolation,
            )
        )

    transforms.extend(
        [
            FiniteNonzeroZScoreD(
                keys=("image",),
                clip_value=cfg.normalize_clip,
            ),
            BuildTargetMaskD(
                keys=("label",),
                target_mask=cfg.target_mask,
                label_et=cfg.label_et,
                label_necr=cfg.label_necr,
            ),
        ]
    )

    if training:
        transforms.append(
            RandCropByPosNegLabeld(
                keys=keys,
                label_key="label",
                spatial_size=cfg.patch_size,
                pos=cfg.positive_crop_ratio,
                neg=1.0 - cfg.positive_crop_ratio,
                num_samples=cfg.samples_per_volume,
                image_key="image",
                image_threshold=0.0,
                allow_smaller=False,
            )
        )

        # Diagnostic overfit mode must isolate the learnability of the
        # data/target/model pipeline, so random augmentation is disabled.
        if not cfg.overfit_volumes and not cfg.controlled_training:
            transforms.extend(
                [
                    RandFlipd(keys=keys, spatial_axis=0, prob=0.5),
                    RandFlipd(keys=keys, spatial_axis=1, prob=0.5),
                    RandFlipd(keys=keys, spatial_axis=2, prob=0.5),
                    RandScaleIntensityd(
                        keys=("image",),
                        factors=0.10,
                        prob=0.20,
                    ),
                    RandShiftIntensityd(
                        keys=("image",),
                        offsets=0.10,
                        prob=0.20,
                    ),
                    RandGaussianNoised(
                        keys=("image",),
                        std=0.02,
                        prob=0.15,
                    ),
                ]
            )

    return Compose(transforms)


# ---------------------------------------------------------------------------
# Data records and patch monitoring
# ---------------------------------------------------------------------------

def report_empty_target_volumes(
    dataframe: pd.DataFrame,
    cfg: Config,
    output_path: Path,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for _, row in dataframe.iterrows():
        mask = np.asanyarray(
            nib.load(str(row["mask_path"])).dataobj
        )

        if cfg.target_mask == "et":
            foreground = int((mask == cfg.label_et).sum())
        elif cfg.target_mask == "necr":
            foreground = int((mask == cfg.label_necr).sum())
        elif cfg.target_mask == "combined":
            foreground = int(
                np.isin(mask, [cfg.label_et, cfg.label_necr]).sum()
            )
        else:
            raise ValueError(f"Unknown target mask: {cfg.target_mask}")

        records.append(
            {
                "patient_id": str(row["patient_id"]),
                "phase_id": str(row["phase_id"]),
                "phase_name": str(row["phase_name"]),
                "split": str(row["split"]),
                "target_foreground_voxels": foreground,
                "empty_target": foreground == 0,
            }
        )

    report = pd.DataFrame(records)
    report.to_csv(output_path, index=False)

    print(
        "Empty target volumes by split:",
        report.groupby("split")["empty_target"].sum().to_dict(),
    )

    return report

def make_records(
    dataframe: pd.DataFrame,
    modalities: tuple[str, ...],
    split: str,
) -> list[dict[str, Any]]:
    split_dataframe = dataframe.loc[dataframe["split"] == split].copy()

    records: list[dict[str, Any]] = []

    for _, row in split_dataframe.iterrows():
        records.append(
            {
                "image": [
                    str(row[f"{modality}_path"])
                    for modality in modalities
                ],
                "label": str(row["mask_path"]),
                "patient_id": str(row["patient_id"]),
                "phase_id": str(row["phase_id"]),
                "phase_name": str(row["phase_name"]),
                "split": split,
            }
        )

    if not records:
        raise RuntimeError(f"No records in split {split!r}.")

    return records


def patch_statistics(
    loader: DataLoader,
    min_foreground_voxels: int,
    limit: int = 256,
) -> dict[str, Any]:
    """
    Detailed sanity check of the training patch sampler.

    This function DOES NOT train the model.
    It only inspects patches produced by the training DataLoader.

    In addition to foreground statistics, it reports which
    patient/phase combinations produced empty patches.
    """

    foreground_counts: list[int] = []

    zero_foreground_records: list[dict[str, str]] = []

    checked = 0

    for batch in loader:
        labels = batch["label"]

        # Metadata may be collated as lists/tuples by MONAI.
        patient_ids = batch.get("patient_id")
        phase_ids = batch.get("phase_id")
        phase_names = batch.get("phase_name")

        for index, label in enumerate(labels):
            foreground_voxels = int(label.sum().item())

            foreground_counts.append(foreground_voxels)
            checked += 1

            if foreground_voxels == 0:
                patient_id = (
                    str(patient_ids[index])
                    if patient_ids is not None
                    else "unknown"
                )

                phase_id = (
                    str(phase_ids[index])
                    if phase_ids is not None
                    else "unknown"
                )

                phase_name = (
                    str(phase_names[index])
                    if phase_names is not None
                    else "unknown"
                )

                zero_foreground_records.append(
                    {
                        "patient_id": patient_id,
                        "phase_id": phase_id,
                        "phase_name": phase_name,
                    }
                )

            if checked >= limit:
                break

        if checked >= limit:
            break

    if not foreground_counts:
        raise RuntimeError(
            "No patches were produced by the training loader."
        )

    # --------------------------------------------------------------
    # Foreground categories
    # --------------------------------------------------------------

    zero_foreground = sum(
        count == 0
        for count in foreground_counts
    )

    weak_foreground = sum(
        0 < count < min_foreground_voxels
        for count in foreground_counts
    )

    valid_positive = sum(
        count >= min_foreground_voxels
        for count in foreground_counts
    )

    # Additional useful bins
    fg_1_99 = sum(
        1 <= count < 100
        for count in foreground_counts
    )

    fg_100_999 = sum(
        100 <= count < 1000
        for count in foreground_counts
    )

    fg_1000_plus = sum(
        count >= 1000
        for count in foreground_counts
    )

    # --------------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------------

    mean_foreground = float(np.mean(foreground_counts))
    median_foreground = float(np.median(foreground_counts))
    min_foreground = int(np.min(foreground_counts))
    max_foreground = int(np.max(foreground_counts))

    # --------------------------------------------------------------
    # Zero-foreground patient/phase summary
    # --------------------------------------------------------------

    zero_summary: dict[str, int] = {}

    for record in zero_foreground_records:
        key = (
            f"{record['patient_id']} | "
            f"{record['phase_id']} | "
            f"{record['phase_name']}"
        )

        zero_summary[key] = zero_summary.get(key, 0) + 1

    # --------------------------------------------------------------
    # Print detailed sanity report
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("PATCH SAMPLING SANITY CHECK")
    print("=" * 72)

    print(f"Checked patches              : {checked}")
    print(f"Minimum foreground threshold: {min_foreground_voxels}")
    print()

    print("FOREGROUND DISTRIBUTION")
    print("-" * 72)

    print(
        f"Foreground = 0              : "
        f"{zero_foreground:4d} "
        f"({zero_foreground / checked:.1%})"
    )

    print(
        f"Foreground 1-99             : "
        f"{fg_1_99:4d} "
        f"({fg_1_99 / checked:.1%})"
    )

    print(
        f"Foreground 100-999         : "
        f"{fg_100_999:4d} "
        f"({fg_100_999 / checked:.1%})"
    )

    print(
        f"Foreground >=1000          : "
        f"{fg_1000_plus:4d} "
        f"({fg_1000_plus / checked:.1%})"
    )

    print()

    print("TRAINING POSITIVE / NEGATIVE")
    print("-" * 72)

    print(
        f"Valid positive patches      : "
        f"{valid_positive:4d} "
        f"({valid_positive / checked:.1%})"
    )

    print(
        f"Weak foreground patches     : "
        f"{weak_foreground:4d} "
        f"({weak_foreground / checked:.1%})"
    )

    print(
        f"Zero-foreground patches     : "
        f"{zero_foreground:4d} "
        f"({zero_foreground / checked:.1%})"
    )

    print()

    print("FOREGROUND VOXEL STATISTICS")
    print("-" * 72)

    print(f"Mean                        : {mean_foreground:.1f}")
    print(f"Median                      : {median_foreground:.1f}")
    print(f"Minimum                     : {min_foreground}")
    print(f"Maximum                     : {max_foreground}")

    print()

    if zero_summary:
        print("ZERO-FOREGROUND PATCH SOURCES")
        print("-" * 72)

        for key, count in sorted(zero_summary.items()):
            print(f"{count:3d} patch(es) | {key}")
    else:
        print("ZERO-FOREGROUND PATCH SOURCES")
        print("-" * 72)
        print("None.")

    print("=" * 72)
    print()

    return {
        "checked_patches": checked,
        "min_foreground_voxels": min_foreground_voxels,

        "positive_patches_meeting_threshold": valid_positive,
        "weak_foreground_patches": weak_foreground,
        "negative_patches": zero_foreground,

        "actual_positive_fraction": (
            valid_positive / checked
        ),
        "actual_negative_fraction": (
            zero_foreground / checked
        ),

        "zero_foreground_patches": zero_foreground,

        "foreground_1_99": fg_1_99,
        "foreground_100_999": fg_100_999,
        "foreground_1000_plus": fg_1000_plus,

        "foreground_voxel_mean": mean_foreground,
        "foreground_voxel_median": median_foreground,
        "foreground_voxel_min": min_foreground,
        "foreground_voxel_max": max_foreground,

        "zero_foreground_sources": zero_summary,
    }
# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def binary_metrics(
    probability: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    prediction = probability >= threshold
    target = target > 0.5

    tp = int((prediction & target).sum().item())
    fp = int((prediction & ~target).sum().item())
    fn = int((~prediction & target).sum().item())

    predicted_voxels = int(prediction.sum().item())
    target_voxels = int(target.sum().item())

    # Policy for empty target:
    # - exclude from primary Dice, precision, recall averages
    # - report empty counts and false-positive voxels separately
    if target_voxels == 0:
        return {
            "dice": math.nan,
            "precision": math.nan,
            "recall": math.nan,
            "predicted_voxels": predicted_voxels,
            "target_voxels": target_voxels,
            "false_positive_voxels": fp,
            "empty_target": 1.0,
            "predicted_empty": float(predicted_voxels == 0),
            "false_positive_empty_case": float(predicted_voxels > 0),
        }

    epsilon = 1e-6
    dice = (2.0 * tp + epsilon) / (
        2.0 * tp + fp + fn + epsilon
    )
    precision = (tp + epsilon) / (tp + fp + epsilon)
    recall = (tp + epsilon) / (tp + fn + epsilon)

    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "predicted_voxels": predicted_voxels,
        "target_voxels": target_voxels,
        "false_positive_voxels": fp,
        "empty_target": 0.0,
        "predicted_empty": float(predicted_voxels == 0),
        "false_positive_empty_case": 0.0,
    }


def summarize_patient_metrics(
    patient_dataframe: pd.DataFrame,
) -> dict[str, float]:
    nonempty = patient_dataframe.loc[
        patient_dataframe["target_voxels"] > 0
    ]

    if nonempty.empty:
        raise RuntimeError(
            "No non-empty target volumes are available for primary Dice."
        )

    return {
        "dice_mean": float(nonempty["dice"].mean()),
        "dice_std": float(nonempty["dice"].std(ddof=0)),
        "dice_median": float(nonempty["dice"].median()),
        "dice_min": float(nonempty["dice"].min()),
        "dice_max": float(nonempty["dice"].max()),
        "precision_mean": float(nonempty["precision"].mean()),
        "recall_mean": float(nonempty["recall"].mean()),
        "empty_target_cases": int(
            patient_dataframe["empty_target"].sum()
        ),
        "nonempty_target_cases": int(len(nonempty)),
        "predicted_empty_cases": int(
            patient_dataframe["predicted_empty"].sum()
        ),
        "false_positive_empty_cases": int(
            patient_dataframe["false_positive_empty_case"].sum()
        ),
        "false_positive_voxels": int(
            patient_dataframe["false_positive_voxels"].sum()
        ),
    }


# ---------------------------------------------------------------------------
# Model, training and full-volume inference
# ---------------------------------------------------------------------------


def summarize_by_phase(
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Return phase-wise primary Dice/precision/recall summaries."""
    output: dict[str, dict[str, float]] = {}

    for phase_name, group in dataframe.groupby("phase_name"):
        nonempty = group.loc[group["target_voxels"] > 0]
        if nonempty.empty:
            output[str(phase_name)] = {
                "dice_mean": math.nan,
                "precision_mean": math.nan,
                "recall_mean": math.nan,
                "nonempty_cases": 0.0,
                "empty_cases": float(len(group)),
            }
            continue

        output[str(phase_name)] = {
            "dice_mean": float(nonempty["dice"].mean()),
            "precision_mean": float(nonempty["precision"].mean()),
            "recall_mean": float(nonempty["recall"].mean()),
            "nonempty_cases": float(len(nonempty)),
            "empty_cases": float((group["target_voxels"] == 0).sum()),
        }

    return output


def create_model(cfg: Config) -> torch.nn.Module:
    return UNet(
        spatial_dims=3,
        in_channels=len(cfg.modalities),
        out_channels=1,
        channels=cfg.model_channels,
        strides=cfg.model_strides,
        num_res_units=cfg.num_res_units,
        dropout=cfg.dropout,
    )


def create_experiment_directory(cfg: Config) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modality_tag = "-".join(cfg.modalities)
    name = cfg.run_name or f"monai_unet_{modality_tag}_{cfg.target_mask}"

    directory = Path(cfg.output_root) / f"{stamp}_{name}"

    if directory.exists():
        raise FileExistsError(
            f"Experiment directory already exists: {directory}"
        )

    directory.mkdir(parents=True, exist_ok=False)
    (directory / "tensorboard").mkdir()
    return directory


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_validation_dice: float,
    selected_threshold: float,
    cfg: Config,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_validation_dice": best_validation_dice,
            "selected_threshold": selected_threshold,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": asdict(cfg),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def soft_dice_score_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> float:
    """Batch mean soft Dice score at training time (threshold-free)."""
    probability = torch.sigmoid(logits)
    target = target.float()

    dims = tuple(range(2, target.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return float(dice.mean().detach().item())


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: Config,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    losses: list[float] = []
    patch_dice_scores: list[float] = []

    for step, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=cfg.amp and device.type == "cuda",
        ):
            logits = model(image)
            loss = criterion(logits, label)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch={epoch}, step={step}; "
                f"image range=({image.min().item():.4g}, "
                f"{image.max().item():.4g}), "
                f"foreground voxels={int(label.sum().item())}."
            )

        patch_dice_scores.append(
            soft_dice_score_from_logits(logits, label)
        )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=cfg.grad_clip_norm,
        )

        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                f"Non-finite gradients at epoch={epoch}, step={step}."
            )

        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.detach().item()))

    if not losses:
        raise RuntimeError("Training epoch had no batches.")

    return float(np.mean(losses)), float(np.mean(patch_dice_scores))


@torch.no_grad()
def evaluate_full_volumes(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    threshold: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Full-volume validation/test.

    threshold=None means validation threshold search. Test must always receive
    the threshold chosen by the best validation checkpoint.
    """
    model.eval()
    results_by_threshold: dict[float, list[dict[str, Any]]] = {
        candidate: []
        for candidate in cfg.threshold_candidates
    }

    fixed_results: list[dict[str, Any]] = []

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        logits = sliding_window_inference(
            inputs=image,
            roi_size=cfg.roi_size,
            sw_batch_size=cfg.sw_batch_size,
            predictor=model,
            overlap=cfg.overlap,
            mode="gaussian",
        )

        probability = torch.sigmoid(logits).detach().cpu()
        label = label.detach().cpu()

        patient_id = str(batch["patient_id"][0])
        phase_id = str(batch["phase_id"][0])
        phase_name = str(batch["phase_name"][0])

        candidates = (
            (threshold,)
            if threshold is not None
            else cfg.threshold_candidates
        )

        for candidate in candidates:
            metrics = binary_metrics(
                probability=probability[0, 0],
                target=label[0, 0],
                threshold=candidate,
            )

            record = {
                "patient_id": patient_id,
                "phase_id": phase_id,
                "phase_name": phase_name,
                "threshold": candidate,
                **metrics,
            }

            if threshold is None:
                results_by_threshold[candidate].append(record)
            else:
                fixed_results.append(record)

    if threshold is not None:
        patient_dataframe = pd.DataFrame(fixed_results)
        summary = summarize_patient_metrics(patient_dataframe)
        summary["phase_summary"] = summarize_by_phase(
            patient_dataframe
        )
        return summary, patient_dataframe

    threshold_summaries: list[dict[str, float]] = []

    for candidate, records in results_by_threshold.items():
        patient_dataframe = pd.DataFrame(records)
        summary = summarize_patient_metrics(patient_dataframe)
        threshold_summaries.append(
            {
                "threshold": candidate,
                "dice_mean": summary["dice_mean"],
                "precision_mean": summary["precision_mean"],
                "recall_mean": summary["recall_mean"],
            }
        )

    sweep = pd.DataFrame(threshold_summaries)

    # Only validation Dice determines the operating threshold.
    best_row = sweep.loc[sweep["dice_mean"].idxmax()]
    selected_threshold = float(best_row["threshold"])

    selected_dataframe = pd.DataFrame(
        results_by_threshold[selected_threshold]
    )
    selected_summary = summarize_patient_metrics(selected_dataframe)
    selected_summary["selected_threshold"] = selected_threshold
    selected_summary["threshold_sweep"] = threshold_summaries
    selected_summary["phase_summary"] = summarize_by_phase(
        selected_dataframe
    )

    return selected_summary, selected_dataframe


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def run_smoke_test(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: Config,
    device: torch.device,
    experiment_dir: Path,
) -> None:
    print("Running smoke test...")

    batch = next(iter(train_loader))
    image = batch["image"].to(device)
    label = batch["label"].to(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(
        device_type=device.type,
        enabled=cfg.amp and device.type == "cuda",
    ):
        logits = model(image)
        loss = criterion(logits, label)

    if logits.shape != label.shape:
        raise AssertionError(
            f"Model output shape {logits.shape} != label shape {label.shape}."
        )

    if not torch.isfinite(loss):
        raise FloatingPointError("Smoke-test loss is not finite.")

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        cfg.grad_clip_norm,
    )
    scaler.step(optimizer)
    scaler.update()

    # Full-volume sliding-window inference on one validation volume.
    validation_batch = next(iter(validation_loader))
    validation_image = validation_batch["image"].to(device)

    with torch.no_grad():
        output = sliding_window_inference(
            validation_image,
            roi_size=cfg.roi_size,
            sw_batch_size=1,
            predictor=model,
            overlap=cfg.overlap,
            mode="gaussian",
        )

    if output.shape != validation_image.shape[:1] + (1,) + validation_image.shape[2:]:
        raise AssertionError(
            "Sliding-window output shape differs from input spatial shape."
        )

    checkpoint_path = experiment_dir / "smoke_test_checkpoint.pt"

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=0,
        best_validation_dice=0.0,
        selected_threshold=0.5,
        cfg=cfg,
    )

    checkpoint = load_checkpoint(checkpoint_path, model, device)

    if checkpoint["selected_threshold"] != 0.5:
        raise AssertionError("Checkpoint threshold save/load failed.")

    print("Smoke test passed.")


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="MONAI 3D U-Net GBM segmentation baseline."
    )

    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--output-root", default="experiments")
    parser.add_argument("--modalities", default="t1")
    parser.add_argument(
        "--target-mask",
        choices=("et", "necr", "combined"),
        default="et",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-size", default="128,128,64")
    parser.add_argument("--samples-per-volume", type=int, default=4)
    parser.add_argument("--positive-crop-ratio", type=float, default=0.8)
    parser.add_argument("--roi-size", default="128,128,64")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--sw-batch-size", type=int, default=2)
    parser.add_argument(
        "--spacing",
        default=None,
        help="Optional target voxel spacing, for example: 1.0,1.0,1.0",
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overfit-volumes", type=int, default=0)
    parser.add_argument(
        "--controlled-training",
        action="store_true",
        help=(
            "Controlled diagnostic: train only on target-positive training volumes, "
            "use positive-only crops, disable augmentation, and use the "
            "diagnostic optimizer."
        ),
    )
    parser.add_argument(
        "--controlled-learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate used only in --controlled-training mode.",
    )
    parser.add_argument(
        "--single-batch-overfit-steps",
        type=int,
        default=400,
        help="Diagnostic: repeatedly train on one fixed training batch.",
    )
    parser.add_argument("--no-amp", action="store_true")

    arguments = parser.parse_args()

    modalities = tuple(
        modality.strip().lower()
        for modality in arguments.modalities.split(",")
        if modality.strip()
    )

    if not modalities:
        raise ValueError("At least one modality must be selected.")

    if not 0.0 <= arguments.positive_crop_ratio <= 1.0:
        raise ValueError("--positive-crop-ratio must be within [0, 1].")

    if not 0.0 <= arguments.overlap < 1.0:
        raise ValueError("--overlap must be within [0, 1).")

    if arguments.controlled_learning_rate <= 0:
        raise ValueError("--controlled-learning-rate must be > 0.")

    spacing = (
        parse_csv_tuple(arguments.spacing, 3, float)
        if arguments.spacing
        else None
    )

    return Config(
        master_csv=arguments.master_csv,
        output_root=arguments.output_root,
        modalities=modalities,
        target_mask=arguments.target_mask,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        seed=arguments.seed,
        patch_size=parse_csv_tuple(arguments.patch_size, 3, int),
        roi_size=parse_csv_tuple(arguments.roi_size, 3, int),
        samples_per_volume=arguments.samples_per_volume,
        positive_crop_ratio=arguments.positive_crop_ratio,
        overlap=arguments.overlap,
        sw_batch_size=arguments.sw_batch_size,
        spacing=spacing,
        run_name=arguments.run_name,
        smoke_test=arguments.smoke_test,
        overfit_volumes=arguments.overfit_volumes,
        controlled_training=arguments.controlled_training,
        controlled_learning_rate=arguments.controlled_learning_rate,
        single_batch_overfit_steps=arguments.single_batch_overfit_steps,
        amp=not arguments.no_amp,
    )



def run_single_batch_overfit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: Config,
    device: torch.device,
) -> None:
    """
    Hard sanity test:
    repeatedly train on exactly one fixed patch batch.

    No validation, sliding-window inference, random crop, or changing batch.
    If this cannot overfit, the issue is in the model/loss/input pipeline.
    """
    batch = next(iter(train_loader))
    image = batch["image"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)

    print()
    print("=" * 72)
    print("SINGLE-BATCH OVERFIT SANITY TEST")
    print("=" * 72)
    print(f"Image shape         : {tuple(image.shape)}")
    print(f"Label shape         : {tuple(label.shape)}")
    print(f"Foreground voxels   : {int(label.sum().item())}")
    print(f"Image min/max       : {image.min().item():.4f} / {image.max().item():.4f}")
    print(f"Steps               : {cfg.single_batch_overfit_steps}")
    print(f"Learning rate       : {optimizer.param_groups[0]['lr']:.2e}")
    print("Augmentation        : disabled")
    print("Loss                : DiceCE")
    print()

    model.train()

    best_dice = -1.0

    for step in range(1, cfg.single_batch_overfit_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=False,
        ):
            logits = model(image)
            loss = criterion(logits, label)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at single-batch step {step}: "
                f"{loss.item()}"
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.grad_clip_norm,
        )

        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"Non-finite gradients at single-batch step {step}."
            )

        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 20 == 0 or step == cfg.single_batch_overfit_steps:
            with torch.no_grad():
                diagnostic_logits = model(image)
                probability = torch.sigmoid(diagnostic_logits)
                dims = tuple(range(2, label.ndim))
                intersection = (probability * label).sum(dim=dims)
                denominator = probability.sum(dim=dims) + label.sum(dim=dims)
                soft_dice = (
                    (2.0 * intersection + 1.0)
                    / (denominator + 1.0)
                ).mean().item()

                prediction = probability >= 0.5
                target_bool = label > 0.5
                tp = (prediction & target_bool).sum().item()
                fp = (prediction & ~target_bool).sum().item()
                fn = (~prediction & target_bool).sum().item()
                hard_dice = (
                    (2.0 * tp + 1e-6)
                    / (2.0 * tp + fp + fn + 1e-6)
                )

                best_dice = max(best_dice, hard_dice)

                print(
                    f"Step {step:04d}/{cfg.single_batch_overfit_steps} | "
                    f"loss={loss.item():.4f} | "
                    f"soft_dice={soft_dice:.4f} | "
                    f"hard_dice={hard_dice:.4f} | "
                    f"pred_voxels={int(prediction.sum().item())} | "
                    f"mean_prob={probability.mean().item():.4f} | "
                    f"logit=({diagnostic_logits.min().item():.3f},"
                    f"{diagnostic_logits.max().item():.3f})"
                )

    print()
    print(f"Best hard Dice: {best_dice:.4f}")
    print("=" * 72)

    if best_dice < 0.90:
        print(
            "RESULT: single-batch overfit FAILED. "
            "Do not tune the full dataset yet."
        )
    else:
        print(
            "RESULT: single-batch overfit PASSED. "
            "The model/loss/input pipeline can memorize one batch."
        )

def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    set_determinism(seed=cfg.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Modalities: {cfg.modalities}")
    print(f"Target: {cfg.target_mask}")
    if cfg.controlled_training:
        print(
            "Training mode: CONTROLLED "
            "(non-empty training volumes, positive-only crops, no augmentation)"
        )

    master_dataframe = read_master_csv(Path(cfg.master_csv))
    validate_patient_splits(master_dataframe)
    validate_requested_modalities(master_dataframe, cfg.modalities)

    experiment_dir = create_experiment_directory(cfg)

    spatial_report = validate_nifti_spatial_integrity(
        dataframe=master_dataframe,
        modalities=cfg.modalities,
        allowed_labels={0, 1, 2, 3},
    )

    spatial_report.to_csv(
        experiment_dir / "dataset_sanity_report.csv",
        index=False,
    )

    report_empty_target_volumes(
        dataframe=master_dataframe,
        cfg=cfg,
        output_path=experiment_dir / "target_foreground_report.csv",
    )

    with (experiment_dir / "config.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(asdict(cfg), handle, indent=2)

    master_dataframe.to_csv(
        experiment_dir / "dataset_manifest.csv",
        index=False,
    )

    train_dataframe = master_dataframe.loc[
        master_dataframe["split"] == "train"
    ].copy()

    if cfg.overfit_volumes:
        # Select N non-empty patients for the diagnostic.
        # The argument counts patients, not individual phase volumes.
        patient_foreground: list[tuple[str, int]] = []

        for patient_id, patient_df in train_dataframe.groupby("patient_id"):
            patient_max_foreground = 0

            for _, row in patient_df.iterrows():
                mask = np.asanyarray(
                    nib.load(str(row["mask_path"])).dataobj
                )
                if cfg.target_mask == "et":
                    foreground_voxels = int((mask == cfg.label_et).sum())
                elif cfg.target_mask == "necr":
                    foreground_voxels = int((mask == cfg.label_necr).sum())
                else:
                    foreground_voxels = int(
                        np.isin(mask, [cfg.label_et, cfg.label_necr]).sum()
                    )
                patient_max_foreground = max(
                    patient_max_foreground,
                    foreground_voxels,
                )

            if patient_max_foreground > 0:
                patient_foreground.append(
                    (str(patient_id), patient_max_foreground)
                )

        if cfg.overfit_volumes <= 0:
            raise ValueError("--overfit-volumes must be >= 1 when enabled.")

        if len(patient_foreground) < cfg.overfit_volumes:
            raise ValueError(
                f"Requested {cfg.overfit_volumes} overfit patients, but only "
                f"{len(patient_foreground)} non-empty training patients exist."
            )

        selected_patient_ids = {
            patient_id
            for patient_id, _ in sorted(
                patient_foreground,
                key=lambda item: (-item[1], item[0]),
            )[:cfg.overfit_volumes]
        }

        train_dataframe = train_dataframe.loc[
            train_dataframe["patient_id"].isin(selected_patient_ids)
        ].copy()

        print(
            f"Overfit diagnostic enabled: "
            f"{len(selected_patient_ids)} patients / "
            f"{len(train_dataframe)} training volumes"
        )
        print(
            "Selected training patient count:",
            len(selected_patient_ids),
        )

    if cfg.controlled_training:
        controlled_foreground: dict[str, int] = {}

        for _, row in train_dataframe.iterrows():
            mask = np.asanyarray(
                nib.load(str(row["mask_path"])).dataobj
            )

            if cfg.target_mask == "et":
                foreground_voxels = int((mask == cfg.label_et).sum())
            elif cfg.target_mask == "necr":
                foreground_voxels = int((mask == cfg.label_necr).sum())
            else:
                foreground_voxels = int(
                    np.isin(mask, [cfg.label_et, cfg.label_necr]).sum()
                )

            controlled_foreground[str(row["patient_id"])] = max(
                controlled_foreground.get(str(row["patient_id"]), 0),
                foreground_voxels,
            )

        nonempty_patient_ids = {
            patient_id
            for patient_id, count in controlled_foreground.items()
            if count > 0
        }

        train_dataframe = train_dataframe.loc[
            train_dataframe["patient_id"].isin(nonempty_patient_ids)
        ].copy()

        print(
            "Controlled training enabled:"
            f" kept {len(nonempty_patient_ids)} non-empty training patients "
            f"and {len(train_dataframe)} volumes."
        )
        print(
            "Controlled mode: positive-only sampling, "
            f"no augmentation, lr={cfg.controlled_learning_rate:g}, weight_decay=0."
        )

    train_records = make_records(
        train_dataframe,
        cfg.modalities,
        "train",
    )

    validation_records = make_records(
        master_dataframe,
        cfg.modalities,
        "val",
    )

    test_records = make_records(
        master_dataframe,
        cfg.modalities,
        "test",
    )

    # --------------------------------------------------------------
    # Transform configuration
    # --------------------------------------------------------------
    transform_cfg = cfg

    if cfg.overfit_volumes or cfg.controlled_training:
        # Controlled/overfit diagnostics:
        # - positive-only crops
        # - augmentation disabled inside build_transforms()
        transform_cfg = Config(**asdict(cfg))
        transform_cfg.positive_crop_ratio = 1.0
        if cfg.overfit_volumes:
            transform_cfg.samples_per_volume = max(
                cfg.samples_per_volume,
                8,
            )

    # --------------------------------------------------------------
    # Datasets
    # --------------------------------------------------------------
    train_dataset = Dataset(
        data=train_records,
        transform=build_transforms(
            transform_cfg,
            training=True,
        ),
    )

    # Dedicated one-record, positive-only, no-augmentation dataset for the
    # single-batch diagnostic.
    single_batch_dataset = None
    if cfg.run_name.startswith("single_batch_overfit"):
        positive_train_records = []

        for record in train_records:
            positive_train_records.append(record)

        if not positive_train_records:
            raise RuntimeError(
                "No training records are available for single-batch overfit."
            )

        single_batch_cfg = Config(**asdict(cfg))
        single_batch_cfg.overfit_volumes = 1
        single_batch_cfg.positive_crop_ratio = 1.0
        single_batch_cfg.samples_per_volume = 1

        single_batch_dataset = Dataset(
            data=[positive_train_records[0]],
            transform=build_transforms(
                single_batch_cfg,
                training=True,
            ),
        )

    validation_dataset = Dataset(
        data=validation_records,
        transform=build_transforms(
            cfg,
            training=False,
        ),
    )

    test_dataset = Dataset(
        data=test_records,
        transform=build_transforms(
            cfg,
            training=False,
        ),
    )

    # IMPORTANT:
    # In overfit mode, evaluate on the SAME volumes used for training.
    # In normal mode, evaluate on the independent validation split.
    overfit_eval_dataset = None

    if cfg.overfit_volumes:
        overfit_eval_dataset = Dataset(
            data=train_records,
            transform=build_transforms(
                cfg,
                training=False,
            ),
        )

    # --------------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------------
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=list_data_collate,
        worker_init_fn=worker_init_fn,
        generator=generator,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )

    single_batch_loader = None
    if single_batch_dataset is not None:
        single_batch_loader = DataLoader(
            single_batch_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=list_data_collate,
            pin_memory=device.type == "cuda",
        )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    overfit_eval_loader = None

    if overfit_eval_dataset is not None:
        overfit_eval_loader = DataLoader(
            overfit_eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

    diagnostic_mode = (
        cfg.overfit_volumes > 0
        or cfg.controlled_training
        or cfg.run_name.startswith("single_batch_overfit")
    )

    if cfg.controlled_training:
        effective_learning_rate = cfg.controlled_learning_rate
    elif cfg.overfit_volumes > 0 or cfg.run_name.startswith("single_batch_overfit"):
        effective_learning_rate = cfg.overfit_learning_rate
    else:
        effective_learning_rate = cfg.learning_rate
    effective_weight_decay = (
        cfg.overfit_weight_decay
        if diagnostic_mode
        else cfg.weight_decay
    )

    # --------------------------------------------------------------
    # Patch sampling sanity check
    # --------------------------------------------------------------
    patch_report = patch_statistics(
        train_loader,
        min_foreground_voxels=cfg.min_foreground_voxels,
    )

    patch_report["requested_positive_crop_ratio"] = (
        transform_cfg.positive_crop_ratio
    )
    patch_report["samples_per_volume"] = (
        transform_cfg.samples_per_volume
    )
    patch_report["training_volumes"] = len(train_records)
    patch_report["overfit_volumes"] = cfg.overfit_volumes
    patch_report["controlled_training"] = cfg.controlled_training
    patch_report["effective_learning_rate"] = effective_learning_rate
    patch_report["effective_weight_decay"] = effective_weight_decay
    patch_report["amp_enabled"] = cfg.amp and device.type == "cuda"

    with (experiment_dir / "patch_sampling_report.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            patch_report,
            handle,
            indent=2,
        )

    #print("Patch sampling:", patch_report)

    model = create_model(cfg).to(device)

    criterion = DiceCELoss(
        sigmoid=True,
        softmax=False,
        to_onehot_y=False,
        include_background=True,
        lambda_dice=1.0,
        lambda_ce=0.25,
        reduction="mean",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=effective_learning_rate,
        weight_decay=effective_weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
    )

    if cfg.controlled_training and cfg.amp and device.type == "cuda":
        print(
            "WARNING: controlled training is using AMP. "
            "For the most numerically stable diagnostic, pass --no-amp."
        )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=cfg.amp and device.type == "cuda",
    )

    writer = SummaryWriter(experiment_dir / "tensorboard")

    if cfg.single_batch_overfit_steps > 0 and cfg.run_name.startswith("single_batch_overfit"):
        if single_batch_loader is None:
            raise RuntimeError(
                "Single-batch diagnostic loader was not created."
            )

        # Use the overfit LR even though --overfit-volumes is not required
        # for this diagnostic.
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = cfg.overfit_learning_rate
            parameter_group["weight_decay"] = cfg.overfit_weight_decay

        run_single_batch_overfit(
            model=model,
            train_loader=single_batch_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device,
        )
        writer.close()
        return

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    #print(f"Trainable parameters: {parameter_count:,}")

    if cfg.smoke_test:
        run_smoke_test(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device,
            experiment_dir=experiment_dir,
        )
        writer.close()
        return

    best_dice = -math.inf
    best_threshold = 0.5
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()

        train_loss, train_patch_dice = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device,
            epoch=epoch,
        )

        if cfg.overfit_volumes:
            if overfit_eval_loader is None:
                raise RuntimeError(
                    "Overfit evaluation loader was not initialized."
                )
            evaluation_loader = overfit_eval_loader
        else:
            evaluation_loader = validation_loader

        validation_metrics, validation_dataframe = evaluate_full_volumes(
            model=model,
            loader=evaluation_loader,
            cfg=cfg,
            device=device,
            threshold=None,
        )

        validation_dice = validation_metrics["dice_mean"]
        selected_threshold = validation_metrics["selected_threshold"]

        if not math.isfinite(validation_dice):
            raise FloatingPointError(
                f"Validation Dice is non-finite at epoch {epoch}."
            )

        scheduler.step(validation_dice)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_patch_soft_dice": train_patch_dice,
            "validation_dice": validation_dice,
            "validation_precision": validation_metrics["precision_mean"],
            "validation_recall": validation_metrics["recall_mean"],
            "evaluation_mode": (
                "overfit_train"
                if cfg.overfit_volumes
                else "validation"
            ),
            "selected_threshold": selected_threshold,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_start,
        }

        history.append(row)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar(
            "train/patch_soft_dice",
            train_patch_dice,
            epoch,
        )
        writer.add_scalar("validation/dice", validation_dice, epoch)
        writer.add_scalar(
            "validation/precision",
            validation_metrics["precision_mean"],
            epoch,
        )
        writer.add_scalar(
            "validation/recall",
            validation_metrics["recall_mean"],
            epoch,
        )
        writer.add_scalar(
            "validation/threshold",
            selected_threshold,
            epoch,
        )
        writer.add_scalar(
            "optimizer/learning_rate",
            optimizer.param_groups[0]["lr"],
            epoch,
        )

        improved = validation_dice > best_dice + 1e-5

        if improved:
            best_dice = validation_dice
            best_threshold = selected_threshold
            epochs_without_improvement = 0

            save_checkpoint(
                path=experiment_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_validation_dice=best_dice,
                selected_threshold=best_threshold,
                cfg=cfg,
            )

            validation_dataframe.to_csv(
                experiment_dir / "validation_predictions.csv",
                index=False,
            )

            pd.DataFrame(
                validation_metrics["threshold_sweep"]
            ).to_csv(
                experiment_dir / "validation_threshold_sweep.csv",
                index=False,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            path=experiment_dir / "last_model.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_validation_dice=best_dice,
            selected_threshold=best_threshold,
            cfg=cfg,
        )

        pd.DataFrame(history).to_csv(
            experiment_dir / "training_history.csv",
            index=False,
        )

        metric_name = (
            "overfit_dice"
            if cfg.overfit_volumes
            else "val_dice"
        )

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"loss={train_loss:.4f} | "
            f"patch_dice={train_patch_dice:.4f} | "
            f"{metric_name}={validation_dice:.4f} | "
            f"threshold={selected_threshold:.2f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if epochs_without_improvement >= cfg.early_stopping_patience:
            print("Early stopping activated.")
            break

    checkpoint = load_checkpoint(
        experiment_dir / "best_model.pt",
        model=model,
        device=device,
    )

    # Critical: test gets the validation-selected threshold from best checkpoint.
    test_threshold = float(checkpoint["selected_threshold"])

    test_metrics, test_dataframe = evaluate_full_volumes(
        model=model,
        loader=test_loader,
        cfg=cfg,
        device=device,
        threshold=test_threshold,
    )

    test_dataframe.to_csv(
        experiment_dir / "test_predictions.csv",
        index=False,
    )

    final_metrics = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_dice": float(
            checkpoint["best_validation_dice"]
        ),
        "validation_selected_threshold": test_threshold,
        "test": test_metrics,
        "test_phase_summary": test_metrics.get("phase_summary", {}),
        "parameter_count": parameter_count,
        "device": str(device),
    }

    with (experiment_dir / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(final_metrics, handle, indent=2)

    writer.close()

    print("\nTraining complete.")
    print(f"Experiment directory: {experiment_dir}")
    if cfg.controlled_training:
        print(
            "Controlled training completed. "
            "This run is a diagnostic baseline, not the final clinical model."
        )
    print(
        f"Best validation Dice: "
        f"{checkpoint['best_validation_dice']:.4f}"
    )
    print(f"Validation-selected threshold: {test_threshold:.2f}")
    print(f"Test Dice: {test_metrics['dice_mean']:.4f}")
    print(f"Test precision: {test_metrics['precision_mean']:.4f}")
    print(f"Test recall: {test_metrics['recall_mean']:.4f}")


if __name__ == "__main__":
    main()

