"""Create a valid, patient-level indexed GBM dataset manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PHASES = {0: "preop", 1: "postop", 2: "recurrence"}
MODALITIES = ("t1", "t1ce", "t2", "flair", "adc")


def normalize_patient_id(value: object) -> str:
    return str(value).replace("\ufeff", "").strip().lower()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/app/data/input/dataset"),
        help="RHUH-GBM_nii_v1 könyvtár",
    )

    parser.add_argument(
        "--clinical-csv",
        type=Path,
        default=Path("/app/data/input/clinical.csv"),
        help="Clinical CSV fájl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-patients", type=int, default=24)
    parser.add_argument("--val-patients", type=int, default=8)
    parser.add_argument("--test-patients", type=int, default=8)

    return parser.parse_args()


def create_patient_splits(
    patient_ids: list[str],
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
) -> dict[str, set[str]]:
    requested = train_count + val_count + test_count

    if len(patient_ids) != requested:
        raise ValueError(
            f"A split {requested} beteget vár, de {len(patient_ids)} beteg található."
        )

    shuffled = np.random.default_rng(seed).permutation(
        sorted(patient_ids)
    ).tolist()

    splits = {
        "train": set(shuffled[:train_count]),
        "val": set(shuffled[train_count:train_count + val_count]),
        "test": set(shuffled[train_count + val_count:]),
    }

    if (
        splits["train"] & splits["val"]
        or splits["train"] & splits["test"]
        or splits["val"] & splits["test"]
    ):
        raise AssertionError("Betegszintű split átfedés található.")

    return splits


def main() -> None:
    args = parse_args()

    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root nem található: {args.dataset_root}")

    if not args.clinical_csv.is_file():
        raise FileNotFoundError(f"Clinical CSV nem található: {args.clinical_csv}")

    clinical = pd.read_csv(
        args.clinical_csv,
        sep=";",
        encoding="utf-8-sig",
    )

    patient_column = next(
        (
            column
            for column in clinical.columns
            if column.strip().lower().replace(" ", "_") == "patient_id"
        ),
        None,
    )

    if patient_column is None:
        raise KeyError("A clinical CSV nem tartalmaz Patient ID / patient_id oszlopot.")

    clinical = clinical.rename(columns={patient_column: "patient_id"})
    clinical["patient_id"] = clinical["patient_id"].map(normalize_patient_id)
    clinical = clinical.loc[clinical["patient_id"].ne("")].copy()

    if clinical["patient_id"].duplicated().any():
        duplicates = clinical.loc[
            clinical["patient_id"].duplicated(),
            "patient_id",
        ].tolist()
        raise ValueError(f"Duplikált clinical betegazonosító(k): {duplicates}")

    splits = create_patient_splits(
        patient_ids=clinical["patient_id"].tolist(),
        train_count=args.train_patients,
        val_count=args.val_patients,
        test_count=args.test_patients,
        seed=args.seed,
    )

    split_map = {
        patient_id: split_name
        for split_name, patient_ids in splits.items()
        for patient_id in patient_ids
    }

    records: list[dict[str, object]] = []
    missing_files: list[str] = []

    for patient_id in clinical["patient_id"]:
        filesystem_patient_id = patient_id.upper()

        for phase_id, phase_name in PHASES.items():
            phase_dir = args.dataset_root / filesystem_patient_id / str(phase_id)

            record: dict[str, object] = {
                "patient_id": patient_id,
                "phase_id": phase_id,
                "phase_name": phase_name,
                "split": split_map[patient_id],
            }

            mask_path = phase_dir / (
                f"{filesystem_patient_id}_{phase_id}_segmentations.nii.gz"
            )
            record["mask_path"] = str(mask_path)
            record["mask_exists"] = mask_path.is_file()

            for modality in MODALITIES:
                modality_path = phase_dir / (
                    f"{filesystem_patient_id}_{phase_id}_{modality}.nii.gz"
                )
                record[f"{modality}_path"] = str(modality_path)
                record[f"{modality}_exists"] = modality_path.is_file()

            # Visszafelé kompatibilitás a régebbi kóddal.
            record["image_path"] = record["t1_path"]
            record["image_exists"] = record["t1_exists"]

            absent = [
                key
                for key, exists in record.items()
                if key.endswith("_exists") and not exists
            ]

            if absent:
                missing_files.append(
                    f"{filesystem_patient_id}, phase={phase_id}: {', '.join(absent)}"
                )

            records.append(record)

    if missing_files:
        raise FileNotFoundError(
            "Hiányzó szükséges NIfTI fájlok:\n"
            + "\n".join(missing_files[:50])
        )

    manifest = pd.DataFrame(records).merge(
        clinical,
        on="patient_id",
        how="left",
        validate="many_to_one",
    )

    # Kötelező betegszintű integritás-ellenőrzés.
    for split_name in ("train", "val", "test"):
        if manifest.loc[manifest["split"] == split_name, "patient_id"].nunique() == 0:
            raise RuntimeError(f"Üres split: {split_name}")

    train_ids = set(manifest.loc[manifest["split"] == "train", "patient_id"])
    val_ids = set(manifest.loc[manifest["split"] == "val", "patient_id"])
    test_ids = set(manifest.loc[manifest["split"] == "test", "patient_id"])

    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(
        args.output_dir / "master_dataset.csv",
        index=False,
        encoding="utf-8",
    )

    for split_name, patient_ids in splits.items():
        clinical.loc[
            clinical["patient_id"].isin(patient_ids)
        ].to_csv(
            args.output_dir / f"{split_name}_patients.csv",
            index=False,
            encoding="utf-8",
        )

    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(
            {name: sorted(ids) for name, ids in splits.items()},
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Manifest: {args.output_dir / 'master_dataset.csv'}")
    print(f"Volumenek száma: {len(manifest)}")
    print(
        "Betegek split szerint:",
        manifest.groupby("split")["patient_id"].nunique().to_dict(),
    )


if __name__ == "__main__":
    main()