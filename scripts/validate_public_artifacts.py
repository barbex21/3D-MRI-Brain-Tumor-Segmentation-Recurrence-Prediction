#!/usr/bin/env python3
"""Validate public, aggregate-only artifacts without loading private research data."""

from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "Results" / "experiment_audit.csv"
EXPECTED_COLUMNS = [
    "target",
    "modalities",
    "validation_dice",
    "test_dice",
    "selected_threshold",
    "experiment_id",
    "source_file",
    "verification_status",
]
REQUIRED_FILES = [
    "README.md",
    "Code/03_monai_unet_controlled_v10.py",
    "Results/experiment_audit.csv",
    "docs/data_dictionary.md",
    "docs/dataset_and_citation.md",
    "docs/reproducibility.md",
    "docs/results_summary.md",
]
ALLOWED_TARGETS = {"et", "necr"}
ALLOWED_MODALITIES = {"t1", "t1ce", "t2", "flair"}
ALLOWED_STATUSES = {
    "verified_json",
    "recovered_summary_and_report",
    "recovered_comparison_report",
}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def tracked_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [
        item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    ]


def validate_privacy(paths: list[str]) -> None:
    forbidden_suffixes = (
        ".nii",
        ".nii.gz",
        ".dcm",
        ".dicom",
        ".pt",
        ".pth",
        ".ckpt",
        ".onnx",
        ".safetensors",
        ".h5",
        ".hdf5",
    )
    forbidden_names = {
        ".env",
        "credentials.json",
        "secrets.json",
    }
    violations = []
    for path in paths:
        normalized = path.replace("\\", "/").lower()
        name = normalized.rsplit("/", 1)[-1]
        if normalized.startswith("data/"):
            violations.append(path)
        elif name in forbidden_names or name.startswith(".env."):
            violations.append(path)
        elif name.endswith(forbidden_suffixes):
            violations.append(path)
        elif any(token in name for token in ("clinical_data", "master_dataset", "patient_table")):
            violations.append(path)
    if violations:
        fail("private or model artifacts are tracked: " + ", ".join(violations))


def validate_audit() -> None:
    if not AUDIT.is_file():
        fail(f"missing audit CSV: {AUDIT.relative_to(ROOT)}")
    with AUDIT.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_COLUMNS:
            fail(f"unexpected CSV columns: {reader.fieldnames}")
        rows = list(reader)

    if len(rows) != 13:
        fail(f"expected 13 experiment rows, found {len(rows)}")

    seen_ids: set[str] = set()
    statuses: dict[str, int] = {}
    for line_number, row in enumerate(rows, start=2):
        if row["target"] not in ALLOWED_TARGETS:
            fail(f"invalid target on CSV line {line_number}")
        if not row["experiment_id"] or row["experiment_id"] in seen_ids:
            fail(f"missing or duplicate experiment_id on CSV line {line_number}")
        seen_ids.add(row["experiment_id"])

        modalities = row["modalities"].split("+")
        if not modalities or any(item not in ALLOWED_MODALITIES for item in modalities):
            fail(f"invalid modality on CSV line {line_number}")

        for field in ("validation_dice", "test_dice", "selected_threshold"):
            try:
                value = float(row[field])
            except ValueError:
                fail(f"missing or non-numeric {field} on CSV line {line_number}")
            if not math.isfinite(value) or not 0 <= value <= 1:
                fail(f"{field} outside [0, 1] on CSV line {line_number}")

        if not row["source_file"]:
            fail(f"missing provenance on CSV line {line_number}")
        if row["verification_status"] not in ALLOWED_STATUSES:
            fail(f"unknown verification status on CSV line {line_number}")
        statuses[row["verification_status"]] = statuses.get(row["verification_status"], 0) + 1

    if statuses.get("verified_json") != 10:
        fail(f"expected 10 JSON-verified rows; found {statuses.get('verified_json', 0)}")
    if statuses.get("recovered_summary_and_report") != 2:
        fail("expected two summary rows cross-checked against individual reports")
    if statuses.get("recovered_comparison_report") != 1:
        fail("expected one row recovered with the comparison report")

    threshold_row = next(
        row for row in rows if row["experiment_id"] == "20260827_075617_controlled_t1ce_t2_necr_100ep"
    )
    if float(threshold_row["selected_threshold"]) != 0.80:
        fail("T1ce+T2 NECR threshold must match the comparison report (0.80)")


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        fail("required public files are missing: " + ", ".join(missing))
    paths = tracked_paths()
    validate_privacy(paths)
    validate_audit()
    print("Public artifact checks passed: required files, 13-row audit, provenance, thresholds, and tracked-path privacy.")


if __name__ == "__main__":
    main()
