from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

import nibabel as nib
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("build_labels_from_nifti")

EXPECTED_DEPTH = 155
LABEL_DTYPE = np.int16


def read_patient_ids(csv_path: Path) -> list[str]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [row["patient_id"].strip() for row in reader if row["patient_id"].strip()]


def find_timepoint_files(patient_dir: Path, patient_id: str) -> list[tuple[int, Path]]:
    pattern = re.compile(rf"^{re.escape(patient_id)}_(\d+)_tumorMask\.nii\.gz$")
    found = {}
    for path in patient_dir.rglob(f"{patient_id}_*_tumorMask.nii.gz"):
        m = pattern.match(path.name)
        if not m:
            continue
        tp = int(m.group(1))
        if tp in found:
            raise ValueError(
                f"{patient_id}: duplicate timepoint {tp} — found both "
                f"'{found[tp]}' and '{path}'"
            )
        found[tp] = path
    return sorted(found.items())


def to_dhw(vol: np.ndarray, patient_id: str, timepoint: int, expected_depth: int) -> np.ndarray:
    if vol.ndim != 3:
        raise ValueError(
            f"{patient_id} tp{timepoint}: expected a 3D volume, got ndim={vol.ndim} "
            f"shape={vol.shape}"
        )
    if vol.shape[0] == expected_depth:
        return vol
    depth_axes = [i for i, s in enumerate(vol.shape) if s == expected_depth]
    if len(depth_axes) == 1:
        return np.moveaxis(vol, depth_axes[0], 0)
    raise ValueError(
        f"{patient_id} tp{timepoint}: cannot unambiguously find a depth axis of size "
        f"{expected_depth} in NIfTI shape {vol.shape}"
    )


def build_patient_label(
    data_dir: Path, patient_id: str, expected_depth: int
) -> np.ndarray:
    patient_dir = data_dir / patient_id
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"{patient_id}: no folder at {patient_dir}")

    timepoints = find_timepoint_files(patient_dir, patient_id)
    if not timepoints:
        raise FileNotFoundError(
            f"{patient_id}: no '{patient_id}_<N>_tumorMask.nii.gz' files found under {patient_dir}"
        )

    tp_numbers = [tp for tp, _ in timepoints]
    if tp_numbers != sorted(tp_numbers) or len(set(tp_numbers)) != len(tp_numbers):
        raise ValueError(f"{patient_id}: unexpected/duplicate timepoint numbers {tp_numbers}")
    if tp_numbers != list(range(tp_numbers[0], tp_numbers[0] + len(tp_numbers))):
        log.warning(
            "%s: timepoint numbers are non-contiguous (%s) — proceeding with "
            "whatever was found, in ascending order",
            patient_id, tp_numbers,
        )

    volumes = []
    ref_hw = None
    for tp, path in timepoints:
        raw = np.asarray(nib.load(str(path)).get_fdata())
        vol = to_dhw(raw, patient_id, tp, expected_depth)
        if ref_hw is None:
            ref_hw = vol.shape[1:]
        elif vol.shape[1:] != ref_hw:
            raise ValueError(
                f"{patient_id} tp{tp}: (H,W)={vol.shape[1:]} does not match this "
                f"patient's earlier timepoint(s) (H,W)={ref_hw}"
            )
        volumes.append(vol.astype(LABEL_DTYPE))

    return np.stack(volumes, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Root folder containing <patient_id>/ subfolders")
    ap.add_argument(
        "--patients-csv",
        default=str(Path(__file__).resolve().parents[1] / "missing_label_patients.csv"),
    )
    ap.add_argument("--out-dir", default="/home/alaa.mohamed/MIU2")
    ap.add_argument("--expected-depth", type=int, default=EXPECTED_DEPTH)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite an existing *_label.npy if present")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patient_ids = read_patient_ids(Path(args.patients_csv))
    log.info("Loaded %d patient ID(s) from %s", len(patient_ids), args.patients_csv)

    written, skipped, errors = 0, 0, []

    for patient_id in patient_ids:
        out_path = out_dir / f"{patient_id}_label.npy"
        if out_path.exists() and not args.overwrite:
            log.info("%s: %s already exists, skipping (pass --overwrite to replace)", patient_id, out_path)
            skipped += 1
            continue
        try:
            label = build_patient_label(data_dir, patient_id, args.expected_depth)
        except Exception as exc:
            log.error("%s: %s", patient_id, exc)
            errors.append(f"{patient_id}: {exc}")
            continue

        np.save(out_path, label)
        log.info("%s: wrote %s shape=%s dtype=%s", patient_id, out_path, label.shape, label.dtype)
        written += 1

    log.info(
        "Done. written=%d skipped=%d errors=%d (of %d total)",
        written, skipped, len(errors), len(patient_ids),
    )
    if errors:
        log.warning("Patients with errors:\n  " + "\n  ".join(errors))


if __name__ == "__main__":
    main()
