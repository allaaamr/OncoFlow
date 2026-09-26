from __future__ import annotations

import argparse
import ast
import csv
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("build_days_from_csv")


def read_patient_ids(csv_path: Path) -> list[str]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [row["patient_id"].strip() for row in reader if row["patient_id"].strip()]


def load_days_by_patient(csv_path: Path) -> dict[str, list[float]]:
    out = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[row["Patient_ID"]] = ast.literal_eval(row["days"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/alaa.mohamed/MUGlioma2.csv")
    ap.add_argument(
        "--patients-csv",
        default=str(Path(__file__).resolve().parents[1] / "missing_days_patients.csv"),
    )
    ap.add_argument("--out-dir", default="/home/alaa.mohamed/MIU2")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite an existing *_days.npy if present")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patient_ids = read_patient_ids(Path(args.patients_csv))
    days_by_patient = load_days_by_patient(Path(args.csv))
    log.info("Loaded %d patient ID(s) from %s", len(patient_ids), args.patients_csv)

    written, skipped, errors = 0, 0, []

    for patient_id in patient_ids:
        out_path = out_dir / f"{patient_id}_days.npy"
        if out_path.exists() and not args.overwrite:
            log.info("%s: %s already exists, skipping (pass --overwrite to replace)", patient_id, out_path)
            skipped += 1
            continue
        if patient_id not in days_by_patient:
            msg = f"{patient_id}: not found in {args.csv}"
            log.error(msg)
            errors.append(msg)
            continue

        days = np.array(days_by_patient[patient_id], dtype=np.float32)

        mismatch = None
        for kind in ("label", "treatment"):
            sibling = out_dir / f"{patient_id}_{kind}.npy"
            if sibling.exists():
                arr = np.load(sibling, mmap_mode="r")
                n = arr.shape[0] if kind == "label" else len(arr)
                if n != len(days):
                    mismatch = f"{patient_id}: days length {len(days)} != {kind} length {n}"
                    break

        if mismatch:
            log.error(mismatch)
            errors.append(mismatch)
            continue

        np.save(out_path, days)
        log.info("%s: wrote %s days=%s", patient_id, out_path, days.tolist())
        written += 1

    log.info(
        "Done. written=%d skipped=%d errors=%d (of %d total)",
        written, skipped, len(errors), len(patient_ids),
    )
    if errors:
        log.warning("Patients with errors:\n  " + "\n  ".join(errors))


if __name__ == "__main__":
    main()
