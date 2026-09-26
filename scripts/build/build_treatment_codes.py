from __future__ import annotations

import argparse
import ast
import csv
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("build_treatment_codes")

CODE_MAP = {"CRT": 0, "TMZ": 1, "IMT": 2, "None": 3}


def read_patient_ids(csv_path: Path) -> list[str]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [row["patient_id"].strip() for row in reader if row["patient_id"].strip()]


def load_treatment_by_patient(csv_path: Path) -> dict[str, list[str]]:
    out = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[row["Patient_ID"]] = ast.literal_eval(row["treatment"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/alaa.mohamed/MUGlioma2.csv")
    ap.add_argument(
        "--patients-csv",
        default=str(Path(__file__).resolve().parents[1] / "missing_treatment_patients.csv"),
    )
    ap.add_argument("--out-dir", default="/home/alaa.mohamed/MIU2")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite an existing *_treatment.npy if present")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patient_ids = read_patient_ids(Path(args.patients_csv))
    treatment_by_patient = load_treatment_by_patient(Path(args.csv))
    log.info("Loaded %d patient ID(s) from %s", len(patient_ids), args.patients_csv)

    written, skipped, errors = 0, 0, []
    code_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for patient_id in patient_ids:
        out_path = out_dir / f"{patient_id}_treatment.npy"
        if out_path.exists() and not args.overwrite:
            log.info("%s: %s already exists, skipping (pass --overwrite to replace)", patient_id, out_path)
            skipped += 1
            continue
        if patient_id not in treatment_by_patient:
            msg = f"{patient_id}: not found in {args.csv}"
            log.error(msg)
            errors.append(msg)
            continue

        try:
            codes = [CODE_MAP[c] for c in treatment_by_patient[patient_id]]
        except KeyError as exc:
            msg = f"{patient_id}: unrecognized treatment label {exc} in {treatment_by_patient[patient_id]}"
            log.error(msg)
            errors.append(msg)
            continue
        codes = np.array(codes, dtype=np.int64)

        mismatch = None
        for kind in ("label", "days"):
            sibling = out_dir / f"{patient_id}_{kind}.npy"
            if sibling.exists():
                arr = np.load(sibling, mmap_mode="r")
                n = arr.shape[0] if kind == "label" else len(arr)
                if n != len(codes):
                    mismatch = f"{patient_id}: treatment length {len(codes)} != {kind} length {n}"
                    break
        if mismatch:
            log.error(mismatch)
            errors.append(mismatch)
            continue

        np.save(out_path, codes)
        for c in codes:
            code_counts[int(c)] += 1
        log.info("%s: wrote %s codes=%s", patient_id, out_path, codes.tolist())
        written += 1

    log.info(
        "Done. written=%d skipped=%d errors=%d (of %d total). code distribution=%s",
        written, skipped, len(errors), len(patient_ids), code_counts,
    )
    if errors:
        log.warning("Patients with errors:\n  " + "\n  ".join(errors))


if __name__ == "__main__":
    main()
