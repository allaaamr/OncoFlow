from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("fix_days_treatment_shape")


def find_target_files(data_dir: Path) -> List[Path]:
    return sorted(data_dir.glob("*_days.npy")) + sorted(data_dir.glob("*_treatment.npy"))


def plan_fix(path: Path) -> Tuple[str, Tuple[int, ...], Optional[np.ndarray]]:
    arr = np.load(path)
    if arr.ndim == 1:
        return "already_1d", arr.shape, None
    if arr.ndim >= 2 and all(d == 1 for d in arr.shape[1:]):
        return "squeezable", arr.shape, arr.reshape(arr.shape[0]).astype(arr.dtype)
    return "unexpected", arr.shape, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="Directory containing *_days.npy / *_treatment.npy files (e.g. the CFB-GBM data_dir)")
    ap.add_argument("--apply", action="store_true", help="Actually overwrite files. Without this flag, only previews what would change.")
    ap.add_argument(
        "--backup-dir", default=None,
        help="Where to copy each file's ORIGINAL contents before overwriting "
             "(default: <data-dir>/_shape_fix_backup). Ignored without --apply.",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"--data-dir not found: {data_dir}")
    backup_dir = Path(args.backup_dir) if args.backup_dir else data_dir / "_shape_fix_backup"

    files = find_target_files(data_dir)
    if not files:
        log.warning("No *_days.npy / *_treatment.npy files found under %s", data_dir)
        return
    log.info("Found %d days/treatment file(s) under %s", len(files), data_dir)

    to_fix: List[Tuple[Path, np.ndarray]] = []
    unexpected: List[Tuple[Path, Tuple[int, ...]]] = []
    already_ok = 0

    for path in files:
        status, orig_shape, fixed = plan_fix(path)
        if status == "already_1d":
            already_ok += 1
        elif status == "squeezable":
            log.info("%s: %s -> %s", path.name, orig_shape, fixed.shape)
            to_fix.append((path, fixed))
        else:
            unexpected.append((path, orig_shape))
            log.error(
                "%s: unexpected shape %s (not a plain (S,1)-style column vector) -- left untouched",
                path.name, orig_shape,
            )

    log.info(
        "Summary: %d already (S,), %d fixable (S,1)->(S,), %d unexpected shape (skipped)",
        already_ok, len(to_fix), len(unexpected),
    )

    if not args.apply:
        log.info(
            "Dry run (no --apply passed) -- no files were modified. "
            "Re-run with --apply to write the %d fix(es) above.", len(to_fix),
        )
        return

    if not to_fix:
        log.info("Nothing to fix -- no files written.")
    else:
        backup_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for path, fixed in to_fix:
            backup_path = backup_dir / path.name
            if not backup_path.exists():
                shutil.copy2(path, backup_path)
            np.save(path, fixed)
            written += 1
        log.info("Wrote %d fixed file(s) (originals backed up to %s).", written, backup_dir)

    if unexpected:
        log.warning(
            "%d file(s) had an unexpected shape and were left untouched -- inspect manually: %s",
            len(unexpected), ", ".join(p.name for p, _ in unexpected),
        )


if __name__ == "__main__":
    main()
