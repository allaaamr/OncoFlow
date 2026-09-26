#!/usr/bin/env python

import argparse
import glob
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import preprocessing as preprocessing_utils
from src.data.dataset import (
    FLAIR_INDEX,
    assert_disjoint_splits,
    n4_kwargs_from_dc,
    patient_id_from_image_path,
    split_patients_by_id,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default=None, help="Override data.data_dir")
    parser.add_argument("--n4-cache-dir", default=None, help="Override data.n4_cache_dir")
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute and overwrite every volume's cache entry even if already cached.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    dc = cfg["data"]

    data_dir = args.data_dir or dc["data_dir"]
    n4_cache_dir = args.n4_cache_dir or dc.get("n4_cache_dir")
    if not n4_cache_dir:
        raise ValueError(
            "[precompute_n4] data.n4_cache_dir (or --n4-cache-dir) is required — this script's "
            "entire purpose is to populate a persistent, on-disk N4 cache."
        )

    split_csv_path = dc.get("split_csv_path")
    if not split_csv_path:
        raise ValueError(
            "[precompute_n4] data.split_csv_path is required — N4 is precomputed per patient-level "
            "split so train/val/test PatientDataset instances all read from the same cache."
        )

    train_ids, val_ids, test_ids = split_patients_by_id(
        data_dir=data_dir,
        seed=int(dc.get("split_seed", 0)),
        save_csv_path=split_csv_path,
        require_existing=True,
    )
    assert_disjoint_splits(train_ids, val_ids, test_ids)
    all_ids = sorted(set(train_ids) | set(val_ids) | set(test_ids))

    flair_index = int(dc.get("flair_index", FLAIR_INDEX))
    n4_kwargs = n4_kwargs_from_dc(dc)

    print("=" * 60)
    print("Precomputing N4 bias-field correction for every split")
    print("=" * 60)
    print(f"Data dir:      {data_dir}")
    print(f"Split CSV:     {split_csv_path} (train={len(train_ids)} val={len(val_ids)} test={len(test_ids)})")
    print(f"N4 cache dir:  {n4_cache_dir}")
    print(f"N4 kwargs:     {n4_kwargs}")
    print(f"Force:         {args.force}")
    print("=" * 60)

    allowed = set(all_ids)
    image_files = sorted(glob.glob(os.path.join(data_dir, "*_image.npy")))
    image_files = [p for p in image_files if patient_id_from_image_path(p) in allowed]
    print(f"Found {len(image_files)}/{len(sorted(glob.glob(os.path.join(data_dir, '*_image.npy'))))} "
          "patient image files matching the configured split.")

    n_cached, n_computed, n_failed = 0, 0, 0
    for i, img_path in enumerate(image_files):
        pid = patient_id_from_image_path(img_path)
        img_all = np.load(img_path, mmap_mode="r")
        n_sessions = img_all.shape[0]
        for s in range(n_sessions):
            cache_key = preprocessing_utils.compute_n4_cache_key(pid, s, flair_index, img_path, n4_kwargs)
            cache_path = os.path.join(n4_cache_dir, f"{cache_key}.npy")
            if os.path.exists(cache_path) and not args.force:
                n_cached += 1
                continue
            try:
                volume = np.asarray(img_all[s, flair_index, :, :, :])
                mask = preprocessing_utils.compute_foreground_mask(volume)
                volume, _ = preprocessing_utils.sanitize_non_finite(volume)
                preprocessing_utils.n4_correct_with_cache(
                    volume, mask, n4_cache_dir, cache_key, require_cache=False, **n4_kwargs
                )
                n_computed += 1
            except Exception as e:
                n_failed += 1
                print(f"[precompute_n4] FAILED patient={pid} session={s}: {e}", flush=True)
        print(f"[{i + 1}/{len(image_files)}] {pid}: {n_sessions} session(s) done", flush=True)

    print("=" * 60)
    print(f"Done. computed={n_computed} already_cached={n_cached} failed={n_failed}")
    print(f"Cache dir: {n4_cache_dir}")
    print("=" * 60)
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
