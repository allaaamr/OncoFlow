#!/usr/bin/env python

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import (
    FLAIR_INDEX,
    fit_train_image_stats,
    n4_kwargs_from_dc,
    save_image_stats,
    split_patients_by_id,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default=None, help="Override data.data_dir")
    parser.add_argument("--out-path", default=None, help="Override data.image_stats_path")
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing stats file at the output path (default: refuse).",
    )
    parser.add_argument(
        "--allow-inline-n4", action="store_true",
        help=(
            "Compute N4 inline for any volume missing from n4_cache_dir instead of failing "
            "(default: require scripts/precompute_n4.py to have already cached every train "
            "volume). Only for ad-hoc/small-scale use — normal runs should precompute first."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    dc = cfg["data"]

    data_dir = args.data_dir or dc["data_dir"]
    split_csv_path = dc.get("split_csv_path")
    if not split_csv_path:
        raise ValueError(
            "[fit_image_stats] data.split_csv_path is required — image statistics must be fit "
            "on the SAME train patients this run will train on. Create the split first (see "
            "src/data/dataset.py::split_patients_by_id) or point split_csv_path at an existing one."
        )

    out_path = args.out_path or dc.get("image_stats_path")
    if not out_path:
        raise ValueError(
            "[fit_image_stats] data.image_stats_path is required — it names where the fitted "
            "frozen statistics will be written, and is what Trainer/precompute_latents.py read "
            "back later."
        )
    if os.path.exists(out_path) and not args.force:
        raise FileExistsError(
            f"[fit_image_stats] '{out_path}' already exists. Refusing to silently overwrite "
            "previously-fit statistics that other runs/checkpoints may already depend on — pass "
            "--force if you intend to refit (e.g. after changing normalization_clip_pct or the "
            "split), and re-run scripts/precompute_latents.py afterwards if a latent cache exists."
        )

    train_ids, val_ids, test_ids = split_patients_by_id(
        data_dir=data_dir,
        seed=int(dc.get("split_seed", 0)),
        save_csv_path=split_csv_path,
        require_existing=True,
    )

    flair_index = int(dc.get("flair_index", FLAIR_INDEX))
    clip_pct = tuple(dc.get("normalization_clip_pct", (0.5, 99.5)))
    apply_n4 = bool(dc.get("apply_n4_bias_correction", False))
    n4_cache_dir = dc.get("n4_cache_dir")
    n4_kwargs = n4_kwargs_from_dc(dc)
    require_n4_cache = not args.allow_inline_n4
    reservoir_size = int(dc.get("image_stats_reservoir_size", 2_000_000))
    seed = int(dc.get("image_stats_seed", 0))

    print("=" * 60)
    print("Fitting frozen train-only MRI intensity normalization statistics")
    print("=" * 60)
    print(f"Data dir:          {data_dir}")
    print(f"Split CSV:         {split_csv_path} (train={len(train_ids)} val={len(val_ids)} test={len(test_ids)})")
    print(f"Clip percentiles:  {clip_pct}")
    print(
        f"Apply N4:          {apply_n4}"
        + (f" (cache_dir={n4_cache_dir}, require_cache={require_n4_cache}, kwargs={n4_kwargs})" if apply_n4 else "")
    )
    print(f"Reservoir size:    {reservoir_size} (seed={seed})")
    print(f"Output:            {out_path}")
    print("=" * 60)

    stats_by_channel = fit_train_image_stats(
        data_dir=data_dir,
        train_patient_ids=train_ids,
        flair_index=flair_index,
        clip_pct=clip_pct,
        reservoir_size=reservoir_size,
        seed=seed,
        apply_n4=apply_n4,
        n4_cache_dir=n4_cache_dir,
        n4_kwargs=n4_kwargs,
        require_n4_cache=require_n4_cache,
    )

    for channel, stats in stats_by_channel.items():
        print(
            f"[channel={channel}] clip=[{stats.clip_lo:.4f}, {stats.clip_hi:.4f}] "
            f"mean={stats.mean:.4f} std={stats.std:.4f} "
            f"n_scans={stats.n_scans} n_foreground_voxels={stats.n_foreground_voxels} "
            f"zero_variance={stats.zero_variance}"
        )
        if stats.zero_variance:
            print(
                f"[fit_image_stats] WARNING: channel={channel} has near-zero variance after "
                "clipping — check the input data before trusting this normalization."
            )

    save_image_stats(
        out_path,
        stats_by_channel,
        train_patient_ids=train_ids,
        data_dir=data_dir,
        apply_n4=apply_n4,
    )
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
