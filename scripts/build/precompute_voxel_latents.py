#!/usr/bin/env python

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import preprocessing as preprocessing_utils
from src.data.dataset import load_image_stats, n4_kwargs_from_dc
from src.models.vae_wrapper import FrozenMedVAE3D, FlairOnlyEncoder3D, build_dummy_vae3d


def build_encoder(cvae_cfg, flair_index, device):
    backend = cvae_cfg.get("backend", "medvae")
    if backend == "medvae":
        vae = FrozenMedVAE3D.from_config(cvae_cfg)
    elif backend == "dummy":
        vae = build_dummy_vae3d(cvae_cfg)
    else:
        raise ValueError(f"Unknown context_vae.backend={backend!r}")
    vae = vae.to(device).eval()
    return FlairOnlyEncoder3D(vae=vae, flair_index=flair_index, strict=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache-dir", default=None, help="Override data.voxel_latent_cache_dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true", help="Re-encode even if the cache file exists")
    ap.add_argument("--splits", default="train,val,test",
                    help="Comma-separated splits to cache (stats always use train only)")
    ap.add_argument("--backend", default=None, help="Override context_vae.backend (e.g. 'dummy' for tests)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dc = cfg["data"]
    if dc.get("representation") != "voxel":
        raise ValueError("This script is for data.representation='voxel' configs.")
    if "context_vae" not in cfg:
        raise ValueError("Config has no `context_vae:` block.")
    cvae_cfg = dict(cfg["context_vae"])
    if args.backend:
        cvae_cfg["backend"] = args.backend

    data_dir = dc["data_dir"]
    voxel_shape = tuple(int(v) for v in dc["voxel_shape"])
    shape_tag = "x".join(map(str, voxel_shape))
    cache_dir = args.cache_dir or dc.get("voxel_latent_cache_dir")
    if not cache_dir:
        raise ValueError("Set data.voxel_latent_cache_dir (or pass --cache-dir).")
    os.makedirs(cache_dir, exist_ok=True)

    clip_pct = tuple(dc.get("normalization_clip_pct", (0.5, 99.5)))
    flair_index = dc.get("flair_index", 2)
    policy = dc.get("normalization_policy", "whole_volume")
    apply_n4 = bool(dc.get("apply_n4_bias_correction", False))
    n4_cache_dir = dc.get("n4_cache_dir")
    n4_kwargs = n4_kwargs_from_dc(dc)
    image_stats = None
    if policy == "frozen_train_stats":
        if not dc.get("image_stats_path"):
            raise ValueError("normalization_policy='frozen_train_stats' requires data.image_stats_path")
        image_stats, _ = load_image_stats(dc["image_stats_path"])

    split_of = {}
    with open(dc["split_csv_path"]) as f:
        for r in csv.DictReader(f):
            split_of[r["patient_id"]] = r["split"]
    wanted = set(args.splits.split(","))

    print(f"Data dir: {data_dir}\nCache dir: {cache_dir}\nvoxel_shape: {voxel_shape}\n"
          f"Normalization: {policy}\nContext VAE: {cvae_cfg.get('model_name', 'medvae_4_1_3d')} "
          f"(backend={cvae_cfg.get('backend', 'medvae')})\nDevice: {args.device}\n" + "=" * 60)
    encoder = build_encoder(cvae_cfg, flair_index, args.device)

    n_vox, s1, s2 = 0, 0.0, 0.0
    n_enc = n_skip = n_patients = 0
    latent_shape = None
    image_files = sorted(glob.glob(os.path.join(data_dir, "*_image.npy")))
    for pi, img_path in enumerate(image_files):
        pid = os.path.basename(img_path).replace("_image.npy", "")
        split = split_of.get(pid)
        if split not in wanted:
            continue
        img_raw = np.load(img_path, mmap_mode="r")
        S, C = img_raw.shape[:2]
        idx = flair_index if C > 1 else 0
        n_patients += 1
        for s in range(S):
            path = os.path.join(cache_dir, f"{pid}_s{s}_{shape_tag}.npy")
            z = None
            if os.path.exists(path) and not args.force:
                n_skip += 1
                if split == "train" and s < S - 1:
                    z = np.load(path)
            else:
                n4_key = (preprocessing_utils.compute_n4_cache_key(pid, s, idx, img_path, n4_kwargs)
                          if apply_n4 else None)
                vol = preprocessing_utils.apply_normalization_policy(
                    policy,
                    raw_volume_fn=lambda s=s: np.asarray(img_raw[s, idx]),
                    clip_pct=clip_pct,
                    stats=image_stats.get(idx) if image_stats else None,
                    apply_n4=apply_n4, n4_cache_dir=n4_cache_dir, n4_cache_key=n4_key,
                    n4_kwargs=n4_kwargs, require_n4_cache=True,
                )
                vol = preprocessing_utils.to_model_shape_3d(
                    vol, voxel_shape, representation="voxel", is_label=False
                )
                x = torch.from_numpy(np.ascontiguousarray(vol)).float()[None, None].to(args.device)
                with torch.no_grad():
                    z = encoder(x)[0].float().cpu().numpy()
                tmp = f"{path}.{os.getpid()}.tmp.npy"
                np.save(tmp, z)
                os.replace(tmp, path)
                n_enc += 1
            if z is not None:
                latent_shape = z.shape
                if split == "train" and s < S - 1:
                    z64 = z.astype(np.float64)
                    n_vox += z64.size; s1 += z64.sum(); s2 += (z64 ** 2).sum()
        print(f"[{pi + 1}/{len(image_files)}] {pid} ({split}) sessions={S}", flush=True)

    print("=" * 60)
    print(f"patients={n_patients} encoded={n_enc} already-cached={n_skip}")
    if n_vox == 0:
        print("No train-split context latents found — cannot compute statistics.")
        return
    mean = s1 / n_vox
    std = float(np.sqrt(max(s2 / n_vox - mean ** 2, 0.0)))
    stats = {"latent_mean": float(mean), "latent_std": std, "latent_scale": 1.0 / std,
             "n_voxels": int(n_vox), "voxel_shape": list(voxel_shape),
             "latent_shape": list(latent_shape) if latent_shape else None,
             "normalization_policy": policy, "stats_over": "train-split context visits (excl. last session)"}
    with open(os.path.join(cache_dir, "_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"pooled mean: {mean:.6f}   pooled std: {std:.6f}")
    print("Paste into the config's context_vae block:")
    print(f"  latent_scale: {1.0 / std:.6f}\n  latent_mean: {mean:.6f}\n  use_latent_mean: true")


if __name__ == "__main__":
    main()
