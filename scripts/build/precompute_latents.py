#!/usr/bin/env python

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import preprocessing as preprocessing_utils
from src.data.dataset import load_image_stats, n4_kwargs_from_dc
from src.models.vae_wrapper import FrozenMedVAE3D, FlairOnlyEncoder3D, build_dummy_vae3d

FLAIR_INDEX_DEFAULT = 2


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def patient_id_from_image_path(img_path: str) -> str:
    return os.path.basename(img_path).replace("_image.npy", "")


def build_encoder(vae_cfg: dict, device: str) -> FlairOnlyEncoder3D:
    backend = vae_cfg.get("backend", "medvae")
    if backend == "medvae":
        vae = FrozenMedVAE3D.from_config(vae_cfg)
    elif backend == "dummy":
        vae = build_dummy_vae3d(vae_cfg)
    else:
        raise ValueError(f"Unknown vae.backend={backend!r}, expected 'medvae' or 'dummy'")
    vae = vae.to(device)
    vae.eval()
    return FlairOnlyEncoder3D(
        vae=vae,
        flair_index=vae_cfg.get("flair_index", FLAIR_INDEX_DEFAULT),
        strict=True,
    )


@torch.no_grad()
def precompute_patient(
    img_path: str,
    encoder: FlairOnlyEncoder3D,
    volume_size,
    clip_pct,
    flair_index: int,
    device: str,
    normalization_policy: str = "whole_volume",
    image_stats: dict = None,
    apply_n4: bool = False,
    n4_cache_dir: str = None,
    n4_kwargs: dict = None,
) -> np.ndarray:
    img_raw = np.load(img_path, mmap_mode="r")
    S, C = img_raw.shape[0], img_raw.shape[1]
    idx = flair_index if C > 1 else 0
    pid = os.path.basename(img_path).replace("_image.npy", "")

    latents = []
    for s in range(S):
        n4_cache_key = (
            preprocessing_utils.compute_n4_cache_key(pid, s, idx, img_path, n4_kwargs) if apply_n4 else None
        )
        vol = preprocessing_utils.apply_normalization_policy(
            normalization_policy,
            raw_volume_fn=lambda s=s: np.asarray(img_raw[s, idx, :, :, :]),
            clip_pct=clip_pct,
            stats=image_stats.get(idx) if image_stats else None,
            apply_n4=apply_n4,
            n4_cache_dir=n4_cache_dir,
            n4_cache_key=n4_cache_key,
            n4_kwargs=n4_kwargs,
            require_n4_cache=True,
        )
        vol = preprocessing_utils.crop_pad_volume_3d(vol, volume_size)
        x = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0).to(device)
        z = encoder(x)
        latents.append(z.squeeze(0).cpu().numpy())

    return np.stack(latents).astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default=None, help="Override data.data_dir")
    parser.add_argument("--out-dir", default=None, help="Override cache dir (default: data.latent_dir or <data_dir>/latents)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true", help="Recompute even if a cached latent file already exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dc = cfg["data"]
    if dc.get("representation", "latent") == "voxel":
        raise ValueError(
            "[precompute_latents] data.representation='voxel' — this script precomputes a "
            "frozen-VAE latent cache, which is meaningless in voxel mode (there is no VAE step "
            "at all; PatientDataset resizes MRIs directly to data.voxel_shape at __getitem__ "
            "time). Nothing to precompute here — do not run this script for voxel-mode configs."
        )
    data_dir = args.data_dir or dc["data_dir"]
    volume_size = tuple(dc.get("volume_size", (160, 240, 240)))
    clip_pct = tuple(dc.get("normalization_clip_pct", (0.5, 99.5)))
    flair_index = dc.get("flair_index", FLAIR_INDEX_DEFAULT)
    out_dir = args.out_dir or dc.get("latent_dir") or os.path.join(data_dir, "latents")
    os.makedirs(out_dir, exist_ok=True)

    normalization_policy = dc.get("normalization_policy", "whole_volume")
    apply_n4 = bool(dc.get("apply_n4_bias_correction", False))
    n4_cache_dir = dc.get("n4_cache_dir")
    n4_kwargs = n4_kwargs_from_dc(dc)
    image_stats = None
    if normalization_policy == "frozen_train_stats":
        image_stats_path = dc.get("image_stats_path")
        if not image_stats_path:
            raise ValueError(
                "[precompute_latents] data.normalization_policy='frozen_train_stats' requires "
                "data.image_stats_path — run `python scripts/fit_image_stats.py --config "
                f"{args.config}` first."
            )
        image_stats, image_stats_meta = load_image_stats(image_stats_path)
        print(
            f"[precompute_latents] Loaded frozen image stats from '{image_stats_path}' "
            f"(fit on {len(image_stats_meta.get('train_patient_ids', []))} train patients)."
        )

    print("=" * 60)
    print("Precomputing frozen-VAE latents")
    print("=" * 60)
    print(f"Data dir:    {data_dir}")
    print(f"Out dir:     {out_dir}")
    print(f"Volume size: {volume_size}")
    print(f"Normalization: {normalization_policy}" + (f" (clip_pct={clip_pct})" if normalization_policy == "whole_volume" else ""))
    print(f"VAE:         {cfg['vae'].get('model_name', 'medvae_4_1_3d')} (backend={cfg['vae'].get('backend', 'medvae')})")
    print(f"Device:      {args.device}")
    print("=" * 60)

    encoder = build_encoder(cfg["vae"], args.device)

    image_files = sorted(glob.glob(os.path.join(data_dir, "*_image.npy")))
    print(f"Found {len(image_files)} patient image files.")

    n_done, n_skipped, n_failed = 0, 0, 0
    latent_shape = None
    for i, img_path in enumerate(image_files):
        pid = patient_id_from_image_path(img_path)
        out_path = os.path.join(out_dir, f"{pid}_latent.npy")
        if os.path.exists(out_path) and not args.force:
            n_skipped += 1
            continue
        try:
            latents = precompute_patient(
                img_path, encoder, volume_size, clip_pct, flair_index, args.device,
                normalization_policy=normalization_policy, image_stats=image_stats,
                apply_n4=apply_n4, n4_cache_dir=n4_cache_dir, n4_kwargs=n4_kwargs,
            )
            np.save(out_path, latents)
            latent_shape = latents.shape[1:]
            n_done += 1
            print(f"[{i + 1}/{len(image_files)}] {pid} -> {latents.shape} ({latents.nbytes / 1e6:.1f} MB)", flush=True)
        except Exception as e:
            n_failed += 1
            print(f"[{i + 1}/{len(image_files)}] {pid} FAILED: {e}", flush=True)

    meta = {
        "model_name": cfg["vae"].get("model_name", "medvae_4_1_3d"),
        "backend": cfg["vae"].get("backend", "medvae"),
        "volume_size": list(volume_size),
        "normalization_policy": normalization_policy,
        "normalization_clip_pct": list(clip_pct),
        "image_stats_path": dc.get("image_stats_path") if normalization_policy == "frozen_train_stats" else None,
        "flair_index": flair_index,
        "downsample_factor": cfg["vae"].get("downsample_factor", 4),
        "latent_channels": cfg["vae"].get("latent_channels", 1),
        "latent_spatial_shape": list(latent_shape) if latent_shape is not None else None,
        "note": "Latents are the RAW (unscaled, uncentered) frozen-VAE posterior mode; "
                "vae.latent_scale/vae.latent_mean are applied at load time in TaGeDiff.forward, not here.",
    }
    with open(os.path.join(out_dir, "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print(f"Done. computed={n_done} skipped={n_skipped} failed={n_failed} -> {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
