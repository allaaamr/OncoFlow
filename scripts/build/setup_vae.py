import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vae_wrapper import FrozenMedVAE3D
from src.data import preprocessing as preprocessing_utils

FLAIR_INDEX = 2


def verify_vae(vae: FrozenMedVAE3D, volume_size=(160, 240, 240), device="cpu"):
    print("\n" + "=" * 60)
    print("Verifying medvae_4_1_3d loads and encode/decode shapes...")
    print("=" * 60)

    x = torch.randn(1, 1, *volume_size, device=device)
    with torch.no_grad():
        z = vae.encode(x)
        recon = vae.decode(z)

    expected_latent = tuple(d // vae.downsample_factor for d in volume_size)
    print(f"  Input shape:   {tuple(x.shape)}")
    print(f"  Latent shape:  {tuple(z.shape)}  (expected spatial dims ~= {expected_latent})")
    print(f"  Recon shape:   {tuple(recon.shape)}")
    print(f"  Latent stats:  mean={z.mean():.3f}, std={z.std():.3f}")

    assert z.shape[1] == vae.latent_channels, f"Unexpected latent channels: {z.shape[1]}"
    assert tuple(z.shape[-3:]) == expected_latent, f"Unexpected latent spatial shape: {z.shape[-3:]}"
    assert recon.shape[:2] == (1, 1), f"Unexpected recon channel shape: {recon.shape[:2]}"

    print("\n  medvae_4_1_3d loaded and verified successfully.")


def compute_scale_factor(vae: FrozenMedVAE3D, data_dir: str, volume_size, num_patients: int, device: str):
    print("\n" + "=" * 60)
    print(f"Computing latent_scale/latent_mean from up to {num_patients} real MIU volumes in {data_dir}...")
    print("=" * 60)

    image_files = sorted(glob.glob(os.path.join(data_dir, "*_image.npy")))[:num_patients]
    if not image_files:
        print(f"  No *_image.npy files found in {data_dir}; skipping real-data scale computation.")
        return None

    sum_x = 0.0
    sum_sq = 0.0
    count = 0

    with torch.no_grad():
        for img_path in image_files:
            img = np.load(img_path, mmap_mode="r")
            volume = np.asarray(img[0, FLAIR_INDEX, :, :, :]).astype(np.float32)
            volume = preprocessing_utils.normalize_volume_whole(volume)

            D, H, W = volume.shape
            tD, tH, tW = volume_size
            sd, sh, sw = max(0, (D - tD) // 2), max(0, (H - tH) // 2), max(0, (W - tW) // 2)
            volume = volume[sd:sd + min(D, tD), sh:sh + min(H, tH), sw:sw + min(W, tW)]
            pad = [(0, max(0, t - s)) for s, t in zip(volume.shape, volume_size)]
            volume = np.pad(volume, pad, mode="constant", constant_values=0)

            x = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(device)
            z = vae.encode(x)

            sum_x += z.sum().item()
            sum_sq += (z ** 2).sum().item()
            count += z.numel()
            print(f"  encoded {os.path.basename(img_path)} -> latent {tuple(z.shape)}")

    mean = sum_x / count
    var = sum_sq / count - mean ** 2
    std = max(var, 1e-8) ** 0.5
    scale_factor = 1.0 / std

    print(f"\n  latent mean={mean:.4f} std={std:.4f}")
    print(f"  latent_scale = {scale_factor:.4f}")
    print(f"  latent_mean  = {mean:.4f}")
    print(f"  -> paste BOTH into configs/default.yaml's vae.latent_scale and vae.latent_mean")
    return scale_factor, mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="/home/alaa.mohamed/MIU2")
    parser.add_argument("--volume_size", type=int, nargs=3, default=(160, 240, 240))
    parser.add_argument("--num_scale_patients", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_real_data", action="store_true", help="Only run the synthetic shape check.")
    args = parser.parse_args()

    vae_cfg = {"model_name": "medvae_4_1_3d", "modality": "mri", "gpu_dim": 160, "latent_channels": 1, "downsample_factor": 4}
    vae = FrozenMedVAE3D.from_config(vae_cfg).to(args.device)

    verify_vae(vae, volume_size=tuple(args.volume_size), device=args.device)

    if not args.skip_real_data:
        compute_scale_factor(vae, args.data_dir, tuple(args.volume_size), args.num_scale_patients, args.device)


if __name__ == "__main__":
    main()
