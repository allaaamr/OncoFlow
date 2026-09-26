import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import preprocessing as preprocessing_utils
from src.data.dataset import _read_split_csv
from src.models.vae_wrapper import FrozenMedVAE3D

FLAIR_INDEX = 2
MIU_DIR = "/home/alaa.mohamed/MIU2"
SPLIT_CSV = "/home/alaa.mohamed/OncoFlow+/src/data/splits/mu_patient_split.csv"
VOLUME_SIZE = (160, 240, 240)
NORMALIZATION_CLIP_PCT = (0.5, 99.5)


def compute_metrics(orig: np.ndarray, recon: np.ndarray) -> dict:
    mask = orig != 0
    if mask.sum() < 50:
        mask = np.ones_like(orig, dtype=bool)

    diff = orig[mask] - recon[mask]
    mse = float(np.mean(diff ** 2))

    if mse < 1e-12:
        psnr = float("inf")
    else:
        psnr = float(10.0 * np.log10(1.0 / mse))

    ssim_val = float(ssim_fn(orig, recon, data_range=1.0))

    return {"mse": mse, "psnr": psnr, "ssim": ssim_val}


def build_vae(device: torch.device) -> FrozenMedVAE3D:
    vae_cfg = {
        "model_name": "medvae_4_1_3d",
        "modality": "mri",
        "gpu_dim": 160,
        "latent_channels": 1,
        "downsample_factor": 4,
    }
    vae = FrozenMedVAE3D.from_config(vae_cfg).to(device)
    vae.eval()
    return vae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miu_dir", default=MIU_DIR)
    parser.add_argument("--split_csv", default=SPLIT_CSV)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="evaluations/medvae_recon")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    _, _, test_ids = _read_split_csv(args.split_csv)
    image_files = {
        os.path.basename(p).replace("_image.npy", ""): p
        for p in glob.glob(os.path.join(args.miu_dir, "*_image.npy"))
    }
    test_ids = [pid for pid in test_ids if pid in image_files]
    print(f"=> {len(test_ids)} test patients with available images (of {len(_read_split_csv(args.split_csv)[2])} in split).")

    vae = build_vae(device)

    per_session_rows = []
    with torch.no_grad():
        for i, pid in enumerate(test_ids):
            img_path = image_files[pid]
            img = np.load(img_path, mmap_mode="r")
            n_sessions = img.shape[0]

            for sess in range(n_sessions):
                volume = np.asarray(img[sess, FLAIR_INDEX, :, :, :]).astype(np.float32)
                volume = preprocessing_utils.normalize_volume_whole(volume, clip_pct=NORMALIZATION_CLIP_PCT)
                volume = preprocessing_utils.crop_pad_volume_3d(volume, VOLUME_SIZE)

                x = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(device)
                z = vae.encode(x)
                recon = vae.decode(z)

                orig_np = x.squeeze().float().cpu().numpy()
                recon_np = recon.squeeze().float().cpu().numpy()
                common_shape = tuple(min(o, r) for o, r in zip(orig_np.shape, recon_np.shape))
                orig_crop = orig_np[: common_shape[0], : common_shape[1], : common_shape[2]]
                recon_crop = np.clip(recon_np[: common_shape[0], : common_shape[1], : common_shape[2]], 0.0, 1.0)

                metrics = compute_metrics(orig_crop, recon_crop)
                per_session_rows.append({"patient_id": pid, "session": sess, **metrics})
                print(
                    f"[{i + 1}/{len(test_ids)}] {pid} sess={sess}: "
                    f"MSE={metrics['mse']:.6f} PSNR={metrics['psnr']:.2f}dB SSIM={metrics['ssim']:.4f}"
                )

    mses = np.array([r["mse"] for r in per_session_rows])
    psnrs = np.array([r["psnr"] for r in per_session_rows if np.isfinite(r["psnr"])])
    ssims = np.array([r["ssim"] for r in per_session_rows])

    print("\n" + "=" * 60)
    print(f"MedVAE (medvae_4_1_3d) reconstruction quality on TEST split")
    print(f"({len(test_ids)} patients, {len(per_session_rows)} sessions, FLAIR modality)")
    print("=" * 60)
    print(f"MSE  : mean={mses.mean():.6f}  std={mses.std():.6f}")
    print(f"PSNR : mean={psnrs.mean():.2f} dB  std={psnrs.std():.2f} dB")
    print(f"SSIM : mean={ssims.mean():.4f}  std={ssims.std():.4f}")
    print("=" * 60)

    per_session_path = os.path.join(args.out_dir, "per_session_metrics.csv")
    with open(per_session_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "session", "mse", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(per_session_rows)

    summary_path = os.path.join(args.out_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std"])
        writer.writerow(["mse", mses.mean(), mses.std()])
        writer.writerow(["psnr_db", psnrs.mean(), psnrs.std()])
        writer.writerow(["ssim", ssims.mean(), ssims.std()])

    print(f"\n=> Saved per-session metrics to {per_session_path}")
    print(f"=> Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
