#!/usr/bin/env python

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import preprocessing as preprocessing_utils
from src.evaluation.metrics import compute_mse, compute_psnr, compute_ssim
from src.models.vae_wrapper import FrozenMedVAE3D

FLAIR_INDEX = 2


@torch.no_grad()
def old_encode_no_domain_fix(vae: FrozenMedVAE3D, x01: torch.Tensor) -> torch.Tensor:
    from monai.inferers import sliding_window_inference
    from medvae.utils.extras import roi_size_calc

    def predict_mode(patch: torch.Tensor) -> torch.Tensor:
        return vae.mvae.model.encode(patch).mode()

    roi_size = roi_size_calc(x01.shape[-3:], target_gpu_dim=vae.mvae.gpu_dim)
    return sliding_window_inference(
        inputs=x01, roi_size=roi_size, sw_batch_size=1, mode="gaussian", predictor=predict_mode,
    )


@torch.no_grad()
def old_decode_no_domain_fix(vae: FrozenMedVAE3D, z: torch.Tensor) -> torch.Tensor:
    recon_native = vae.mvae.decode(z)
    return recon_native.clamp(0.0, 1.0)


def laplacian_variance(slice_2d: np.ndarray) -> float:
    s = slice_2d.astype(np.float64)
    lap = (
        -4.0 * s[1:-1, 1:-1]
        + s[:-2, 1:-1] + s[2:, 1:-1]
        + s[1:-1, :-2] + s[1:-1, 2:]
    )
    return float(lap.var())


def radial_power_spectrum(slice_2d: np.ndarray, n_bins: int = 80):
    F = np.fft.fftshift(np.fft.fft2(slice_2d.astype(np.float64)))
    power = np.abs(F) ** 2
    h, w = slice_2d.shape
    yy, xx = np.meshgrid(np.arange(h) - h / 2.0, np.arange(w) - w / 2.0, indexing="ij")
    r = np.sqrt(yy ** 2 + xx ** 2)
    r_max = min(h, w) / 2.0
    bins = np.linspace(0, r_max, n_bins + 1)
    bin_idx = np.digitize(r.ravel(), bins) - 1
    power_flat = power.ravel()
    radial_mean = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.any():
            radial_mean[i] = power_flat[mask].mean()
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, radial_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="/home/alaa.mohamed/MIU2")
    parser.add_argument("--patient", default="PatientID_0003")
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--volume_size", type=int, nargs=3, default=(160, 240, 240))
    parser.add_argument("--clip_pct", type=float, nargs=2, default=(0.5, 99.5))
    parser.add_argument("--zoom_frac", type=float, default=0.35,
                         help="Fraction of the axial slice's side length used for the zoomed crop (row 4).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_path", default="scripts/medvae_debug_out/preprocessing_fix_comparison.png")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    img_path = os.path.join(args.data_dir, f"{args.patient}_image.npy")
    img = np.load(img_path, mmap_mode="r")
    raw_vol = np.asarray(img[args.session, FLAIR_INDEX]).astype(np.float32)
    x01_np = preprocessing_utils.normalize_volume_whole(raw_vol, clip_pct=tuple(args.clip_pct))
    x01_np = preprocessing_utils.crop_pad_volume_3d(x01_np, tuple(args.volume_size))
    print(f"[GT] {args.patient} sess={args.session}: shape={x01_np.shape} "
          f"min={x01_np.min():.4f} max={x01_np.max():.4f} mean={x01_np.mean():.4f}")

    x01 = torch.from_numpy(x01_np).float()[None, None].to(device)

    vae_cfg = {"model_name": "medvae_4_1_3d", "modality": "mri", "gpu_dim": args.volume_size[0],
               "latent_channels": 1, "downsample_factor": 4}
    vae = FrozenMedVAE3D.from_config(vae_cfg).to(device).eval()

    with torch.no_grad():
        z_old = old_encode_no_domain_fix(vae, x01)
        recon_old = old_decode_no_domain_fix(vae, z_old)

        z_new = vae.encode(x01)
        recon_new = vae.decode(z_new)

    common = tuple(min(a, b) for a, b in zip(x01_np.shape, recon_old.shape[-3:]))
    gt_c = x01_np[:common[0], :common[1], :common[2]]
    old_c = recon_old.squeeze().float().cpu().numpy()[:common[0], :common[1], :common[2]]
    new_c = recon_new.squeeze().float().cpu().numpy()[:common[0], :common[1], :common[2]]

    def volume_metrics(pred: np.ndarray) -> dict:
        return {
            "mse": compute_mse(gt_c, pred),
            "psnr": compute_psnr(gt_c, pred),
            "ssim": compute_ssim(gt_c, pred, device=str(device), spatial_dims=3),
        }

    m_old = volume_metrics(old_c)
    m_new = volume_metrics(new_c)

    print(f"\n{'':10}{'MSE':>10}{'PSNR':>10}{'SSIM':>10}")
    print(f"{'OLD':10}{m_old['mse']:>10.5f}{m_old['psnr']:>10.2f}{m_old['ssim']:>10.4f}")
    print(f"{'NEW':10}{m_new['mse']:>10.5f}{m_new['psnr']:>10.2f}{m_new['ssim']:>10.4f}")

    D, H, W = gt_c.shape
    zd, zh, zw = D // 2, H // 2, W // 2
    views = {
        "Axial": (gt_c[zd], old_c[zd], new_c[zd]),
        "Coronal": (gt_c[:, zh, :], old_c[:, zh, :], new_c[:, zh, :]),
        "Sagittal": (gt_c[:, :, zw], old_c[:, :, zw], new_c[:, :, zw]),
    }

    lap_gt = laplacian_variance(views["Axial"][0])
    lap_old = laplacian_variance(views["Axial"][1])
    lap_new = laplacian_variance(views["Axial"][2])
    print(f"\nLaplacian variance (axial slice, higher = sharper): "
          f"GT={lap_gt:.6f}  OLD={lap_old:.6f}  NEW={lap_new:.6f}")
    print(f"  NEW/GT = {lap_new / lap_gt:.3f}   OLD/GT = {lap_old / lap_gt:.3f}")

    crop_h = int(H * args.zoom_frac)
    crop_w = int(W * args.zoom_frac)
    ch0, cw0 = (H - crop_h) // 2, (W - crop_w) // 2
    zoom_gt = views["Axial"][0][ch0:ch0 + crop_h, cw0:cw0 + crop_w]
    zoom_old = views["Axial"][1][ch0:ch0 + crop_h, cw0:cw0 + crop_w]
    zoom_new = views["Axial"][2][ch0:ch0 + crop_h, cw0:cw0 + crop_w]

    r_gt, p_gt = radial_power_spectrum(views["Axial"][0])
    r_old, p_old = radial_power_spectrum(views["Axial"][1])
    r_new, p_new = radial_power_spectrum(views["Axial"][2])

    fig = plt.figure(figsize=(13, 17))
    gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 1, 1.1])

    def imshow_row(row, imgs, row_title_prefix, extra_titles=None):
        for c, (name, im) in enumerate(imgs.items()):
            ax = fig.add_subplot(gs[row, c])
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            title = f"{row_title_prefix} {name}"
            if extra_titles is not None:
                title += f"\n{extra_titles}"
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    imshow_row(0, {k: v[0] for k, v in views.items()}, "GT")
    imshow_row(1, {k: v[1] for k, v in views.items()}, "OLD (pre-fix)",
               f"MSE={m_old['mse']:.4f} PSNR={m_old['psnr']:.2f} SSIM={m_old['ssim']:.4f}")
    imshow_row(2, {k: v[2] for k, v in views.items()}, "NEW (fixed)",
               f"MSE={m_new['mse']:.4f} PSNR={m_new['psnr']:.2f} SSIM={m_new['ssim']:.4f}")

    ax = fig.add_subplot(gs[3, 0]); ax.imshow(zoom_gt, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"GT zoom\nLaplacian-var={lap_gt:.5f}", fontsize=9); ax.axis("off")
    ax = fig.add_subplot(gs[3, 1]); ax.imshow(zoom_old, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"OLD zoom\nLaplacian-var={lap_old:.5f} ({lap_old/lap_gt:.2f}x GT)", fontsize=9); ax.axis("off")
    ax = fig.add_subplot(gs[3, 2]); ax.imshow(zoom_new, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"NEW zoom\nLaplacian-var={lap_new:.5f} ({lap_new/lap_gt:.2f}x GT)", fontsize=9); ax.axis("off")

    err_old = np.abs(views["Axial"][0] - views["Axial"][1])
    err_new = np.abs(views["Axial"][0] - views["Axial"][2])
    vmax_err = max(err_old.max(), err_new.max(), 1e-6)
    ax = fig.add_subplot(gs[4, 0]); im = ax.imshow(err_old, cmap="magma", vmin=0, vmax=vmax_err)
    ax.set_title("|GT-OLD| axial", fontsize=9); ax.axis("off"); fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(gs[4, 1]); im = ax.imshow(err_new, cmap="magma", vmin=0, vmax=vmax_err)
    ax.set_title("|GT-NEW| axial", fontsize=9); ax.axis("off"); fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[4, 2])
    ax.semilogy(r_gt, p_gt + 1e-12, label="GT", color="black", linewidth=1.5)
    ax.semilogy(r_old, p_old + 1e-12, label="OLD (pre-fix)", color="tab:red")
    ax.semilogy(r_new, p_new + 1e-12, label="NEW (fixed)", color="tab:blue")
    ax.set_xlabel("spatial frequency (radius, px)", fontsize=8)
    ax.set_ylabel("radial power (log)", fontsize=8)
    ax.set_title("Axial power spectrum\n(faster high-freq roll-off = blurrier)", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    fig.suptitle(
        f"{args.patient} sess={args.session} — MedVAE preprocessing fix: OLD (no domain conversion, "
        f"[0,1]->encoder directly) vs NEW (current code, [0,1]<->[-1,1] conversion at the VAE boundary)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out_path, dpi=160, bbox_inches="tight")
    print(f"\n[Saved] {args.out_path}")


if __name__ == "__main__":
    main()
