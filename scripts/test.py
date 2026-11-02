import argparse
import os
import sys
import csv
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

from scipy.ndimage import binary_dilation, binary_erosion

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.TaGeDiff import TaGeDiff
from src.data import preprocessing as preprocessing_utils
from src.data.dataset import n4_kwargs_from_dc
from src.evaluation.metrics import (
    compute_mse, compute_psnr, compute_ssim, compute_ms_ssim as compute_msssim,
    resolve_data_range,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

FLAIR_INDEX = 2

TREATMENT_NAMES = {0: "CRT", 1: "TMZ", 2: "IMT"}

ROI_DILATION_ITERS = 3

NATIVE_SHAPE_MM_ISOTROPIC = (155, 240, 240)


def treatment_name(x):
    return TREATMENT_NAMES.get(int(x), f"Unknown({x})")


def resolve_voxel_spacing_mm(volume_size, spacing_override=None) -> tuple:
    if spacing_override is not None:
        return tuple(float(x) for x in spacing_override)
    return tuple(float(n) / float(w) for n, w in zip(NATIVE_SHAPE_MM_ISOTROPIC, volume_size))


def _to_display(img_2d, fg_mask=None):
    img_2d = np.asarray(img_2d, dtype=np.float32)
    fg = fg_mask if fg_mask is not None else (img_2d != 0.0)
    out = np.zeros_like(img_2d)
    if not fg.any():
        return out
    lo, hi = float(img_2d[fg].min()), float(img_2d[fg].max())
    if hi - lo < 1e-6:
        out[fg] = 0.5
        return out
    out[fg] = (img_2d[fg] - lo) / (hi - lo)
    return out

def mask_boundary(mask: np.ndarray, width: int = 2) -> np.ndarray:
    mask = mask.astype(bool)
    eroded = binary_erosion(mask, iterations=width)
    return mask & (~eroded)


def make_tumor_change_maps(last_vol, gt_vol, pred_vol, gt_tumor_mask, dilation_iters=ROI_DILATION_ITERS, eps=0.03, wrong_dir_thresh=0.1):
    last = np.asarray(last_vol, dtype=np.float32)
    gt = np.asarray(gt_vol, dtype=np.float32)
    pred = np.asarray(pred_vol, dtype=np.float32)

    tumor = gt_tumor_mask.astype(bool)
    roi = binary_dilation(tumor, iterations=dilation_iters)
    border = mask_boundary(roi, width=1)

    delta_gt = gt - last
    delta_pred = pred - last

    gt_growth = (delta_gt > eps) & roi
    gt_shrink = (delta_gt < -eps) & roi
    gt_stable = roi & ~(gt_growth | gt_shrink)

    pred_growth = (delta_pred > eps) & roi
    pred_shrink = (delta_pred < -eps) & roi
    pred_stable = roi & ~(pred_growth | pred_shrink)

    correct_growth = gt_growth & pred_growth
    correct_shrink = gt_shrink & pred_shrink
    missed_growth = gt_growth & ~pred_growth
    missed_shrink = gt_shrink & ~pred_shrink
    wrong_direction_flip = (gt_growth & pred_shrink) | (gt_shrink & pred_growth)
    wrong_direction = wrong_direction_flip & (np.abs(delta_pred) > wrong_dir_thresh)
    correct_any = correct_growth | correct_shrink
    missed_any = (missed_growth | missed_shrink) & ~wrong_direction

    stats = {
        "roi_voxels": int(roi.sum()),
        "gt_growth_voxels": int(gt_growth.sum()),
        "gt_shrink_voxels": int(gt_shrink.sum()),
        "growth_recall": float(correct_growth.sum() / max(gt_growth.sum(), 1)),
        "shrink_recall": float(correct_shrink.sum() / max(gt_shrink.sum(), 1)),
        "change_recall": float(correct_any.sum() / max(gt_growth.sum() + gt_shrink.sum(), 1)),
        "wrong_direction_rate": float(wrong_direction.sum() / max(gt_growth.sum() + gt_shrink.sum(), 1)),
        "tumor_delta_mae": float(np.mean(np.abs(delta_gt[roi] - delta_pred[roi]))) if roi.any() else 0.0,
    }

    return {
        "roi": roi, "border": border,
        "gt_growth": gt_growth, "gt_shrink": gt_shrink, "gt_stable": gt_stable,
        "pred_growth": pred_growth, "pred_shrink": pred_shrink, "pred_stable": pred_stable,
        "correct_any": correct_any, "missed_any": missed_any, "wrong_direction": wrong_direction,
        "delta_gt": delta_gt, "delta_pred": delta_pred,
    }, stats


def resolve_axial_idx(roi_mask_3d: np.ndarray, D: int) -> int:
    roi_depths = np.where(roi_mask_3d.any(axis=(1, 2)))[0]
    return int(roi_depths[len(roi_depths) // 2]) if len(roi_depths) > 0 else D // 2


def resolve_change_region_mask(baseline_mask: np.ndarray = None, followup_mask: np.ndarray = None):
    parts, used = [], []
    if baseline_mask is not None:
        parts.append(baseline_mask.astype(bool)); used.append("baseline")
    if followup_mask is not None:
        parts.append(followup_mask.astype(bool)); used.append("followup")
    if not parts:
        raise ValueError("resolve_change_region_mask: need at least one of baseline_mask/followup_mask")
    union = parts[0].copy()
    for p in parts[1:]:
        union |= p
    return union, ("+".join(used) if union.any() else "none")


MR_MIN_REGION_VOXELS = 100
MR_STD_EPS = 1e-6


def compute_change_magnitude_ratio(delta_gt: np.ndarray, delta_pred: np.ndarray, region_mask: np.ndarray) -> dict:
    region_mask = region_mask.astype(bool)
    n_voxels = int(region_mask.sum())

    if n_voxels < MR_MIN_REGION_VOXELS:
        std_pred = std_gt = float("nan")
        nan_reason = f"region has {n_voxels} voxel(s) < MR_MIN_REGION_VOXELS={MR_MIN_REGION_VOXELS}"
        return {"mr": float("nan"), "std_pred": std_pred, "std_gt": std_gt,
                "ccc": float("nan"), "n_voxels": n_voxels, "nan_reason": nan_reason}

    dp = np.asarray(delta_pred, dtype=np.float64)[region_mask]
    dg = np.asarray(delta_gt, dtype=np.float64)[region_mask]
    std_pred, std_gt = float(dp.std()), float(dg.std())

    if std_gt < MR_STD_EPS:
        nan_reason = f"std(d_gt)={std_gt:.3g} < MR_STD_EPS={MR_STD_EPS:.0e}"
        return {"mr": float("nan"), "std_pred": std_pred, "std_gt": std_gt,
                "ccc": float("nan"), "n_voxels": n_voxels, "nan_reason": nan_reason}

    mr = std_pred / std_gt
    mean_p, mean_g = float(dp.mean()), float(dg.mean())
    cov = float(np.mean((dp - mean_p) * (dg - mean_g)))
    var_p, var_g = float(dp.var()), float(dg.var())
    ccc = 2.0 * cov / (var_p + var_g + (mean_p - mean_g) ** 2)

    return {"mr": float(mr), "std_pred": std_pred, "std_gt": std_gt,
            "ccc": float(ccc), "n_voxels": n_voxels, "nan_reason": ""}


def _mr_color(mr: float) -> str:
    if not np.isfinite(mr):
        return "#cccccc"
    d = abs(mr - 1.0)
    if d <= 0.15:
        return "#98fb98"
    if d <= 0.35:
        return "#ffbf47"
    return "#ff7f7f"


def tumor_bbox_slices(mask: np.ndarray, margin: int = 8):
    if not mask.any():
        return None
    idxs = np.where(mask)
    return tuple(
        slice(max(0, int(ax.min()) - margin), min(mask.shape[d] - 1, int(ax.max()) + margin) + 1)
        for d, ax in enumerate(idxs)
    )


def compute_tumor_region_metrics(gt_dhw, pred_dhw, tumor_mask_dhw, device: str, data_range: float):
    bbox = tumor_bbox_slices(tumor_mask_dhw)
    if bbox is None:
        nan = float("nan")
        return {"mse": nan, "ssim": nan, "ms_ssim": nan, "psnr": nan}
    gt_box = gt_dhw[bbox]
    pred_box = pred_dhw[bbox]
    return {
        "mse": compute_mse(gt_box, pred_box),
        "ssim": compute_ssim(gt_box, pred_box, device=device, spatial_dims=3, data_range=data_range),
        "ms_ssim": compute_msssim(gt_box, pred_box, device=device, spatial_dims=3, data_range=data_range),
        "psnr": compute_psnr(gt_box, pred_box, data_range=data_range),
    }


def _overlay_2d(base_img_2d, masks_colors, border_2d, body_alpha=0.55, border_alpha=0.9, fg_mask=None):
    base = _to_display(base_img_2d, fg_mask=fg_mask)
    rgb = np.stack([base, base, base], axis=-1)
    for mask_2d, color in masks_colors:
        if mask_2d.any():
            rgb[mask_2d] = (1 - body_alpha) * rgb[mask_2d] + body_alpha * np.array(color, dtype=np.float32)
    pink = np.array([1.0, 0.25, 0.70], dtype=np.float32)
    if border_2d.any():
        rgb[border_2d] = (1 - border_alpha) * rgb[border_2d] + border_alpha * pink
    return rgb.clip(0, 1)


def save_clinical_report(last_vol, gt_vol, pred_vol, maps, stats, save_path, patient_id, t_idx,
                          steps, last_treatment=None, target_treatment=None, metrics=None,
                          tumor_metrics=None, axial_idx=None, change_magnitude=None):
    if not HAS_MPL:
        return

    D = gt_vol.shape[0]
    if axial_idx is None:
        axial_idx = resolve_axial_idx(maps["roi"], D)

    def s2(vol_or_mask_3d):
        return vol_or_mask_3d[axial_idx]

    gt_2d, pred_2d, last_2d = s2(gt_vol), s2(pred_vol), s2(last_vol)
    border_2d = s2(maps["border"])

    bg_fg_mask = gt_2d != 0.0

    top_imgs = [
        _overlay_2d(last_2d, [(s2(maps["roi"]), (1.0, 0.25, 0.70))], border_2d, body_alpha=0.22, fg_mask=bg_fg_mask),
        _overlay_2d(gt_2d, [(s2(maps["gt_growth"]), (0, 0.85, 0)), (s2(maps["gt_shrink"]), (0.1, 0.25, 1.0)),
                             (s2(maps["gt_stable"]), (0.8, 0.8, 0.8))], border_2d, fg_mask=bg_fg_mask),
        _overlay_2d(pred_2d, [(s2(maps["correct_any"]), (0, 0.9, 0)), (s2(maps["missed_any"]), (1.0, 0.05, 0.05)),
                               (s2(maps["wrong_direction"]), (0.1, 0.25, 1.0))], border_2d, body_alpha=0.65,
                    fg_mask=bg_fg_mask),
    ]
    bottom_imgs = [
        _to_display(last_2d, fg_mask=bg_fg_mask),
        _to_display(gt_2d, fg_mask=bg_fg_mask),
        _to_display(pred_2d, fg_mask=bg_fg_mask),
    ]
    col_titles = [
        f"LAST VISIT (t={t_idx-1})\n{treatment_name(last_treatment)}",
        f"GROUND TRUTH (t={t_idx})\n{treatment_name(target_treatment)}",
        f"PREDICTION (t={t_idx})\n{treatment_name(target_treatment)}",
    ]

    fig = plt.figure(figsize=(14, 9), facecolor="black")
    fig.suptitle(
        f"PATIENT: {patient_id} | TARGET t={t_idx} | AXIAL SLICE: {axial_idx}/{D} (ROI-centered) | STEPS: {steps}",
        fontsize=13, color="white", fontweight="bold",
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, width_ratios=[1, 1, 1, 0.7], height_ratios=[0.12, 1, 1],
                            left=0.03, right=0.98, top=0.90, bottom=0.06, wspace=0.06, hspace=0.1)

    for c in range(3):
        axh = fig.add_subplot(gs[0, c]); axh.set_facecolor("black")
        axh.text(0.5, 0.5, col_titles[c], color="white", ha="center", va="center", fontsize=11, fontweight="bold")
        axh.axis("off")
    for c in range(3):
        ax = fig.add_subplot(gs[1, c]); ax.imshow(top_imgs[c]); ax.axis("off")
    for c in range(3):
        ax = fig.add_subplot(gs[2, c]); ax.imshow(bottom_imgs[c], cmap="gray", vmin=0, vmax=1); ax.axis("off")

    legend_ax = fig.add_subplot(gs[:, 3]); legend_ax.set_facecolor("black"); legend_ax.axis("off")
    y = 0.98
    def txt(s, dy=0.045, size=10, color="white", weight="normal"):
        nonlocal y
        legend_ax.text(0.02, y, s, color=color, fontsize=size, fontweight=weight, va="top", ha="left", transform=legend_ax.transAxes)
        y -= dy

    txt("CHANGE METRICS (full 3D ROI)", dy=0.05, size=11, weight="bold")
    txt(f"ROI voxels: {stats['roi_voxels']}", dy=0.04)
    txt(f"Growth recall: {stats['growth_recall']:.3f}", dy=0.04, color="#98fb98")
    txt(f"Shrink recall: {stats['shrink_recall']:.3f}", dy=0.04, color="#99aaff")
    txt(f"Overall recall: {stats['change_recall']:.3f}", dy=0.04, color="#d9f99d")
    txt(f"Wrong dir. rate: {stats['wrong_direction_rate']:.3f}", dy=0.04, color="#ff7f7f")
    txt(f"Tumor delta MAE: {stats['tumor_delta_mae']:.4f}", dy=0.05)

    if metrics is not None:
        txt("VOLUME METRICS (whole brain)", dy=0.05, size=11, weight="bold")
        txt(f"MSE: {metrics['mse']:.6f}", dy=0.04)
        txt(f"SSIM: {metrics['ssim']:.4f}", dy=0.04)
        txt(f"MS-SSIM: {metrics.get('msssim', float('nan')):.4f}", dy=0.04)
        txt(f"PSNR: {metrics['psnr']:.2f} dB", dy=0.04)

    if tumor_metrics is not None:
        txt("VOLUME METRICS (tumor region)", dy=0.05, size=11, weight="bold")
        txt(f"MSE: {tumor_metrics['mse']:.6f}", dy=0.04)
        txt(f"SSIM: {tumor_metrics['ssim']:.4f}", dy=0.04)
        txt(f"MS-SSIM: {tumor_metrics.get('ms_ssim', float('nan')):.4f}", dy=0.04)
        txt(f"PSNR: {tumor_metrics['psnr']:.2f} dB", dy=0.04)

    if change_magnitude is not None:
        cm_t, cm_d = change_magnitude["tumor"], change_magnitude["dilated"]
        dilation_mm = change_magnitude["dilation_mm"]
        txt("CHANGE MAGNITUDE", dy=0.05, size=11, weight="bold")
        txt(f"MR (tumor): {cm_t['mr']:.2f}  [std_pred {cm_t['std_pred']:.3f} / std_gt {cm_t['std_gt']:.3f}]",
            dy=0.04, color=_mr_color(cm_t["mr"]))
        txt(f"MR (dilated {dilation_mm:.1f}mm): {cm_d['mr']:.2f}  [std_pred {cm_d['std_pred']:.3f} / std_gt {cm_d['std_gt']:.3f}]",
            dy=0.04, color=_mr_color(cm_d["mr"]))
        txt(f"CCC (tumor): {cm_t['ccc']:.2f}", dy=0.04)
        txt(f"CCC (dilated): {cm_d['ccc']:.2f}", dy=0.05)

    plt.savefig(save_path, bbox_inches="tight", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


DEFAULT_ERR_THRESHOLD = 0.02


def save_change_maps_figure(delta_gt, delta_pred, err, roi_mask_3d, axial_idx, save_path,
                             patient_id, t_idx, steps, err_threshold=DEFAULT_ERR_THRESHOLD):
    if not HAS_MPL:
        return

    pink = "#ff40b3"

    def s2(v):
        return v[axial_idx]

    roi = roi_mask_3d.astype(bool)
    if roi.any():
        vmax_change = float(np.percentile(np.abs(delta_gt[roi]), 99))
        vmax_err = float(np.percentile(np.abs(err[roi]), 99))
    else:
        vmax_change = float(np.percentile(np.abs(delta_gt), 99)) if delta_gt.size else 1.0
        vmax_err = float(np.percentile(np.abs(err), 99)) if err.size else 1.0
    vmax_change = max(vmax_change, 1e-6)
    vmax_err = max(vmax_err, err_threshold + 1e-6)

    dgt_2d, dpred_2d, err_2d, roi_2d = s2(delta_gt), s2(delta_pred), s2(err), s2(roi)
    abs_err_2d = np.abs(err_2d)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), facecolor="black")
    fig.suptitle(
        f"PATIENT: {patient_id} | TARGET t={t_idx} | AXIAL SLICE: {axial_idx} | CHANGE MAPS | STEPS: {steps}",
        fontsize=13, color="white", fontweight="bold",
    )

    def panel(ax, img_2d, title, vmin, vmax, cmap_name="coolwarm", mask_below=None):
        ax.set_facecolor("black")
        if mask_below is not None:
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad(color="black")
            img_2d = np.ma.masked_where(img_2d < mask_below, img_2d)
        else:
            cmap = cmap_name
        im = ax.imshow(img_2d, cmap=cmap, vmin=vmin, vmax=vmax)
        if roi_2d.any():
            ax.contour(roi_2d.astype(float), levels=[0.5], colors=[pink], linewidths=1.2)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.axis("off")
        return im

    im0 = panel(axes[0], dgt_2d, "Δ true (I1 − I0)", -vmax_change, vmax_change)
    im1 = panel(axes[1], dpred_2d, "Δ predicted (Î1 − I0)", -vmax_change, vmax_change)
    im2 = panel(axes[2], abs_err_2d, "|error| (|Î1 − I1|)", err_threshold, vmax_err,
                cmap_name="inferno", mask_below=err_threshold)

    shared_cbar = fig.colorbar(im1, ax=[axes[0], axes[1]], fraction=0.046, pad=0.03)
    shared_cbar.set_label(f"Δ intensity  [-{vmax_change:.3g}, {vmax_change:.3g}]", color="white", fontsize=9)
    shared_cbar.ax.tick_params(colors="white", labelsize=8)
    shared_cbar.outline.set_edgecolor("white")

    err_cbar = fig.colorbar(im2, ax=axes[2], fraction=0.09, pad=0.03)
    err_cbar.set_label(f"|error|  [{err_threshold:.3g}, {vmax_err:.3g}]", color="white", fontsize=9)
    err_cbar.ax.tick_params(colors="white", labelsize=8)
    err_cbar.outline.set_edgecolor("white")

    plt.savefig(save_path, bbox_inches="tight", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_error_map_figure(err, roi_mask_3d, axial_idx, save_path, patient_id, t_idx, steps,
                           err_threshold=DEFAULT_ERR_THRESHOLD):
    if not HAS_MPL:
        return

    pink = "#ff40b3"
    roi = roi_mask_3d.astype(bool)
    if roi.any():
        vmax_err = float(np.percentile(np.abs(err[roi]), 99))
    else:
        vmax_err = float(np.percentile(np.abs(err), 99)) if err.size else 1.0
    vmax_err = max(vmax_err, err_threshold + 1e-6)

    err_2d, roi_2d = err[axial_idx], roi[axial_idx]
    abs_err_2d = np.abs(err_2d)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7), facecolor="black")
    fig.suptitle(
        f"PATIENT: {patient_id} | TARGET t={t_idx} | AXIAL SLICE: {axial_idx} | "
        f"|PREDICTION ERROR| (|Î1 − I1|) | THRESH: {err_threshold:.3g} | STEPS: {steps}",
        fontsize=11, color="white", fontweight="bold",
    )
    ax.set_facecolor("black")
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(color="black")
    masked_err_2d = np.ma.masked_where(abs_err_2d < err_threshold, abs_err_2d)
    im = ax.imshow(masked_err_2d, cmap=cmap, vmin=err_threshold, vmax=vmax_err)
    if roi_2d.any():
        ax.contour(roi_2d.astype(float), levels=[0.5], colors=[pink], linewidths=1.2)
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"|error|  [{err_threshold:.3g}, {vmax_err:.3g}]", color="white", fontsize=9)
    cbar.ax.tick_params(colors="white", labelsize=8)
    cbar.outline.set_edgecolor("white")

    plt.savefig(save_path, bbox_inches="tight", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list) -> dict:
    for o in overrides:
        key, val = o.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() in ("true", "false"):
                    val = val.lower() == "true"
        d[keys[-1]] = val
    return cfg


def read_patient_ids_from_split(csv_path: str, patient_id_col: str = "patient_id",
                                 split_col: str = "split", split: str = None) -> list:
    with open(str(csv_path), "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        fieldnames = reader.fieldnames
        pid_col = patient_id_col if patient_id_col in fieldnames else fieldnames[0]
        has_split = split_col is not None and split_col in fieldnames
        patient_ids = []
        for row in reader:
            if split is not None and has_split and str(row[split_col]).strip() != str(split):
                continue
            pid = str(row[pid_col]).strip()
            if pid:
                patient_ids.append(pid)
    seen, unique_ids = set(), []
    for pid in patient_ids:
        if pid not in seen:
            seen.add(pid); unique_ids.append(pid)
    if not unique_ids:
        raise ValueError(f"No patient IDs found in {csv_path}")
    return unique_ids


def resolve_ema_policy(ema_policy: str, ckpt: dict) -> bool:
    has_ema = bool(ckpt.get("ema"))
    if ema_policy == "force_raw":
        return False
    if ema_policy == "force_ema":
        if not has_ema:
            raise RuntimeError(
                "--ema-policy=force_ema was requested but this checkpoint has no (non-empty) "
                "'ema' state. Refusing to silently fall back to raw weights."
            )
        return True
    if ema_policy == "auto":
        if not has_ema:
            print("[InferViz] --ema-policy=auto: no EMA state in checkpoint, evaluating RAW weights.")
        return has_ema
    raise ValueError(f"Unknown ema_policy={ema_policy!r}, expected auto/force_ema/force_raw")


def read_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu")


def resolve_training_mode(cfg: dict, ckpt: dict, checkpoint_path: str) -> dict:
    ckpt_mode = ckpt.get("training_mode")
    cfg.setdefault("training", {})
    cfg_mode = cfg["training"].get("training_mode", "diffusion")
    if ckpt_mode is None:
        print(f"[InferViz] WARNING: checkpoint has no saved training_mode; trusting --config={cfg_mode!r}.")
        return cfg
    if ckpt_mode != cfg_mode:
        print(f"[InferViz] CORRECTING training.training_mode: {cfg_mode!r} -> {ckpt_mode!r} (from checkpoint).")
    cfg["training"]["training_mode"] = ckpt_mode
    objective_cfg = ckpt.get("objective_cfg")
    if objective_cfg:
        section = "flow_matching" if ckpt_mode == "flow_matching" else "diffusion"
        cfg.setdefault(section, {})
        cfg[section] = {**cfg[section], **objective_cfg}
    return cfg


def resolve_representation(cfg: dict, ckpt: dict, checkpoint_path: str) -> dict:
    ckpt_repr = ckpt.get("representation")
    cfg.setdefault("data", {})
    cfg_repr = cfg["data"].get("representation", "latent")

    if ckpt_repr is None:
        print(
            f"[InferViz] WARNING: checkpoint '{checkpoint_path}' has no saved representation "
            f"(pre-voxel-mode). Trusting --config's data.representation={cfg_repr!r}."
        )
        return cfg

    if ckpt_repr != cfg_repr:
        print(
            f"[InferViz] CORRECTING data.representation: --config says {cfg_repr!r} but "
            f"checkpoint '{checkpoint_path}' was trained with {ckpt_repr!r}. Using the "
            "checkpoint's value."
        )
    cfg["data"]["representation"] = ckpt_repr

    ckpt_voxel_shape = ckpt.get("voxel_shape")
    if ckpt_repr == "voxel" and ckpt_voxel_shape:
        cfg["data"]["voxel_shape"] = list(ckpt_voxel_shape)

    return cfg


def resolve_condition_on_treatment(cfg: dict, ckpt: dict, checkpoint_path: str) -> dict:
    ckpt_cot = ckpt.get("condition_on_treatment")
    cfg.setdefault("context", {})
    cfg_cot = cfg["context"].get("condition_on_treatment", True)

    if ckpt_cot is None:
        print(
            f"[InferViz] WARNING: checkpoint '{checkpoint_path}' has no saved "
            f"condition_on_treatment (pre-this-feature). Trusting --config's "
            f"context.condition_on_treatment={cfg_cot!r}."
        )
        return cfg

    if ckpt_cot != cfg_cot:
        print(
            f"[InferViz] CORRECTING context.condition_on_treatment: --config says {cfg_cot!r} "
            f"but checkpoint '{checkpoint_path}' was trained with {ckpt_cot!r}. Using the "
            "checkpoint's value."
        )
    cfg["context"]["condition_on_treatment"] = ckpt_cot

    return cfg


def resolve_temporal_aggregator_variant(cfg: dict, ckpt: dict, checkpoint_path: str) -> dict:
    ckpt_variant = ckpt.get("temporal_aggregator_variant")
    cfg.setdefault("temporal_aggregator", {})
    cfg_variant = cfg["temporal_aggregator"].get("variant", "temporal_spatial")

    if ckpt_variant is None:
        ckpt_variant = "patch_temporal_spatial"
        print(
            f"[InferViz] WARNING: checkpoint '{checkpoint_path}' has no saved "
            f"temporal_aggregator_variant (pre-this-feature) — every such checkpoint used "
            f"'patch_temporal_spatial'. Trusting that, not --config's temporal_aggregator."
            f"variant={cfg_variant!r}."
        )

    if ckpt_variant != cfg_variant:
        print(
            f"[InferViz] CORRECTING temporal_aggregator.variant: --config says {cfg_variant!r} "
            f"but checkpoint '{checkpoint_path}' was trained with {ckpt_variant!r}. Using the "
            "checkpoint's value."
        )
    cfg["temporal_aggregator"]["variant"] = ckpt_variant

    return cfg


def load_checkpoint(model: torch.nn.Module, ckpt: dict, ema_policy: str, device: str):
    schema_version = ckpt.get("schema_version", 1)
    if schema_version < 3:
        raise ValueError(
            f"Checkpoint predates schema v3 (found v{schema_version}) — trained under the OLD "
            "2D pipeline, architecturally incompatible with this 3D pipeline."
        )
    use_ema = resolve_ema_policy(ema_policy, ckpt)
    weights = ckpt["ema"] if use_ema else ckpt["model"]
    current = model.state_dict()
    loaded = skipped = 0
    for k, v in weights.items():
        if k in current:
            current[k].copy_(v); loaded += 1
        else:
            skipped += 1
    model.load_state_dict(current)
    model.to(device)
    tag = "EMA" if use_ema else "raw"
    print(f"[InferViz] WEIGHTS USED: {tag.upper()} (ema_policy={ema_policy}) — "
          f"epoch={ckpt.get('epoch','?')}  loaded={loaded}  skipped={skipped}")
    return {
        "epoch": ckpt.get("epoch", None), "weights_used": tag, "schema_version": schema_version,
        "geno_preproc": preprocessing_utils.GenomicsPreprocessor.from_dict(ckpt.get("geno_preproc")),
        "image_stats": (
            {int(k): preprocessing_utils.ImageIntensityStats.from_dict(v) for k, v in ckpt["image_stats"].items()}
            if ckpt.get("image_stats") else None
        ),
        "normalization_policy": ckpt.get("normalization_policy", "whole_volume"),
        "normalization_clip_pct": tuple(ckpt.get("normalization_clip_pct", (0.5, 99.5))),
        "volume_size": tuple(ckpt.get("volume_size", (160, 240, 240))),
        "representation": ckpt.get("representation", "latent"),
        "voxel_shape": tuple(ckpt["voxel_shape"]) if ckpt.get("voxel_shape") else None,
        "condition_on_treatment": bool(ckpt.get("condition_on_treatment", True)),
        "temporal_aggregator_variant": ckpt.get("temporal_aggregator_variant", "patch_temporal_spatial"),
    }


def load_patient(data_dir: str, patient_id: str, cfg: dict, geno_preproc=None) -> dict:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    def load(suffix: str):
        file_path = data_path / f"{patient_id}_{suffix}.npy"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")
        return np.load(str(file_path), mmap_mode="r")

    data = {"image": load("image"), "label": load("label"), "treatment": load("treatment"), "days": load("days")}
    data["patient_id"] = patient_id
    data["image_path"] = str(data_path / f"{patient_id}_image.npy")
    if data["image"].ndim != 5:
        raise ValueError(f"[InferViz] Patient '{patient_id}': expected image (S,C,D,H,W), got {data['image'].shape}")

    data["latent"] = None
    latent_dir = cfg["data"].get("latent_dir")
    if latent_dir and cfg["data"].get("representation", "latent") == "latent":
        latent_path = Path(latent_dir) / f"{patient_id}_latent.npy"
        if not latent_path.exists():
            raise FileNotFoundError(f"Missing precomputed latent file: {latent_path}")
        data["latent"] = np.load(str(latent_path), mmap_mode="r")
        print(f"[InferViz] Patient '{patient_id}': precomputed latent={data['latent'].shape}")

    geno_path = data_path / f"{patient_id}_geno.npy"
    real_geno = cfg["data"].get("real_geno", False)
    genomic_dim = cfg["data"].get("genomic_dim", None)
    if real_geno and geno_path.exists():
        raw_geno = np.asarray(np.load(str(geno_path), mmap_mode="r"), dtype=np.float32).reshape(-1)
        data["geno"] = geno_preproc.transform(raw_geno) if (geno_preproc is not None and geno_preproc.is_fit) else raw_geno
        data["geno_mask"] = True
    else:
        if genomic_dim is None:
            raise ValueError("Missing cfg['data']['genomic_dim']; needed for dummy genomics.")
        data["geno"] = np.zeros((genomic_dim,), dtype=np.float32)
        data["geno_mask"] = False

    print(f"[InferViz] Patient '{patient_id}': image={data['image'].shape}, label={data['label'].shape}")
    return data


def build_batch(patient_data: dict, t_idx: int, volume_size: tuple, device: str,
                 normalization_policy: str, normalization_clip_pct: tuple, norm_cache,
                 representation: str = "latent", image_stats=None, apply_n4: bool = False,
                 n4_cache_dir=None, n4_kwargs=None):
    images = patient_data["image"]
    labels = patient_data["label"]
    days_abs = np.asarray(patient_data["days"], dtype=np.float32)
    treat_abs = np.asarray(patient_data["treatment"], dtype=np.int64)
    geno = patient_data["geno"]
    geno_mask = bool(patient_data.get("geno_mask", False))
    patient_id = patient_data.get("patient_id")
    image_path = patient_data.get("image_path")

    def normalize_and_resize(session_idx):
        n4_cache_key = (
            preprocessing_utils.compute_n4_cache_key(patient_id, session_idx, FLAIR_INDEX, image_path, n4_kwargs)
            if apply_n4 and patient_id is not None and image_path is not None else None
        )
        vol = preprocessing_utils.apply_normalization_policy(
            normalization_policy,
            raw_volume_fn=lambda: np.asarray(images[session_idx, FLAIR_INDEX, :, :, :]),
            cache=norm_cache, cache_key=session_idx, clip_pct=normalization_clip_pct,
            stats=image_stats.get(FLAIR_INDEX) if image_stats else None,
            apply_n4=apply_n4, n4_cache_dir=n4_cache_dir, n4_cache_key=n4_cache_key,
            n4_kwargs=n4_kwargs, require_n4_cache=False,
        )
        return preprocessing_utils.to_model_shape_3d(vol, volume_size, representation=representation, is_label=False)

    target_vol = normalize_and_resize(t_idx)
    target_lbl = preprocessing_utils.to_model_shape_3d(
        np.asarray(labels[t_idx]), volume_size, representation=representation, is_label=True,
    )

    ctx_idxs = list(range(0, t_idx))
    n_ctx = len(ctx_idxs)
    target_day_abs = float(days_abs[t_idx])
    last_input_day_abs = float(days_abs[ctx_idxs[-1]]) if n_ctx > 0 else target_day_abs

    ctx_vols, ctx_lbls, ctx_days_rel, ctx_treats = [], [], [], []
    for ct in ctx_idxs:
        v = normalize_and_resize(ct)
        ctx_vols.append(torch.from_numpy(v).float().unsqueeze(0))
        ctx_lbls.append(preprocessing_utils.to_model_shape_3d(
            np.asarray(labels[ct]), volume_size, representation=representation, is_label=True,
        ))
        ctx_days_rel.append(float(days_abs[ct] - target_day_abs))
        ctx_treats.append(int(treat_abs[ct]))

    target_day_rel = np.float32(target_day_abs - last_input_day_abs)
    target_treat = int(treat_abs[t_idx])
    D, H, W = volume_size

    input_images = (
        torch.stack(ctx_vols, dim=0).unsqueeze(0).to(device)
        if n_ctx > 0 else torch.zeros((1, 0, 1, D, H, W), device=device)
    )
    input_labels = (
        torch.from_numpy(np.stack(ctx_lbls)).float().unsqueeze(0).to(device)
        if n_ctx > 0 else torch.zeros((1, 0, D, H, W), device=device)
    )
    batch = {
        "input_images": input_images,
        "input_labels": input_labels,
        "input_days": (torch.tensor([ctx_days_rel], dtype=torch.float32, device=device) if n_ctx > 0
                        else torch.zeros((1, 0), dtype=torch.float32, device=device)),
        "input_treatments": (torch.tensor([ctx_treats], dtype=torch.long, device=device) if n_ctx > 0
                              else torch.zeros((1, 0), dtype=torch.long, device=device)),
        "input_mask": torch.ones((1, n_ctx), dtype=torch.bool, device=device),
        "target_day": torch.tensor([target_day_rel], dtype=torch.float32, device=device),
        "target_treatment": torch.tensor([target_treat], dtype=torch.long, device=device),
        "geno": torch.tensor([geno], dtype=torch.float32, device=device),
        "geno_mask": torch.tensor([geno_mask], dtype=torch.bool, device=device),
    }

    last_vol = None
    last_lbl = None
    if n_ctx > 0:
        last_vol = ctx_vols[-1].squeeze(0).numpy()
        last_lbl = ctx_lbls[-1]

    latent_raw = patient_data.get("latent")
    target_latent_raw = None
    if latent_raw is not None:
        if n_ctx > 0:
            ctx_lat = [np.asarray(latent_raw[ct]) for ct in ctx_idxs]
            batch["input_latents"] = torch.from_numpy(np.stack(ctx_lat)).float().unsqueeze(0).to(device)
        else:
            Cz, dl, hl, wl = latent_raw.shape[1:]
            batch["input_latents"] = torch.zeros((1, 0, Cz, dl, hl, wl), device=device)
        target_latent_raw = torch.from_numpy(np.array(latent_raw[t_idx])).float().to(device)

    meta = {"num_previous_timepoints": n_ctx, "previous_treatments": ctx_treats, "target_treatment": target_treat}
    return batch, target_vol, target_lbl, last_vol, last_lbl, meta, target_latent_raw


@torch.no_grad()
def generate_volume(model: TaGeDiff, batch: dict, num_steps: int, normalization_policy: str = "whole_volume"):
    out = model.generate(
        input_images=batch["input_images"], input_days=batch["input_days"],
        input_treatments=batch["input_treatments"], input_mask=batch["input_mask"],
        genomics=batch["geno"], geno_mask=batch["geno_mask"],
        target_day=batch["target_day"], target_treatment=batch["target_treatment"],
        input_latents=batch.get("input_latents"),
        input_labels=batch.get("input_labels"),
        ddim_steps=num_steps, ddim_eta=0.0, fm_steps=num_steps,
    )
    mri_t = out["generated_mri"].squeeze(0).squeeze(0).float()
    if normalization_policy == "whole_volume":
        mri_t = mri_t.clamp(0.0, 1.0)
    mri = mri_t.cpu().numpy()
    return mri, out["z_0"]


def report(name, z):
    z = z.float()
    print(
        name,
        "mean=", z.mean().item(),
        "std=", z.std().item(),
        "abs_mean=", z.abs().mean().item(),
        "min=", z.min().item(),
        "max=", z.max().item(),
    )


def evaluate_and_visualize_patient(model, patient_id, patient_data, volume_size, steps,
                                    device, weights_used, normalization_policy,
                                    normalization_clip_pct, out_dir: Path, representation: str = "latent",
                                    image_stats=None, apply_n4: bool = False, n4_cache_dir=None, n4_kwargs=None,
                                    voxel_spacing_mm=None, err_threshold=DEFAULT_ERR_THRESHOLD,
                                    axial_slice_override: int = None):
    num_sessions = int(patient_data["label"].shape[0])
    norm_cache = preprocessing_utils.VolumeNormalizationCache(maxsize=32)
    voxel_spacing_mm = voxel_spacing_mm or resolve_voxel_spacing_mm(volume_size)
    dilation_mm = float(ROI_DILATION_ITERS * np.mean(voxel_spacing_mm))
    rows = []
    change_rows = []

    for t_idx in tqdm(range(1, num_sessions), desc=f"Patient {patient_id}", leave=False):
        batch, gt_vol, gt_lbl, last_vol, last_lbl, meta, target_latent_raw = build_batch(
            patient_data, t_idx, volume_size, device, normalization_policy, normalization_clip_pct, norm_cache,
            representation=representation, image_stats=image_stats, apply_n4=apply_n4,
            n4_cache_dir=n4_cache_dir, n4_kwargs=n4_kwargs,
        )
        pred_vol, z_0 = generate_volume(model, batch, num_steps=steps, normalization_policy=normalization_policy)

        if target_latent_raw is not None:
            latent_scale = model.latent_scale
            latent_mean = model.latent_mean
            generated_scaled = z_0.squeeze(0)
            print(f"[InferViz] Patient '{patient_id}' t={t_idx} latent debug:")
            report("real scaled", (target_latent_raw - latent_mean) * latent_scale)
            report("generated scaled", generated_scaled)
            report("real decoder input", target_latent_raw)
            report("generated decoder input", generated_scaled / latent_scale + latent_mean)

        common = tuple(min(a, b) for a, b in zip(gt_vol.shape, pred_vol.shape))
        gt_c = gt_vol[:common[0], :common[1], :common[2]]
        pred_c = pred_vol[:common[0], :common[1], :common[2]]
        lbl_c = gt_lbl[:common[0], :common[1], :common[2]]
        last_c = (last_vol[:common[0], :common[1], :common[2]] if last_vol is not None else gt_c)
        last_lbl_c = (last_lbl[:common[0], :common[1], :common[2]] if last_lbl is not None else None)

        data_range = resolve_data_range(normalization_policy, image_stats)
        metrics = {
            "mse": compute_mse(gt_c, pred_c),
            "ssim": compute_ssim(gt_c, pred_c, device=device, spatial_dims=3, data_range=data_range),
            "msssim": compute_msssim(gt_c, pred_c, device=device, spatial_dims=3, data_range=data_range),
            "psnr": compute_psnr(gt_c, pred_c, data_range=data_range),
        }

        tumor_mask = lbl_c > 0
        tumor_metrics = compute_tumor_region_metrics(gt_c, pred_c, tumor_mask, device, data_range)

        delta_gt = gt_c - last_c
        delta_pred = pred_c - last_c
        err = pred_c - gt_c

        baseline_mask = (last_lbl_c > 0) if last_lbl_c is not None else None
        change_region, mask_used = resolve_change_region_mask(baseline_mask, tumor_mask)
        dilated_region = (
            binary_dilation(change_region, iterations=ROI_DILATION_ITERS) if change_region.any() else change_region
        )
        mr_tumor = compute_change_magnitude_ratio(delta_gt, delta_pred, change_region)
        mr_dilated = compute_change_magnitude_ratio(delta_gt, delta_pred, dilated_region)
        change_magnitude = {"tumor": mr_tumor, "dilated": mr_dilated, "dilation_mm": dilation_mm}

        for region_name, region_result, region_dilation_mm in (
            ("tumor", mr_tumor, 0.0), ("dilated", mr_dilated, dilation_mm),
        ):
            change_rows.append({
                "patient_id": patient_id, "target_index": t_idx, "region": region_name,
                "mr": region_result["mr"], "std_pred": region_result["std_pred"], "std_gt": region_result["std_gt"],
                "ccc": region_result["ccc"], "n_voxels": region_result["n_voxels"],
                "mask_used": mask_used, "dilation_mm": region_dilation_mm, "nan_reason": region_result["nan_reason"],
            })

        if tumor_mask.any():
            maps, stats = make_tumor_change_maps(
                last_c, gt_c, pred_c, tumor_mask,
                eps=0.03 * data_range, wrong_dir_thresh=0.1 * data_range,
            )
            D = gt_c.shape[0]
            if axial_slice_override is not None:
                axial_idx = max(0, min(int(axial_slice_override), D - 1))
                if axial_idx != axial_slice_override:
                    print(f"[InferViz] Patient '{patient_id}' t={t_idx}: --axial-slice={axial_slice_override} "
                          f"out of range [0, {D-1}], clamped to {axial_idx}.")
            else:
                axial_idx = resolve_axial_idx(maps["roi"], D)
            last_treatment = meta["previous_treatments"][-1] if meta["previous_treatments"] else meta["target_treatment"]
            save_clinical_report(
                last_c, gt_c, pred_c, maps, stats,
                save_path=str(out_dir / f"{patient_id}_t{t_idx}_report.png"),
                patient_id=patient_id, t_idx=t_idx, steps=steps,
                last_treatment=last_treatment, target_treatment=meta["target_treatment"], metrics=metrics,
                tumor_metrics=tumor_metrics, axial_idx=axial_idx, change_magnitude=change_magnitude,
            )
            save_change_maps_figure(
                delta_gt, delta_pred, err, maps["roi"], axial_idx,
                save_path=str(out_dir / f"{patient_id}_t{t_idx}_changemaps.png"),
                patient_id=patient_id, t_idx=t_idx, steps=steps,
                err_threshold=err_threshold * data_range,
            )
            save_error_map_figure(
                err, maps["roi"], axial_idx,
                save_path=str(out_dir / f"{patient_id}_t{t_idx}_errormap.png"),
                patient_id=patient_id, t_idx=t_idx, steps=steps,
                err_threshold=err_threshold * data_range,
            )
        else:
            stats = {"roi_voxels": 0, "growth_recall": float("nan"), "shrink_recall": float("nan"),
                      "change_recall": float("nan"), "wrong_direction_rate": float("nan"), "tumor_delta_mae": float("nan")}

        rows.append({
            "patient_id": patient_id, "timepoint": t_idx,
            "num_previous_timepoints": meta["num_previous_timepoints"],
            "target_treatment": meta["target_treatment"],
            "weights_used": weights_used,
            **metrics, **stats,
            "mse_tumor": tumor_metrics["mse"],
            "ssim_tumor": tumor_metrics["ssim"],
            "ms_ssim_tumor": tumor_metrics["ms_ssim"],
            "psnr_tumor": tumor_metrics["psnr"],
        })

    return rows, change_rows


def main():
    parser = argparse.ArgumentParser(description="Tumor-change visualization + metrics over a patient split (whole-volume 3D).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patient-id", default=None, help="Evaluate a single patient")
    parser.add_argument("--patients-split", default=None, help="CSV of patient IDs (alternative to --patient-id)")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--split", default=None)
    parser.add_argument("--patient-id-col", default="patient_id")
    parser.add_argument("--save-dir", default="./evaluations/tumor_change_report")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--ema-policy", default="auto", choices=["auto", "force_ema", "force_raw"])
    parser.add_argument("--use-ema", action="store_true", help="DEPRECATED alias for --ema-policy=force_ema.")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--voxel-spacing-mm-override", type=float, nargs=3, default=None,
                         metavar=("DZ", "DY", "DX"),
                         help="Override the approximate (dz,dy,dx) mm/voxel spacing used to report the "
                              "MR_dilated region's dilation radius in mm (see resolve_voxel_spacing_mm's "
                              "docstring for the default's native-1mm-isotropic assumption).")
    parser.add_argument("--err-threshold", type=float, default=DEFAULT_ERR_THRESHOLD,
                         help="|error| below this (in normalization_policy=\"whole_volume\" units — scaled "
                              "by data_range for other policies, same convention as make_tumor_change_maps's "
                              "eps/wrong_dir_thresh) is masked out (rendered black) in the error-map figures, "
                              "so ordinary reconstruction noise doesn't compete visually with real misses.")
    parser.add_argument("--axial-slice", type=int, default=None,
                         help="Render this specific axial (depth) index in the PNG reports instead of the "
                              "default ROI-centered slice picked by resolve_axial_idx. Out-of-range values "
                              "are clamped to [0, D-1] for each timepoint's volume depth D. Statistics/metrics "
                              "are unaffected either way — they are always computed over the full 3D volume.")
    args = parser.parse_args()

    ema_policy = "force_ema" if args.use_ema else args.ema_policy
    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    raw_ckpt = read_checkpoint(args.checkpoint)
    cfg = resolve_training_mode(cfg, raw_ckpt, args.checkpoint)
    cfg = resolve_representation(cfg, raw_ckpt, args.checkpoint)
    cfg = resolve_condition_on_treatment(cfg, raw_ckpt, args.checkpoint)
    cfg = resolve_temporal_aggregator_variant(cfg, raw_ckpt, args.checkpoint)
    resolved_representation = cfg["data"].get("representation", "latent")

    if args.patient_id:
        patient_ids = [args.patient_id]
    elif args.patients_split:
        patient_ids = read_patient_ids_from_split(args.patients_split, args.patient_id_col, args.split_col, args.split)
    else:
        raise ValueError("Pass either --patient-id or --patients-split")

    model = TaGeDiff(cfg).to(args.device)
    model.eval()
    ckpt_info = load_checkpoint(model, raw_ckpt, ema_policy=ema_policy, device=args.device)
    if resolved_representation == "voxel":
        volume_size = tuple(cfg["data"].get("voxel_shape", ckpt_info["voxel_shape"] or (64, 96, 96)))
    else:
        volume_size = tuple(cfg["data"].get("volume_size", ckpt_info["volume_size"]))
    print(f"[InferViz] Representation: {resolved_representation}  |  model volume shape: {volume_size}")
    voxel_spacing_mm = resolve_voxel_spacing_mm(volume_size, spacing_override=args.voxel_spacing_mm_override)
    print(f"[InferViz] Approx. voxel spacing (dz,dy,dx) mm: {voxel_spacing_mm} "
          f"({'override' if args.voxel_spacing_mm_override else 'derived from native ~1mm-isotropic assumption'})")

    apply_n4 = bool(cfg["data"].get("apply_n4_bias_correction", False))
    n4_cache_dir = cfg["data"].get("n4_cache_dir")
    n4_kwargs = n4_kwargs_from_dc(cfg["data"])

    all_rows = []
    all_change_rows = []
    for patient_id in tqdm(patient_ids, desc="Patients"):
        patient_data = load_patient(cfg["data"]["data_dir"], patient_id, cfg, geno_preproc=ckpt_info["geno_preproc"])
        rows, change_rows = evaluate_and_visualize_patient(
            model, patient_id, patient_data, volume_size, args.steps, args.device,
            ckpt_info["weights_used"], ckpt_info["normalization_policy"], ckpt_info["normalization_clip_pct"], out_dir,
            representation=resolved_representation, image_stats=ckpt_info["image_stats"],
            apply_n4=apply_n4, n4_cache_dir=n4_cache_dir, n4_kwargs=n4_kwargs,
            voxel_spacing_mm=voxel_spacing_mm, err_threshold=args.err_threshold,
            axial_slice_override=args.axial_slice,
        )
        all_rows.extend(rows)
        all_change_rows.extend(change_rows)

    csv_path = out_dir / "tumor_change_metrics.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    change_csv_path = out_dir / "change_magnitude_metrics.csv"
    if all_change_rows:
        with open(change_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_change_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_change_rows)

    print(f"\n[InferViz] Done. {len(all_rows)} timepoint(s) evaluated. Metrics: {csv_path}")
    print(f"[InferViz] Change-magnitude metrics ({len(all_change_rows)} region-rows): {change_csv_path}")


if __name__ == "__main__":
    main()
