import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from scipy.ndimage import binary_dilation, binary_erosion

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.TaGeDiff import TaGeDiff
from src.data import preprocessing as preprocessing_utils
from src.data.dataset import n4_kwargs_from_dc
from src.evaluation.metrics import (
    compute_mse,
    compute_psnr,
    compute_ssim,
    compute_ms_ssim,
    resolve_data_range,
    ssim_definition_string,
)
from src.evaluation.change_metrics import (
    compute_region_change_metrics,
    estimate_case_noise,
    build_control_mask,
)
from src.evaluation.registration import rigid_register_to_fixed

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

FLAIR_INDEX = 2

ROI_DILATION_ITERS = 3

NATIVE_SHAPE_MM_ISOTROPIC = (155, 240, 240)

CONTROL_DILATION_MM = 20.0


def resolve_voxel_spacing_mm(volume_size, spacing_override=None) -> tuple:
    if spacing_override is not None:
        return tuple(float(x) for x in spacing_override)
    return tuple(float(n) / float(w) for n, w in zip(NATIVE_SHAPE_MM_ISOTROPIC, volume_size))


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
    }, stats


def tumor_bbox_slices(mask: np.ndarray, margin: int = 8) -> Optional[Tuple[slice, ...]]:
    if not mask.any():
        return None
    idxs = np.where(mask)
    return tuple(
        slice(max(0, int(ax.min()) - margin), min(mask.shape[d] - 1, int(ax.max()) + margin) + 1)
        for d, ax in enumerate(idxs)
    )


def compute_tumor_region_metrics(gt_dhw: np.ndarray, pred_dhw: np.ndarray, tumor_mask_dhw: np.ndarray,
                                  device: str, data_range: float) -> Dict[str, float]:
    bbox = tumor_bbox_slices(tumor_mask_dhw)
    if bbox is None:
        nan = float("nan")
        return {"mse": nan, "ssim": nan, "ms_ssim": nan, "psnr": nan}
    gt_box = gt_dhw[bbox]
    pred_box = pred_dhw[bbox]
    return {
        "mse": compute_mse(gt_box, pred_box),
        "ssim": compute_ssim(gt_box, pred_box, device=device, spatial_dims=3, data_range=data_range),
        "ms_ssim": compute_ms_ssim(gt_box, pred_box, device=device, spatial_dims=3, data_range=data_range),
        "psnr": compute_psnr(gt_box, pred_box, data_range=data_range),
    }


def resolve_change_region_mask(baseline_mask: Optional[np.ndarray] = None, followup_mask: Optional[np.ndarray] = None):
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
        nan_reason = f"region has {n_voxels} voxel(s) < MR_MIN_REGION_VOXELS={MR_MIN_REGION_VOXELS}"
        return {"mr": float("nan"), "std_pred": float("nan"), "std_gt": float("nan"),
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


CHANGE_METRICS_RAW_SUM_FIELDS = (
    "sum_sq_resid", "sum_sq_gt", "sum_sq_pred", "sum_dot_gt_pred",
    "sum_abs_resid", "sum_abs_avg_denom", "sum_abs_gt",
)


def build_change_metrics_rows(
    patient_id: str,
    t_idx: int,
    method: str,
    delta_gt: np.ndarray,
    delta_pred: np.ndarray,
    regions: Dict[str, np.ndarray],
    region_dilation_mm: Dict[str, float],
    region_mask_used: Dict[str, str],
    sigma_noise: float,
    drift: float,
    whole_volume_metrics: Dict[str, float],
) -> List[Dict]:
    rows = []
    for region_name, region_mask in regions.items():
        res = compute_region_change_metrics(
            delta_gt, delta_pred, region_mask, sigma_noise=sigma_noise, drift=drift,
        )
        row = {
            "patient_id": patient_id,
            "target_index": int(t_idx),
            "sample_index": 0,
            "method": method,
            "region": region_name,
            "delta_ss": res["delta_ss"],
            "r_pattern": res["r_pattern"],
            "m_magnitude": res["m_magnitude"],
            "delta_rmae": res["delta_rmae"],
            "delta_ss_l1": res["delta_ss_l1"],
            "norm_d_gt": res["norm_d_gt"],
            "norm_d_pred": res["norm_d_pred"],
            "n_voxels": res["n_voxels"],
            "sigma_noise": res["sigma_noise"],
            "drift": res["drift"],
            "delta_ss_max": res["delta_ss_max"],
            "mask_used": region_mask_used.get(region_name, ""),
            "dilation_mm": region_dilation_mm.get(region_name, float("nan")),
            "nan_reason": res["nan_reason"],
            "mse": whole_volume_metrics.get("mse", float("nan")),
            "ssim": whole_volume_metrics.get("ssim", float("nan")),
            "psnr": whole_volume_metrics.get("psnr", float("nan")),
        }
        for f in CHANGE_METRICS_RAW_SUM_FIELDS:
            row[f] = res[f]
        rows.append(row)
    return rows


CHANGE_METRICS_FIELDS = [
    "patient_id", "target_index", "sample_index", "method", "region",
    "delta_ss", "r_pattern", "m_magnitude", "delta_rmae", "delta_ss_l1",
    "norm_d_gt", "norm_d_pred", "n_voxels", "sigma_noise", "drift", "delta_ss_max",
    "mask_used", "dilation_mm", "nan_reason", "mse", "ssim", "psnr",
] + list(CHANGE_METRICS_RAW_SUM_FIELDS)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: List[str]) -> dict:
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
            print("[EvalSplit] --ema-policy=auto: no EMA state in checkpoint, evaluating RAW weights.")
        return has_ema
    raise ValueError(f"Unknown ema_policy={ema_policy!r}, expected auto/force_ema/force_raw")


def read_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu")


def resolve_training_mode(cfg: dict, ckpt: dict, checkpoint_path: str) -> dict:
    ckpt_mode = ckpt.get("training_mode")
    cfg.setdefault("training", {})
    cfg_mode = cfg["training"].get("training_mode", "diffusion")

    if ckpt_mode is None:
        print(
            f"[EvalSplit] WARNING: checkpoint '{checkpoint_path}' has no saved training_mode "
            f"(pre-schema-v2). Trusting --config's training.training_mode={cfg_mode!r}."
        )
        return cfg

    if ckpt_mode != cfg_mode:
        print(
            f"[EvalSplit] CORRECTING training.training_mode: --config says {cfg_mode!r} but "
            f"checkpoint '{checkpoint_path}' was trained with {ckpt_mode!r}. Using the "
            "checkpoint's value."
        )
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
            f"[EvalSplit] WARNING: checkpoint '{checkpoint_path}' has no saved representation "
            f"(pre-voxel-mode). Trusting --config's data.representation={cfg_repr!r}."
        )
        return cfg

    if ckpt_repr != cfg_repr:
        print(
            f"[EvalSplit] CORRECTING data.representation: --config says {cfg_repr!r} but "
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
            f"[EvalSplit] WARNING: checkpoint '{checkpoint_path}' has no saved "
            f"condition_on_treatment (pre-this-feature). Trusting --config's "
            f"context.condition_on_treatment={cfg_cot!r}."
        )
        return cfg

    if ckpt_cot != cfg_cot:
        print(
            f"[EvalSplit] CORRECTING context.condition_on_treatment: --config says {cfg_cot!r} "
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
            f"[EvalSplit] WARNING: checkpoint '{checkpoint_path}' has no saved "
            f"temporal_aggregator_variant (pre-this-feature) — every such checkpoint used "
            f"'patch_temporal_spatial'. Trusting that, not --config's temporal_aggregator."
            f"variant={cfg_variant!r}."
        )

    if ckpt_variant != cfg_variant:
        print(
            f"[EvalSplit] CORRECTING temporal_aggregator.variant: --config says {cfg_variant!r} "
            f"but checkpoint '{checkpoint_path}' was trained with {ckpt_variant!r}. Using the "
            "checkpoint's value."
        )
    cfg["temporal_aggregator"]["variant"] = ckpt_variant

    return cfg


def load_checkpoint(model: torch.nn.Module, ckpt: dict, ema_policy: str, device: str):
    schema_version = ckpt.get("schema_version", 1)
    if schema_version < 3:
        raise ValueError(
            f"Checkpoint predates schema v3 (found v{schema_version}) — it was trained under "
            "the OLD 2D pipeline (2D MONAI VAE + Conv2d UNet), architecturally incompatible "
            "with this 3D pipeline (medvae_4_1_3d + ConditionalUNet3D). Cannot evaluate a "
            "pre-3D-migration checkpoint here — re-train under the 3D pipeline first."
        )

    use_ema = resolve_ema_policy(ema_policy, ckpt)
    weights = ckpt["ema"] if use_ema else ckpt["model"]

    current = model.state_dict()
    loaded = 0
    skipped = 0

    for k, v in weights.items():
        if k in current and current[k].shape == v.shape:
            current[k].copy_(v)
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(current)
    model.to(device)

    tag = "EMA" if use_ema else "raw"
    print(
        "=" * 60 + f"\n[EvalSplit] WEIGHTS USED: {tag.upper()} "
        f"(ema_policy={ema_policy}) — epoch={ckpt.get('epoch', '?')} "
        f"loaded={loaded} skipped={skipped}\n" + "=" * 60
    )

    return {
        "epoch": ckpt.get("epoch", None),
        "weights_used": tag,
        "schema_version": schema_version,
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


def read_patient_ids_from_split(
    csv_path: str,
    patient_id_col: str = "patient_id",
    split_col: Optional[str] = "split",
    split: Optional[str] = None,
) -> List[str]:
    csv_path = str(csv_path)
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {csv_path}")

        fieldnames = reader.fieldnames
        pid_col = patient_id_col if patient_id_col in fieldnames else fieldnames[0]
        has_split = split_col is not None and split_col in fieldnames

        patient_ids = []
        for row in reader:
            if split is not None and has_split:
                if str(row[split_col]).strip() != str(split):
                    continue
            pid = str(row[pid_col]).strip()
            if pid:
                patient_ids.append(pid)

    seen = set()
    unique_ids = []
    for pid in patient_ids:
        if pid not in seen:
            seen.add(pid)
            unique_ids.append(pid)

    if not unique_ids:
        split_msg = f" for split={split}" if split is not None else ""
        raise ValueError(f"No patient IDs found in {csv_path}{split_msg}")

    return unique_ids


def load_patient(data_dir: str, patient_id: str, cfg: dict, geno_preproc=None) -> dict:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    def load(suffix: str):
        file_path = data_path / f"{patient_id}_{suffix}.npy"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")
        return np.load(str(file_path), mmap_mode="r")

    data = {
        "image": load("image"),
        "image_path": str(data_path / f"{patient_id}_image.npy"),
        "label": load("label"),
        "treatment": load("treatment"),
        "days": load("days"),
    }
    if data["image"].ndim != 5:
        raise ValueError(
            f"[EvalSplit] Patient '{patient_id}': expected image array (S,C,D,H,W), "
            f"got shape {data['image'].shape}"
        )

    geno_path = data_path / f"{patient_id}_geno.npy"
    real_geno = cfg["data"].get("real_geno", False)
    genomic_dim = cfg["data"].get("genomic_dim", None)

    if real_geno and geno_path.exists():
        raw_geno = np.asarray(np.load(str(geno_path), mmap_mode="r"), dtype=np.float32).reshape(-1)
        if geno_preproc is not None and geno_preproc.is_fit:
            data["geno"] = geno_preproc.transform(raw_geno)
        else:
            data["geno"] = raw_geno
        data["geno_mask"] = True
    else:
        if genomic_dim is None:
            raise ValueError("Missing cfg['data']['genomic_dim']; needed for dummy genomics.")
        data["geno"] = np.zeros((genomic_dim,), dtype=np.float32)
        data["geno_mask"] = False

    print(
        f"[EvalSplit] Patient '{patient_id}': "
        f"image={data['image'].shape}, label={data['label'].shape}"
    )
    return data


def _normalize_and_resize_session(images_raw, session_idx, model_shape, normalization_policy,
                                   normalization_clip_pct, norm_cache, representation="latent",
                                   image_stats=None, apply_n4=False, n4_cache_dir=None, n4_kwargs=None,
                                   patient_id=None, image_path=None):
    n4_cache_key = (
        preprocessing_utils.compute_n4_cache_key(patient_id, session_idx, FLAIR_INDEX, image_path, n4_kwargs)
        if apply_n4 and patient_id is not None and image_path is not None else None
    )
    vol = preprocessing_utils.apply_normalization_policy(
        normalization_policy,
        raw_volume_fn=lambda: np.asarray(images_raw[session_idx, FLAIR_INDEX, :, :, :]),
        cache=norm_cache,
        cache_key=session_idx,
        clip_pct=normalization_clip_pct,
        stats=image_stats.get(FLAIR_INDEX) if image_stats else None,
        apply_n4=apply_n4,
        n4_cache_dir=n4_cache_dir,
        n4_cache_key=n4_cache_key,
        n4_kwargs=n4_kwargs,
        require_n4_cache=True,
    )
    return preprocessing_utils.to_model_shape_3d(vol, model_shape, representation=representation, is_label=False)


def build_batch(
    patient_data: dict,
    t_idx: int,
    volume_size: Tuple[int, int, int],
    device: str,
    normalization_policy: str = "whole_volume",
    normalization_clip_pct: Tuple[float, float] = (0.5, 99.5),
    norm_cache: Optional[preprocessing_utils.VolumeNormalizationCache] = None,
    representation: str = "latent",
    image_stats: Optional[dict] = None,
    apply_n4: bool = False,
    n4_cache_dir: Optional[str] = None,
    n4_kwargs: Optional[dict] = None,
    patient_id: Optional[str] = None,
):
    images_raw = patient_data["image"]
    image_path = patient_data.get("image_path")
    labels_raw = patient_data["label"]
    days_abs = np.asarray(patient_data["days"], dtype=np.float32)
    treat_abs = np.asarray(patient_data["treatment"], dtype=np.int64)
    geno = patient_data["geno"]
    geno_mask = bool(patient_data.get("geno_mask", False))

    tgt_vol = _normalize_and_resize_session(images_raw, t_idx, volume_size, normalization_policy,
                                             normalization_clip_pct, norm_cache, representation,
                                             image_stats=image_stats, apply_n4=apply_n4,
                                             n4_cache_dir=n4_cache_dir, n4_kwargs=n4_kwargs,
                                             patient_id=patient_id, image_path=image_path)
    tgt_lbl = preprocessing_utils.to_model_shape_3d(
        np.asarray(labels_raw[t_idx]), volume_size, representation=representation, is_label=True,
    )

    ctx_idxs = list(range(0, t_idx))
    n_ctx = len(ctx_idxs)

    ctx_vols = []
    ctx_lbls = []
    ctx_days_rel = []
    ctx_treats = []

    target_day_abs = float(days_abs[t_idx])
    last_input_day_abs = float(days_abs[ctx_idxs[-1]]) if n_ctx > 0 else target_day_abs

    for ci in ctx_idxs:
        v = _normalize_and_resize_session(images_raw, ci, volume_size, normalization_policy,
                                           normalization_clip_pct, norm_cache, representation,
                                           image_stats=image_stats, apply_n4=apply_n4,
                                           n4_cache_dir=n4_cache_dir, n4_kwargs=n4_kwargs,
                                           patient_id=patient_id, image_path=image_path)
        ctx_vols.append(torch.from_numpy(v).float().unsqueeze(0))
        l = preprocessing_utils.to_model_shape_3d(
            np.asarray(labels_raw[ci]), volume_size, representation=representation, is_label=True,
        )
        ctx_lbls.append(torch.from_numpy(l).float())
        ctx_days_rel.append(float(days_abs[ci] - target_day_abs))
        ctx_treats.append(int(treat_abs[ci]))

    target_day_rel = np.float32(target_day_abs - last_input_day_abs)
    target_treat = int(treat_abs[t_idx])

    D, H, W = volume_size
    input_images = (
        torch.stack(ctx_vols, dim=0).unsqueeze(0).to(device)
        if n_ctx > 0
        else torch.zeros((1, 0, 1, D, H, W), device=device)
    )

    input_labels = (
        torch.stack(ctx_lbls, dim=0).unsqueeze(0).to(device)
        if n_ctx > 0
        else torch.zeros((1, 0, D, H, W), device=device)
    )

    batch = {
        "input_images": input_images,
        "input_labels": input_labels,
        "input_days": (
            torch.tensor([ctx_days_rel], dtype=torch.float32, device=device)
            if n_ctx > 0 else torch.zeros((1, 0), dtype=torch.float32, device=device)
        ),
        "input_treatments": (
            torch.tensor([ctx_treats], dtype=torch.long, device=device)
            if n_ctx > 0 else torch.zeros((1, 0), dtype=torch.long, device=device)
        ),
        "input_mask": torch.ones((1, n_ctx), dtype=torch.bool, device=device),
        "target_day": torch.tensor([target_day_rel], dtype=torch.float32, device=device),
        "target_treatment": torch.tensor([target_treat], dtype=torch.long, device=device),
        "geno": torch.tensor([geno], dtype=torch.float32, device=device),
        "geno_mask": torch.tensor([geno_mask], dtype=torch.bool, device=device),
    }

    gt_vol_np = tgt_vol[None, ...].astype(np.float32)
    lbl_vol_np = tgt_lbl[None, ...]
    last_vol_np = ctx_vols[-1].numpy()
    last_lbl_np = ctx_lbls[-1].numpy()[None, ...]

    meta = {
        "timepoint": int(t_idx),
        "num_previous_timepoints": int(n_ctx),
        "previous_treatments": ctx_treats,
        "target_treatment": int(target_treat),
    }

    return batch, gt_vol_np, lbl_vol_np, last_vol_np, last_lbl_np, meta


@torch.no_grad()
def generate_volume(model: TaGeDiff, batch: dict, num_steps: int,
                     normalization_policy: str = "whole_volume") -> Tuple[np.ndarray, np.ndarray]:
    out = model.generate(
        input_images=batch["input_images"],
        input_days=batch["input_days"],
        input_treatments=batch["input_treatments"],
        input_mask=batch["input_mask"],
        genomics=batch["geno"],
        geno_mask=batch["geno_mask"],
        target_day=batch["target_day"],
        target_treatment=batch["target_treatment"],
        ddim_steps=num_steps,
        ddim_eta=0.0,
        fm_steps=num_steps,
        input_labels=batch["input_labels"],
    )
    gen_t = out["generated_mri"].squeeze(0).float()
    if normalization_policy == "whole_volume":
        gen_t = gen_t.clamp(0.0, 1.0)
    gen = gen_t.cpu().numpy()
    z_0 = out["z_0"].squeeze(0).float().cpu().numpy()
    return gen, z_0


def save_center_slice_report(gt_vol: np.ndarray, pred_vol: np.ndarray, save_path: Path, title: str = ""):
    if not HAS_MPL:
        return
    gt = gt_vol[0] if gt_vol.ndim == 4 else gt_vol
    pred = pred_vol[0] if pred_vol.ndim == 4 else pred_vol
    D, H, W = gt.shape
    zd, zh, zw = D // 2, H // 2, W // 2

    views_gt = [gt[zd], gt[:, zh, :], gt[:, :, zw]]
    views_pred = [pred[zd], pred[:, zh, :], pred[:, :, zw]]
    names = ["Axial", "Coronal", "Sagittal"]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for c in range(3):
        axes[0, c].imshow(views_gt[c], cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title(f"GT {names[c]}")
        axes[0, c].axis("off")
        axes[1, c].imshow(views_pred[c], cmap="gray", vmin=0, vmax=1)
        axes[1, c].set_title(f"Pred {names[c]}")
        axes[1, c].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if np.isfinite(arr).sum() == 0:
        return float("nan")
    return float(np.nanmean(arr))


def nanstd(values: List[float], ddof: int = 1) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= ddof:
        return float("nan")
    return float(np.std(arr, ddof=ddof))


def evaluate_patient(
    model: TaGeDiff,
    patient_id: str,
    patient_data: dict,
    volume_size: Tuple[int, int, int],
    steps: int,
    device: str,
    use_amp: bool,
    weights_used: str,
    normalization_policy: str,
    normalization_clip_pct: Tuple[float, float],
    out_dir: Path,
    save_volumes: bool,
    save_nifti: bool,
    save_viz: bool,
    representation: str = "latent",
    restore_native_resolution: bool = False,
    image_stats: Optional[dict] = None,
    apply_n4: bool = False,
    n4_cache_dir: Optional[str] = None,
    n4_kwargs: Optional[dict] = None,
    voxel_spacing_mm: Optional[tuple] = None,
    register_baseline: bool = True,
) -> Tuple[List[Dict], Dict, List[Dict]]:
    num_sessions = int(patient_data["label"].shape[0])
    if num_sessions < 2:
        print(f"[EvalSplit] Patient {patient_id}: fewer than 2 sessions, nothing to evaluate.")
        nan = float("nan")
        return [], {
            "patient_id": patient_id, "num_timepoints": 0, "weights_used": weights_used,
            "mean_mse": nan, "std_mse": nan, "mean_ms_ssim": nan, "std_ms_ssim": nan,
            "mean_ssim": nan, "std_ssim": nan, "mean_psnr": nan, "std_psnr": nan,
            "mean_latent_mse": nan, "std_latent_mse": nan,
            "mean_mse_tumor": nan, "std_mse_tumor": nan,
            "mean_ssim_tumor": nan, "std_ssim_tumor": nan,
            "mean_ms_ssim_tumor": nan, "std_ms_ssim_tumor": nan,
            "mean_psnr_tumor": nan, "std_psnr_tumor": nan,
            "mean_growth_recall": nan, "std_growth_recall": nan,
            "mean_shrink_recall": nan, "std_shrink_recall": nan,
            "mean_change_recall": nan, "std_change_recall": nan,
            "mean_mr_tumor": nan, "std_mr_tumor": nan,
            "mean_mr_dilated": nan, "std_mr_dilated": nan,
            "mean_ccc_tumor": nan, "std_ccc_tumor": nan,
            "mean_ccc_dilated": nan, "std_ccc_dilated": nan,
        }, []

    voxel_spacing_mm = voxel_spacing_mm or resolve_voxel_spacing_mm(volume_size)
    dilation_mm = float(ROI_DILATION_ITERS * np.mean(voxel_spacing_mm))
    control_dilation_iters = max(1, int(round(CONTROL_DILATION_MM / max(float(np.mean(voxel_spacing_mm)), 1e-6))))
    control_dilation_mm = float(control_dilation_iters * np.mean(voxel_spacing_mm))

    rows = []
    change_rows = []
    mse_values, ms_ssim_values, ssim_values, psnr_values, latent_mse_values = [], [], [], [], []
    mse_native_values, ssim_native_values, ms_ssim_native_values, psnr_native_values = [], [], [], []
    mse_tumor_values, ssim_tumor_values, ms_ssim_tumor_values, psnr_tumor_values = [], [], [], []
    growth_recall_values, shrink_recall_values, change_recall_values = [], [], []
    mr_tumor_values, mr_dilated_values, ccc_tumor_values, ccc_dilated_values = [], [], [], []

    native_shape = tuple(patient_data["image"].shape[-3:])
    image_path = patient_data.get("image_path")

    data_range = resolve_data_range(normalization_policy, image_stats)

    norm_cache = preprocessing_utils.VolumeNormalizationCache(maxsize=32)
    pbar = tqdm(range(1, num_sessions), desc=f"Patient {patient_id}", leave=False)
    for t_idx in pbar:
        batch, gt_vol_np, lbl_vol_np, last_vol_np, last_lbl_np, meta = build_batch(
            patient_data, t_idx, volume_size, device,
            normalization_policy=normalization_policy,
            normalization_clip_pct=normalization_clip_pct,
            norm_cache=norm_cache,
            representation=representation,
            image_stats=image_stats,
            apply_n4=apply_n4,
            n4_cache_dir=n4_cache_dir,
            n4_kwargs=n4_kwargs,
            patient_id=patient_id,
        )

        amp_enabled = bool(use_amp and str(device).startswith("cuda"))
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            pred_vol_np, z_0_np = generate_volume(model, batch, num_steps=steps,
                                                   normalization_policy=normalization_policy)

        common_shape = tuple(min(a, b) for a, b in zip(gt_vol_np.shape, pred_vol_np.shape))
        gt_c = gt_vol_np[..., :common_shape[-3], :common_shape[-2], :common_shape[-1]]
        pred_c = pred_vol_np[..., :common_shape[-3], :common_shape[-2], :common_shape[-1]]
        lbl_c = lbl_vol_np[..., :common_shape[-3], :common_shape[-2], :common_shape[-1]]
        last_c = last_vol_np[..., :common_shape[-3], :common_shape[-2], :common_shape[-1]]
        last_lbl_c = last_lbl_np[..., :common_shape[-3], :common_shape[-2], :common_shape[-1]]

        if register_baseline:
            last_c_reg, last_lbl_c_reg, reg_info = rigid_register_to_fixed(
                gt_c[0], last_c[0], last_lbl_c[0],
            )
            last_c = last_c_reg[None, ...]
            last_lbl_c = last_lbl_c_reg[None, ...]
            if reg_info["status"] != "ok":
                print(f"[EvalSplit] Patient {patient_id} t={t_idx}: baseline registration "
                      f"{reg_info['status']} ({reg_info.get('error', 'SimpleITK not installed')}) "
                      "— change metrics for this timepoint use the UNREGISTERED baseline.")

        mse = compute_mse(gt_c, pred_c)
        ssim = compute_ssim(gt_c, pred_c, device=device, spatial_dims=3, data_range=data_range)
        ms_ssim = compute_ms_ssim(gt_c, pred_c, device=device, spatial_dims=3, data_range=data_range)
        psnr = compute_psnr(gt_c, pred_c, data_range=data_range)

        with torch.no_grad():
            z_target = model.flair_encoder(torch.from_numpy(gt_vol_np).unsqueeze(0).to(device))
            z_target = ((z_target - model.latent_mean) * model.latent_scale).squeeze(0).float().cpu().numpy()
        common_latent = tuple(min(a, b) for a, b in zip(z_target.shape, z_0_np.shape))
        z_t_c = z_target[..., :common_latent[-3], :common_latent[-2], :common_latent[-1]]
        z_0_c = z_0_np[..., :common_latent[-3], :common_latent[-2], :common_latent[-1]]
        latent_mse = float(np.mean((z_t_c - z_0_c) ** 2))

        is_tumor_positive = bool((patient_data["label"][t_idx] > 0).any())

        tumor_mask = lbl_c > 0
        if tumor_mask.any():
            _, change_stats = make_tumor_change_maps(
                last_c, gt_c, pred_c, tumor_mask,
                eps=0.03 * data_range, wrong_dir_thresh=0.1 * data_range,
            )
        else:
            change_stats = {
                "roi_voxels": 0, "gt_growth_voxels": 0, "gt_shrink_voxels": 0,
                "growth_recall": float("nan"), "shrink_recall": float("nan"),
                "change_recall": float("nan"), "wrong_direction_rate": float("nan"),
                "tumor_delta_mae": float("nan"),
            }

        tumor_metrics = compute_tumor_region_metrics(gt_c[0], pred_c[0], tumor_mask[0], device, data_range)

        delta_gt = gt_c - last_c
        delta_pred = pred_c - last_c
        baseline_mask = last_lbl_c > 0
        change_region, mask_used = resolve_change_region_mask(baseline_mask, tumor_mask)
        dilated_region = (
            binary_dilation(change_region, iterations=ROI_DILATION_ITERS) if change_region.any() else change_region
        )
        mr_tumor = compute_change_magnitude_ratio(delta_gt, delta_pred, change_region)
        mr_dilated = compute_change_magnitude_ratio(delta_gt, delta_pred, dilated_region)

        brain_mask = (gt_c != 0) | (last_c != 0)
        control_mask = build_control_mask(brain_mask, change_region, control_dilation_iters)
        noise_diag = estimate_case_noise(delta_gt, control_mask)
        full_mask = np.ones_like(brain_mask, dtype=bool)

        change_regions = {
            "tumor": change_region, "dilated": dilated_region,
            "control": control_mask, "full": full_mask,
        }
        change_region_dilation_mm = {
            "tumor": 0.0, "dilated": dilation_mm,
            "control": control_dilation_mm, "full": float("nan"),
        }
        change_region_mask_used = {
            "tumor": mask_used, "dilated": mask_used,
            "control": "brain_minus_%gmm_dilation" % CONTROL_DILATION_MM, "full": "whole_volume",
        }

        change_rows.extend(build_change_metrics_rows(
            patient_id, t_idx, "model", delta_gt, delta_pred,
            change_regions, change_region_dilation_mm, change_region_mask_used,
            noise_diag["sigma_noise"], noise_diag["drift"],
            whole_volume_metrics={"mse": mse, "ssim": ssim, "psnr": psnr},
        ))

        copy_mse = compute_mse(gt_c, last_c)
        copy_ssim = compute_ssim(gt_c, last_c, device=device, spatial_dims=3, data_range=data_range)
        copy_psnr = compute_psnr(gt_c, last_c, data_range=data_range)
        change_rows.extend(build_change_metrics_rows(
            patient_id, t_idx, "copy_baseline", delta_gt, np.zeros_like(delta_gt),
            change_regions, change_region_dilation_mm, change_region_mask_used,
            noise_diag["sigma_noise"], noise_diag["drift"],
            whole_volume_metrics={"mse": copy_mse, "ssim": copy_ssim, "psnr": copy_psnr},
        ))
        change_rows.extend(build_change_metrics_rows(
            patient_id, t_idx, "oracle", delta_gt, delta_gt,
            change_regions, change_region_dilation_mm, change_region_mask_used,
            noise_diag["sigma_noise"], noise_diag["drift"],
            whole_volume_metrics={"mse": 0.0, "ssim": 1.0, "psnr": float("inf")},
        ))

        mse_values.append(mse); ms_ssim_values.append(ms_ssim)
        ssim_values.append(ssim); psnr_values.append(psnr)
        latent_mse_values.append(latent_mse)
        mse_tumor_values.append(tumor_metrics["mse"]); ssim_tumor_values.append(tumor_metrics["ssim"])
        ms_ssim_tumor_values.append(tumor_metrics["ms_ssim"]); psnr_tumor_values.append(tumor_metrics["psnr"])
        growth_recall_values.append(change_stats["growth_recall"])
        shrink_recall_values.append(change_stats["shrink_recall"])
        change_recall_values.append(change_stats["change_recall"])
        mr_tumor_values.append(mr_tumor["mr"]); mr_dilated_values.append(mr_dilated["mr"])
        ccc_tumor_values.append(mr_tumor["ccc"]); ccc_dilated_values.append(mr_dilated["ccc"])

        row = {
            "patient_id": patient_id,
            "timepoint": int(t_idx),
            "num_previous_timepoints": meta["num_previous_timepoints"],
            "previous_treatments": "|".join(map(str, meta["previous_treatments"])),
            "target_treatment": int(meta["target_treatment"]),
            "is_tumor_positive_target": is_tumor_positive,
            "mse": float(mse),
            "ms_ssim": float(ms_ssim),
            "ssim": float(ssim),
            "psnr": float(psnr),
            "latent_mse": latent_mse,
            "mse_tumor": tumor_metrics["mse"],
            "ssim_tumor": tumor_metrics["ssim"],
            "ms_ssim_tumor": tumor_metrics["ms_ssim"],
            "psnr_tumor": tumor_metrics["psnr"],
            "weights_used": weights_used,
            **change_stats,
            "mr_tumor": mr_tumor["mr"], "std_pred_tumor": mr_tumor["std_pred"], "std_gt_tumor": mr_tumor["std_gt"],
            "ccc_tumor": mr_tumor["ccc"], "n_voxels_tumor": mr_tumor["n_voxels"], "nan_reason_tumor": mr_tumor["nan_reason"],
            "mr_dilated": mr_dilated["mr"], "std_pred_dilated": mr_dilated["std_pred"], "std_gt_dilated": mr_dilated["std_gt"],
            "ccc_dilated": mr_dilated["ccc"], "n_voxels_dilated": mr_dilated["n_voxels"], "nan_reason_dilated": mr_dilated["nan_reason"],
            "mask_used": mask_used, "dilation_mm": dilation_mm,
        }

        if restore_native_resolution:
            images_raw = patient_data["image"]
            native_n4_cache_key = (
                preprocessing_utils.compute_n4_cache_key(patient_id, t_idx, FLAIR_INDEX, image_path, n4_kwargs)
                if apply_n4 and image_path is not None else None
            )
            native_gt = preprocessing_utils.apply_normalization_policy(
                normalization_policy,
                raw_volume_fn=lambda: np.asarray(images_raw[t_idx, FLAIR_INDEX, :, :, :]),
                cache=norm_cache, cache_key=t_idx, clip_pct=normalization_clip_pct,
                stats=image_stats.get(FLAIR_INDEX) if image_stats else None,
                apply_n4=apply_n4, n4_cache_dir=n4_cache_dir,
                n4_cache_key=native_n4_cache_key, n4_kwargs=n4_kwargs, require_n4_cache=True,
            )[None, ...].astype(np.float32)
            native_pred = preprocessing_utils.resize_volume_3d(pred_vol_np, native_shape, mode="trilinear")

            mse_n = compute_mse(native_gt, native_pred)
            ssim_n = compute_ssim(native_gt, native_pred, device=device, spatial_dims=3, data_range=data_range)
            ms_ssim_n = compute_ms_ssim(native_gt, native_pred, device=device, spatial_dims=3, data_range=data_range)
            psnr_n = compute_psnr(native_gt, native_pred, data_range=data_range)

            mse_native_values.append(mse_n); ssim_native_values.append(ssim_n)
            ms_ssim_native_values.append(ms_ssim_n); psnr_native_values.append(psnr_n)
            row.update({
                "mse_native": float(mse_n), "ssim_native": float(ssim_n),
                "ms_ssim_native": float(ms_ssim_n), "psnr_native": float(psnr_n),
                "native_shape": str(native_shape),
            })

        rows.append(row)
        pbar.set_postfix({"mse": f"{mse:.5f}", "ssim": f"{ssim:.4f}"})

        if save_volumes:
            vol_dir = out_dir / "volumes"
            vol_dir.mkdir(parents=True, exist_ok=True)
            np.save(vol_dir / f"{patient_id}_t{t_idx}_gt.npy", gt_vol_np)
            np.save(vol_dir / f"{patient_id}_t{t_idx}_pred.npy", pred_vol_np)
            if save_nifti:
                if not HAS_NIBABEL:
                    print("[EvalSplit] --save-nifti requested but nibabel is not installed; skipping.")
                else:
                    affine = np.eye(4)
                    nib.save(nib.Nifti1Image(gt_vol_np[0], affine), str(vol_dir / f"{patient_id}_t{t_idx}_gt.nii.gz"))
                    nib.save(nib.Nifti1Image(pred_vol_np[0], affine), str(vol_dir / f"{patient_id}_t{t_idx}_pred.nii.gz"))

        if save_viz:
            save_center_slice_report(
                gt_vol_np, pred_vol_np,
                out_dir / "viz" / f"{patient_id}_t{t_idx}.png",
                title=f"{patient_id}  t={t_idx}  MSE={mse:.5f}  SSIM={ssim:.4f}",
            )

    patient_row = {
        "patient_id": patient_id,
        "num_timepoints": len(rows),
        "weights_used": weights_used,
        "mean_mse": nanmean(mse_values), "std_mse": nanstd(mse_values),
        "mean_ms_ssim": nanmean(ms_ssim_values), "std_ms_ssim": nanstd(ms_ssim_values),
        "mean_ssim": nanmean(ssim_values), "std_ssim": nanstd(ssim_values),
        "mean_psnr": nanmean(psnr_values), "std_psnr": nanstd(psnr_values),
        "mean_latent_mse": nanmean(latent_mse_values), "std_latent_mse": nanstd(latent_mse_values),
        "mean_mse_tumor": nanmean(mse_tumor_values), "std_mse_tumor": nanstd(mse_tumor_values),
        "mean_ssim_tumor": nanmean(ssim_tumor_values), "std_ssim_tumor": nanstd(ssim_tumor_values),
        "mean_ms_ssim_tumor": nanmean(ms_ssim_tumor_values), "std_ms_ssim_tumor": nanstd(ms_ssim_tumor_values),
        "mean_psnr_tumor": nanmean(psnr_tumor_values), "std_psnr_tumor": nanstd(psnr_tumor_values),
        "mean_growth_recall": nanmean(growth_recall_values), "std_growth_recall": nanstd(growth_recall_values),
        "mean_shrink_recall": nanmean(shrink_recall_values), "std_shrink_recall": nanstd(shrink_recall_values),
        "mean_change_recall": nanmean(change_recall_values), "std_change_recall": nanstd(change_recall_values),
        "mean_mr_tumor": nanmean(mr_tumor_values), "std_mr_tumor": nanstd(mr_tumor_values),
        "mean_mr_dilated": nanmean(mr_dilated_values), "std_mr_dilated": nanstd(mr_dilated_values),
        "mean_ccc_tumor": nanmean(ccc_tumor_values), "std_ccc_tumor": nanstd(ccc_tumor_values),
        "mean_ccc_dilated": nanmean(ccc_dilated_values), "std_ccc_dilated": nanstd(ccc_dilated_values),
    }
    if restore_native_resolution:
        patient_row.update({
            "mean_mse_native": nanmean(mse_native_values), "std_mse_native": nanstd(mse_native_values),
            "mean_ssim_native": nanmean(ssim_native_values), "std_ssim_native": nanstd(ssim_native_values),
            "mean_ms_ssim_native": nanmean(ms_ssim_native_values), "std_ms_ssim_native": nanstd(ms_ssim_native_values),
            "mean_psnr_native": nanmean(psnr_native_values), "std_psnr_native": nanstd(psnr_native_values),
        })

    print(
        f"[EvalSplit] {patient_id}: n={patient_row['num_timepoints']} "
        f"MSE={patient_row['mean_mse']:.6f} MS-SSIM={patient_row['mean_ms_ssim']:.4f} "
        f"SSIM={patient_row['mean_ssim']:.4f} latent_MSE={patient_row['mean_latent_mse']:.6f} "
        f"| tumor_MSE={patient_row['mean_mse_tumor']:.6f} tumor_SSIM={patient_row['mean_ssim_tumor']:.4f} "
        f"tumor_PSNR={patient_row['mean_psnr_tumor']:.2f} "
        f"growth_recall={patient_row['mean_growth_recall']:.3f}±{patient_row['std_growth_recall']:.3f} "
        f"shrink_recall={patient_row['mean_shrink_recall']:.3f}±{patient_row['std_shrink_recall']:.3f} "
        f"total_recall={patient_row['mean_change_recall']:.3f}±{patient_row['std_change_recall']:.3f} "
        f"| MR(tumor)={patient_row['mean_mr_tumor']:.3f}±{patient_row['std_mr_tumor']:.3f} "
        f"MR(dilated)={patient_row['mean_mr_dilated']:.3f}±{patient_row['std_mr_dilated']:.3f}"
    )

    return rows, patient_row, change_rows


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="TaGeDiff whole-volume quantitative evaluation over a patient split CSV."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--patients-split", required=True, help="CSV containing patient IDs")
    parser.add_argument("--patient-id-col", default="patient_id", help="Patient ID column name")
    parser.add_argument("--split-col", default="split", help="Split column name, if present")
    parser.add_argument("--split", default=None, help="Optional split value to evaluate, e.g. test/val")
    parser.add_argument("--save-dir", default="./evaluations/oncoflow_miu_fm_quantitative", help="Output directory")
    parser.add_argument("--steps", type=int, default=50, help="Number of DDIM/ODE denoising steps")
    parser.add_argument(
        "--ema-policy", default="auto", choices=["auto", "force_ema", "force_raw"],
        help="'auto' (default): EMA if the checkpoint has it, else raw (logged). "
             "'force_ema': require EMA, error if absent. 'force_raw': always evaluate raw weights.",
    )
    parser.add_argument("--use-ema", action="store_true", help="DEPRECATED alias for --ema-policy=force_ema.")
    parser.add_argument("--save-volumes", action="store_true", help="Save each generated/GT volume as .npy (can use significant disk).")
    parser.add_argument("--save-nifti", action="store_true", help="Also save .nii.gz (identity affine — MIU data carries no real affine).")
    parser.add_argument("--no-viz", action="store_true", help="Skip saving center-slice PNG visualizations.")
    parser.add_argument(
        "--restore-native-resolution", action="store_true",
        help="ADDITIONALLY compute a second, clearly-labeled (*_native) set of metrics at each "
             "patient's native on-disk resolution: target = whole-volume-normalized native image "
             "(never resized), prediction = trilinear-upsampled from the model's output shape. "
             "Never replaces or is averaged with the primary (model-shape) metrics.",
    )
    parser.add_argument(
        "--no-register-baseline", action="store_true",
        help="Disable rigid registration of the last context visit onto the target visit's grid "
             "before change metrics (delta_gt/delta_pred, ΔSS/Δ-RMAE, MR/CCC, growth/shrink recall) "
             "are computed. Registration is ON by default — see "
             "src.evaluation.registration.rigid_register_to_fixed's docstring: longitudinal sessions "
             "are not guaranteed pre-registered, and without this most change_metrics.csv rows come "
             "back NaN (registration-induced noise swamps real tumor-region change).",
    )
    parser.add_argument("--override", nargs="*", default=[], help="Override config key=value pairs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--voxel-spacing-mm-override", type=float, nargs=3, default=None,
                         metavar=("DZ", "DY", "DX"),
                         help="Override the approximate (dz,dy,dx) mm/voxel spacing used to report the "
                              "MR_dilated region's dilation radius in mm (see resolve_voxel_spacing_mm's "
                              "docstring for the default's native-1mm-isotropic assumption).")
    args = parser.parse_args()

    ema_policy = "force_ema" if args.use_ema else args.ema_policy

    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_ckpt = read_checkpoint(args.checkpoint)
    cfg = resolve_training_mode(cfg, raw_ckpt, args.checkpoint)
    cfg = resolve_representation(cfg, raw_ckpt, args.checkpoint)
    cfg = resolve_condition_on_treatment(cfg, raw_ckpt, args.checkpoint)
    cfg = resolve_temporal_aggregator_variant(cfg, raw_ckpt, args.checkpoint)
    resolved_training_mode = cfg["training"]["training_mode"]
    resolved_representation = cfg["data"].get("representation", "latent")
    steps_label = (
        "DDIM steps" if resolved_training_mode == "diffusion"
        else f"ODE steps (flow_matching, solver={cfg.get('flow_matching', {}).get('solver', 'euler')})"
    )

    print("=" * 72)
    print("TaGeDiff — Whole-Volume Quantitative Evaluation on Patient Split")
    print("=" * 72)
    print(f"  Device         : {args.device}")
    print(f"  Checkpoint     : {args.checkpoint}")
    print(f"  Training mode  : {resolved_training_mode}")
    print(f"  Representation : {resolved_representation}")
    print(f"  Split CSV      : {args.patients_split}")
    print(f"  Split filter   : {args.split}")
    print(f"  {steps_label:<15}: {args.steps}")
    print(f"  EMA policy     : {ema_policy}")
    print(f"  Register baseline (change metrics): {not args.no_register_baseline}")
    print(f"  Output dir     : {out_dir}")
    print("=" * 72)

    set_seed(args.seed)

    patient_ids = read_patient_ids_from_split(
        csv_path=args.patients_split, patient_id_col=args.patient_id_col,
        split_col=args.split_col, split=args.split,
    )
    print(f"[EvalSplit] Found {len(patient_ids)} patient(s).")

    model = TaGeDiff(cfg).to(args.device)
    model.eval()
    ckpt_info = load_checkpoint(model, raw_ckpt, ema_policy=ema_policy, device=args.device)
    weights_used = ckpt_info["weights_used"]
    geno_preproc = ckpt_info["geno_preproc"]
    image_stats = ckpt_info["image_stats"]
    normalization_policy = ckpt_info["normalization_policy"]
    normalization_clip_pct = ckpt_info["normalization_clip_pct"]
    data_range = resolve_data_range(normalization_policy, image_stats)
    apply_n4 = bool(cfg["data"].get("apply_n4_bias_correction", False))
    n4_cache_dir = cfg["data"].get("n4_cache_dir")
    n4_kwargs = n4_kwargs_from_dc(cfg["data"])
    if resolved_representation == "voxel":
        volume_size = tuple(cfg["data"].get("voxel_shape", ckpt_info["voxel_shape"] or (64, 96, 96)))
    else:
        volume_size = tuple(cfg["data"].get("volume_size", ckpt_info["volume_size"]))
    use_amp = cfg.get("training", {}).get("mixed_precision", True)
    voxel_spacing_mm = resolve_voxel_spacing_mm(volume_size, spacing_override=args.voxel_spacing_mm_override)
    dilation_mm_summary = float(ROI_DILATION_ITERS * np.mean(voxel_spacing_mm))
    print(f"[EvalSplit] Approx. voxel spacing (dz,dy,dx) mm: {voxel_spacing_mm} "
          f"({'override' if args.voxel_spacing_mm_override else 'derived from native ~1mm-isotropic assumption'})")

    all_rows = []
    patient_rows = []
    failed_rows = []
    all_change_rows = []

    for patient_id in tqdm(patient_ids, desc="Patients"):
        try:
            patient_data = load_patient(cfg["data"]["data_dir"], patient_id, cfg, geno_preproc=geno_preproc)
            rows, patient_row, change_rows = evaluate_patient(
                model=model, patient_id=patient_id, patient_data=patient_data,
                volume_size=volume_size, steps=args.steps, device=args.device, use_amp=use_amp,
                weights_used=weights_used, normalization_policy=normalization_policy,
                normalization_clip_pct=normalization_clip_pct, out_dir=out_dir,
                save_volumes=args.save_volumes, save_nifti=args.save_nifti, save_viz=not args.no_viz,
                representation=resolved_representation, restore_native_resolution=args.restore_native_resolution,
                image_stats=image_stats, apply_n4=apply_n4, n4_cache_dir=n4_cache_dir, n4_kwargs=n4_kwargs,
                voxel_spacing_mm=voxel_spacing_mm, register_baseline=not args.no_register_baseline,
            )
            all_rows.extend(rows)
            patient_rows.append(patient_row)
            all_change_rows.extend(change_rows)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[EvalSplit] ERROR for patient {patient_id}: {exc}")
            failed_rows.append({"patient_id": patient_id, "error": repr(exc)})

    timepoint_csv = out_dir / "timepoint_metrics.csv"
    patient_csv = out_dir / "patient_metrics.csv"
    summary_csv = out_dir / "summary_metrics.csv"
    failed_csv = out_dir / "failed_patients.csv"
    change_metrics_csv = out_dir / "change_metrics.csv"

    timepoint_fields = [
        "patient_id", "timepoint", "num_previous_timepoints", "previous_treatments",
        "target_treatment", "is_tumor_positive_target", "mse", "ms_ssim", "ssim", "psnr",
        "latent_mse", "mse_tumor", "ssim_tumor", "ms_ssim_tumor", "psnr_tumor", "weights_used",
        "roi_voxels", "gt_growth_voxels", "gt_shrink_voxels",
        "growth_recall", "shrink_recall", "change_recall",
        "wrong_direction_rate", "tumor_delta_mae",
        "mr_tumor", "std_pred_tumor", "std_gt_tumor", "ccc_tumor", "n_voxels_tumor", "nan_reason_tumor",
        "mr_dilated", "std_pred_dilated", "std_gt_dilated", "ccc_dilated", "n_voxels_dilated", "nan_reason_dilated",
        "mask_used", "dilation_mm",
    ]
    patient_fields = [
        "patient_id", "num_timepoints", "weights_used",
        "mean_mse", "std_mse", "mean_ms_ssim", "std_ms_ssim", "mean_ssim", "std_ssim",
        "mean_psnr", "std_psnr", "mean_latent_mse", "std_latent_mse",
        "mean_mse_tumor", "std_mse_tumor", "mean_ssim_tumor", "std_ssim_tumor",
        "mean_ms_ssim_tumor", "std_ms_ssim_tumor", "mean_psnr_tumor", "std_psnr_tumor",
        "mean_growth_recall", "std_growth_recall",
        "mean_shrink_recall", "std_shrink_recall",
        "mean_change_recall", "std_change_recall",
        "mean_mr_tumor", "std_mr_tumor", "mean_mr_dilated", "std_mr_dilated",
        "mean_ccc_tumor", "std_ccc_tumor", "mean_ccc_dilated", "std_ccc_dilated",
    ]
    if args.restore_native_resolution:
        timepoint_fields += ["mse_native", "ssim_native", "ms_ssim_native", "psnr_native", "native_shape"]
        patient_fields += [
            "mean_mse_native", "std_mse_native", "mean_ssim_native", "std_ssim_native",
            "mean_ms_ssim_native", "std_ms_ssim_native", "mean_psnr_native", "std_psnr_native",
        ]

    write_csv(timepoint_csv, all_rows, fieldnames=timepoint_fields)
    write_csv(patient_csv, patient_rows, fieldnames=patient_fields)
    write_csv(change_metrics_csv, all_change_rows, fieldnames=CHANGE_METRICS_FIELDS)
    if failed_rows:
        write_csv(failed_csv, failed_rows, fieldnames=["patient_id", "error"])

    global_mse = nanmean([r["mse"] for r in all_rows])
    global_ms_ssim = nanmean([r["ms_ssim"] for r in all_rows])
    global_ssim = nanmean([r["ssim"] for r in all_rows])
    global_psnr = nanmean([r["psnr"] for r in all_rows])
    global_latent_mse = nanmean([r["latent_mse"] for r in all_rows])
    global_mse_tumor = nanmean([r["mse_tumor"] for r in all_rows])
    global_ssim_tumor = nanmean([r["ssim_tumor"] for r in all_rows])
    global_ms_ssim_tumor = nanmean([r["ms_ssim_tumor"] for r in all_rows])
    global_psnr_tumor = nanmean([r["psnr_tumor"] for r in all_rows])

    patient_growth_means = [p["mean_growth_recall"] for p in patient_rows]
    patient_shrink_means = [p["mean_shrink_recall"] for p in patient_rows]
    patient_change_means = [p["mean_change_recall"] for p in patient_rows]
    global_growth_recall = nanmean(patient_growth_means)
    global_growth_recall_std = nanstd(patient_growth_means)
    global_shrink_recall = nanmean(patient_shrink_means)
    global_shrink_recall_std = nanstd(patient_shrink_means)
    global_change_recall = nanmean(patient_change_means)
    global_change_recall_std = nanstd(patient_change_means)

    patient_mr_tumor_means = [p["mean_mr_tumor"] for p in patient_rows]
    patient_mr_dilated_means = [p["mean_mr_dilated"] for p in patient_rows]
    patient_ccc_tumor_means = [p["mean_ccc_tumor"] for p in patient_rows]
    patient_ccc_dilated_means = [p["mean_ccc_dilated"] for p in patient_rows]
    global_mr_tumor = nanmean(patient_mr_tumor_means)
    global_mr_tumor_std = nanstd(patient_mr_tumor_means)
    global_mr_dilated = nanmean(patient_mr_dilated_means)
    global_mr_dilated_std = nanstd(patient_mr_dilated_means)
    global_ccc_tumor = nanmean(patient_ccc_tumor_means)
    global_ccc_tumor_std = nanstd(patient_ccc_tumor_means)
    global_ccc_dilated = nanmean(patient_ccc_dilated_means)
    global_ccc_dilated_std = nanstd(patient_ccc_dilated_means)

    summary_rows = [{
        "num_patients_requested": len(patient_ids),
        "num_patients_evaluated": len(patient_rows),
        "num_failed_patients": len(failed_rows),
        "num_timepoints": len(all_rows),
        "weights_used": weights_used,
        "ssim_definition": ssim_definition_string(data_range),
        "data_range": data_range,
        "normalization_policy": normalization_policy,
        "representation": resolved_representation,
        "volume_size": str(volume_size),
        "unet_config": str(cfg.get("unet", {})),
        "vae_config": str(cfg.get("vae", {})),
        "average_mse": global_mse,
        "average_ms_ssim": global_ms_ssim,
        "average_ssim": global_ssim,
        "average_psnr": global_psnr,
        "average_latent_mse": global_latent_mse,
        "average_mse_tumor": global_mse_tumor,
        "average_ssim_tumor": global_ssim_tumor,
        "average_ms_ssim_tumor": global_ms_ssim_tumor,
        "average_psnr_tumor": global_psnr_tumor,
        "average_growth_recall": global_growth_recall, "std_growth_recall": global_growth_recall_std,
        "average_shrink_recall": global_shrink_recall, "std_shrink_recall": global_shrink_recall_std,
        "average_change_recall": global_change_recall, "std_change_recall": global_change_recall_std,
        "average_mr_tumor": global_mr_tumor, "std_mr_tumor": global_mr_tumor_std,
        "average_mr_dilated": global_mr_dilated, "std_mr_dilated": global_mr_dilated_std,
        "average_ccc_tumor": global_ccc_tumor, "std_ccc_tumor": global_ccc_tumor_std,
        "average_ccc_dilated": global_ccc_dilated, "std_ccc_dilated": global_ccc_dilated_std,
        "dilation_mm": dilation_mm_summary,
        "restore_native_resolution": args.restore_native_resolution,
        "register_baseline": not args.no_register_baseline,
        "lpips_note": "not computed — no 3D LPIPS available; volumetric SSIM/MS-SSIM/PSNR/MSE used instead",
    }]
    if args.restore_native_resolution:
        summary_rows[0].update({
            "average_mse_native": nanmean([r["mse_native"] for r in all_rows]),
            "average_ssim_native": nanmean([r["ssim_native"] for r in all_rows]),
            "average_ms_ssim_native": nanmean([r["ms_ssim_native"] for r in all_rows]),
            "average_psnr_native": nanmean([r["psnr_native"] for r in all_rows]),
        })
    write_csv(summary_csv, summary_rows, fieldnames=list(summary_rows[0].keys()))

    print("\n[EvalSplit] Aggregate metrics (whole-volume, weights=%s)" % weights_used)
    print(f"  Patients requested : {len(patient_ids)}")
    print(f"  Patients evaluated : {len(patient_rows)}")
    print(f"  Failed patients    : {len(failed_rows)}")
    print(f"  Timepoints evaluated: {len(all_rows)}")
    print(f"  Average MSE        : {global_mse:.6f}")
    print(f"  Average MS-SSIM    : {global_ms_ssim:.4f}")
    print(f"  Average SSIM       : {global_ssim:.4f} ({ssim_definition_string(data_range)})")
    print(f"  Average PSNR       : {global_psnr:.2f} dB")
    print(f"  Average latent MSE : {global_latent_mse:.6f}")
    print(f"  Average MSE (tumor): {global_mse_tumor:.6f}")
    print(f"  Average SSIM (tumor): {global_ssim_tumor:.4f}")
    print(f"  Average MS-SSIM (tumor): {global_ms_ssim_tumor:.4f}")
    print(f"  Average PSNR (tumor): {global_psnr_tumor:.2f} dB")
    print(f"  Growth recall      : {global_growth_recall:.3f} ± {global_growth_recall_std:.3f}")
    print(f"  Shrink recall      : {global_shrink_recall:.3f} ± {global_shrink_recall_std:.3f}")
    print(f"  Total (change) recall: {global_change_recall:.3f} ± {global_change_recall_std:.3f}")
    print(f"  MR (tumor)         : {global_mr_tumor:.3f} ± {global_mr_tumor_std:.3f} "
          f"(CCC {global_ccc_tumor:.3f} ± {global_ccc_tumor_std:.3f})")
    print(f"  MR (dilated {dilation_mm_summary:.1f}mm)  : {global_mr_dilated:.3f} ± {global_mr_dilated_std:.3f} "
          f"(CCC {global_ccc_dilated:.3f} ± {global_ccc_dilated_std:.3f})")
    print("\n[EvalSplit] Saved outputs")
    print(f"  Timepoint metrics  : {timepoint_csv}")
    print(f"  Patient metrics    : {patient_csv}")
    print(f"  Change metrics     : {change_metrics_csv} "
          f"({len(all_change_rows)} rows: patient x timepoint x method[model/copy_baseline/oracle] x region[tumor/dilated/control/full])")
    print(f"  Summary metrics    : {summary_csv}")
    if failed_rows:
        print(f"  Failed patients    : {failed_csv}")


if __name__ == "__main__":
    main()
