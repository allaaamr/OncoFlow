from __future__ import annotations

from typing import Dict, Optional

import numpy as np

MIN_REGION_VOXELS = 100
K_NOISE = 3.0
EPS = 1e-12


def robust_sigma_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def estimate_case_noise(d_gt: np.ndarray, control_mask: np.ndarray) -> Dict[str, float]:
    control_mask = control_mask.astype(bool)
    n = int(control_mask.sum())
    if n < MIN_REGION_VOXELS:
        return {"sigma_noise": float("nan"), "drift": float("nan"), "drift_flag": False, "n_control_voxels": n}

    vals = np.asarray(d_gt, dtype=np.float64)[control_mask]
    sigma_noise = robust_sigma_mad(vals)
    drift = float(np.median(vals))
    drift_flag = bool(np.isfinite(sigma_noise) and abs(drift) > sigma_noise)
    return {"sigma_noise": sigma_noise, "drift": drift, "drift_flag": drift_flag, "n_control_voxels": n}


def _eligibility(d_gt_region: np.ndarray, n_voxels: int, sigma_noise: float) -> Optional[str]:
    if n_voxels < MIN_REGION_VOXELS:
        return f"n_voxels={n_voxels} < MIN_REGION_VOXELS={MIN_REGION_VOXELS}"
    if np.isfinite(sigma_noise):
        mean_sq = float(np.mean(d_gt_region.astype(np.float64) ** 2))
        noise_floor_sq = (K_NOISE * sigma_noise) ** 2
        if mean_sq < noise_floor_sq:
            return (
                f"mean(d_gt^2)={mean_sq:.3g} < ({K_NOISE:g}*sigma_noise)^2={noise_floor_sq:.3g} "
                "(region change indistinguishable from noise)"
            )
    return None


def _nan_result(n_voxels: int, sigma_noise: float, drift: float, nan_reason: str) -> Dict[str, float]:
    nan = float("nan")
    return {
        "delta_ss": nan, "r_pattern": nan, "m_magnitude": nan,
        "delta_rmae": nan, "delta_ss_l1": nan,
        "norm_d_gt": nan, "norm_d_pred": nan,
        "n_voxels": n_voxels, "sigma_noise": sigma_noise, "drift": drift,
        "delta_ss_max": nan, "nan_reason": nan_reason,
        "sum_sq_resid": nan, "sum_sq_gt": nan, "sum_sq_pred": nan, "sum_dot_gt_pred": nan,
        "sum_abs_resid": nan, "sum_abs_avg_denom": nan, "sum_abs_gt": nan,
    }


def compute_region_change_metrics(
    d_gt: np.ndarray,
    d_pred: np.ndarray,
    region_mask: np.ndarray,
    sigma_noise: float,
    drift: float,
) -> Dict[str, float]:
    region_mask = region_mask.astype(bool)
    n_voxels = int(region_mask.sum())

    g = np.asarray(d_gt, dtype=np.float64)[region_mask]
    p = np.asarray(d_pred, dtype=np.float64)[region_mask]

    reason = _eligibility(g, n_voxels, sigma_noise)
    if reason is not None:
        return _nan_result(n_voxels, sigma_noise, drift, reason)

    resid = g - p
    sum_sq_resid = float(np.sum(resid ** 2))
    sum_sq_gt = float(np.sum(g ** 2))
    sum_sq_pred = float(np.sum(p ** 2))
    sum_dot_gt_pred = float(np.dot(g, p))
    sum_abs_resid = float(np.sum(np.abs(resid)))
    sum_abs_avg_denom = float(np.sum(0.5 * (np.abs(g) + np.abs(p))))
    sum_abs_gt = float(np.sum(np.abs(g)))

    norm_gt = float(np.sqrt(sum_sq_gt))
    norm_pred = float(np.sqrt(sum_sq_pred))

    delta_ss = 1.0 - sum_sq_resid / sum_sq_gt if sum_sq_gt > EPS else float("nan")

    if norm_gt > EPS and norm_pred > EPS:
        m_magnitude = norm_pred / norm_gt
        r_pattern = float(np.dot(p, g)) / (norm_pred * norm_gt)
    elif norm_pred <= EPS:
        m_magnitude = 0.0
        r_pattern = float("nan")
    else:
        m_magnitude = float("nan")
        r_pattern = float("nan")

    delta_rmae = sum_abs_resid / sum_abs_avg_denom if sum_abs_avg_denom > EPS else float("nan")
    delta_ss_l1 = 1.0 - sum_abs_resid / sum_abs_gt if sum_abs_gt > EPS else float("nan")

    mean_sq_gt = sum_sq_gt / n_voxels
    delta_ss_max = (
        1.0 - (sigma_noise ** 2) / mean_sq_gt
        if np.isfinite(sigma_noise) and mean_sq_gt > EPS else float("nan")
    )

    return {
        "delta_ss": delta_ss, "r_pattern": r_pattern, "m_magnitude": m_magnitude,
        "delta_rmae": delta_rmae, "delta_ss_l1": delta_ss_l1,
        "norm_d_gt": norm_gt, "norm_d_pred": norm_pred,
        "n_voxels": n_voxels, "sigma_noise": sigma_noise, "drift": drift,
        "delta_ss_max": delta_ss_max, "nan_reason": "",
        "sum_sq_resid": sum_sq_resid, "sum_sq_gt": sum_sq_gt,
        "sum_sq_pred": sum_sq_pred, "sum_dot_gt_pred": sum_dot_gt_pred,
        "sum_abs_resid": sum_abs_resid, "sum_abs_avg_denom": sum_abs_avg_denom, "sum_abs_gt": sum_abs_gt,
    }


def build_control_mask(brain_mask: np.ndarray, tumor_union_mask: np.ndarray, control_dilation_iters: int) -> np.ndarray:
    from scipy.ndimage import binary_dilation

    brain_mask = brain_mask.astype(bool)
    tumor_union_mask = tumor_union_mask.astype(bool)
    if tumor_union_mask.any():
        dilated = binary_dilation(tumor_union_mask, iterations=max(1, int(control_dilation_iters)))
    else:
        dilated = tumor_union_mask
    return brain_mask & ~dilated
