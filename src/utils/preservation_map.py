import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


PRESERVATION_MODES = ("distance", "local_density", "hard_dilation", "horizon_decay")


def _resize_binary_mask(baseline_mask: torch.Tensor, target_shape=None) -> torch.Tensor:

    m = baseline_mask
    if m.ndim == 4:
        m = m.unsqueeze(1)
    elif m.ndim != 5:
        raise ValueError(
            f"expected baseline_mask with 4 or 5 dims ((B,D,H,W) or (B,1,D,H,W)), "
            f"got shape {tuple(baseline_mask.shape)}"
        )
    m = (m > 0).float()

    if target_shape is not None:
        target_shape = tuple(int(s) for s in target_shape)
        if len(target_shape) != 3:
            raise ValueError(f"target_shape must be a (D,H,W) triple, got {target_shape}")
        m = F.interpolate(m, size=target_shape, mode="nearest")
    return m


def build_distance_preservation_map(
    baseline_mask: torch.Tensor,
    sigma: float,
    spatial_spacing=None,
    target_shape=None,
) -> torch.Tensor:
    
    if sigma <= 0:
        raise ValueError(f"region_aware_source.sigma must be > 0, got {sigma}")

    m = _resize_binary_mask(baseline_mask, target_shape)
    device = m.device
    mask_np = (m.detach().cpu().numpy() > 0)

    P_np = np.empty(mask_np.shape, dtype=np.float32)
    for idx in np.ndindex(mask_np.shape[:2]):
        vol = mask_np[idx]
        if not vol.any():
            P_np[idx] = 1.0
            continue
        d = distance_transform_edt(~vol, sampling=spatial_spacing)
        P_np[idx] = 1.0 - np.exp(-(d.astype(np.float32) ** 2) / (2.0 * float(sigma) ** 2))

    P = torch.from_numpy(P_np).to(device=device, dtype=torch.float32)
    return P.clamp_(0.0, 1.0)


def compute_local_tumor_density(
    baseline_mask: torch.Tensor,
    kernel_size=(5, 11, 11),
    target_shape=None,
) -> torch.Tensor:
    
    kD, kH, kW = (int(k) for k in kernel_size)
    if kD % 2 == 0 or kH % 2 == 0 or kW % 2 == 0:
        raise ValueError(
            f"region_aware_source.kernel_size must have all-odd components for "
            f"symmetric 'same'-shape padding, got {(kD, kH, kW)}"
        )

    m = _resize_binary_mask(baseline_mask, target_shape)
    padding = ((kD - 1) // 2, (kH - 1) // 2, (kW - 1) // 2)
    R = F.avg_pool3d(m, kernel_size=(kD, kH, kW), stride=1, padding=padding, count_include_pad=True)
    return R.clamp(0.0, 1.0)


def build_local_density_preservation_map(
    baseline_mask: torch.Tensor,
    kernel_size=(5, 11, 11),
    p_min: float = 0.1,
    gamma: float = 1.0,
    target_shape=None,
    return_density: bool = False,
):
   
    if not (0.0 <= p_min <= 1.0):
        raise ValueError(f"region_aware_source.p_min must be in [0, 1], got {p_min}")
    if gamma <= 0:
        raise ValueError(f"region_aware_source.gamma must be > 0, got {gamma}")

    R = compute_local_tumor_density(baseline_mask, kernel_size=kernel_size, target_shape=target_shape)
    P = 1.0 - (1.0 - float(p_min)) * R.pow(float(gamma))
    P = P.clamp(float(p_min), 1.0)

    if return_density:
        return P, R
    return P


def build_hard_dilation_preservation_map(
    baseline_mask: torch.Tensor,
    kernel_size=(5, 11, 11),
    target_shape=None,
    return_dilated_mask: bool = False,
):
   
    kD, kH, kW = (int(k) for k in kernel_size)
    if kD % 2 == 0 or kH % 2 == 0 or kW % 2 == 0:
        raise ValueError(
            f"region_aware_source.kernel_size must have all-odd components for "
            f"symmetric 'same'-shape padding, got {(kD, kH, kW)}"
        )
    if kD <= 0 or kH <= 0 or kW <= 0:
        raise ValueError(
            f"region_aware_source.kernel_size components must all be > 0, got {(kD, kH, kW)}"
        )

    m = _resize_binary_mask(baseline_mask, target_shape)
    padding = ((kD - 1) // 2, (kH - 1) // 2, (kW - 1) // 2)
    dilated = F.max_pool3d(m, kernel_size=(kD, kH, kW), stride=1, padding=padding)
    dilated = (dilated > 0).to(dtype=m.dtype)

    P = 1.0 - dilated

    if return_dilated_mask:
        return P, dilated
    return P


def bg_retention_to_lambda(bg_retention: float, bg_retention_horizon_days: float) -> float:
    
    if not (0.0 < bg_retention <= 1.0):
        raise ValueError(
            f"region_aware_source.horizon_decay.bg_retention must be in (0, 1], got {bg_retention}"
        )
    if bg_retention_horizon_days <= 0:
        raise ValueError(
            "region_aware_source.horizon_decay.bg_retention_horizon_days must be > 0, got "
            f"{bg_retention_horizon_days}"
        )
    return -math.log(float(bg_retention)) / float(bg_retention_horizon_days)


def build_horizon_decay_preservation_map(
    baseline_mask: torch.Tensor,
    dt: torch.Tensor,
    lambda_bg: float,
    kernel_size=(5, 11, 11),
    gamma: float = 1.0,
    target_shape=None,
    return_density: bool = False,
):
    
    if gamma <= 0:
        raise ValueError(f"region_aware_source.horizon_decay.gamma must be > 0, got {gamma}")
    if lambda_bg < 0:
        raise ValueError(f"region_aware_source.horizon_decay lambda_bg must be >= 0, got {lambda_bg}")

    B = baseline_mask.shape[0]
    dt_ = torch.as_tensor(dt, dtype=torch.float32).reshape(-1)
    if dt_.shape[0] != B:
        raise ValueError(
            f"horizon_decay dt must have one value per batch element (B={B}), got shape {tuple(dt_.shape)}"
        )

    R = compute_local_tumor_density(baseline_mask, kernel_size=kernel_size, target_shape=target_shape)
    dt_ = dt_.to(device=R.device, dtype=R.dtype).view(B, 1, 1, 1, 1)

    w = R.pow(float(gamma))
    temporal = torch.exp(-float(lambda_bg) * dt_)
    P = (1.0 - w) * temporal
    P = P.clamp(0.0, 1.0)

    if return_density:
        return P, R
    return P


def build_preservation_map(
    baseline_mask: torch.Tensor,
    mode: str,
    target_shape=None,
    sigma: float = None,
    kernel_size=None,
    p_min: float = None,
    gamma: float = None,
    spatial_spacing=None,
    dt: torch.Tensor = None,
    lambda_bg: float = None,
):
    
    if mode == "distance":
        return build_distance_preservation_map(
            baseline_mask, sigma=sigma, spatial_spacing=spatial_spacing, target_shape=target_shape,
        )
    elif mode == "local_density":
        return build_local_density_preservation_map(
            baseline_mask, kernel_size=kernel_size, p_min=p_min, gamma=gamma, target_shape=target_shape,
        )
    elif mode == "hard_dilation":
        return build_hard_dilation_preservation_map(
            baseline_mask, kernel_size=kernel_size, target_shape=target_shape,
        )
    elif mode == "horizon_decay":
        return build_horizon_decay_preservation_map(
            baseline_mask, dt=dt, lambda_bg=lambda_bg, kernel_size=kernel_size, gamma=gamma,
            target_shape=target_shape,
        )
    else:
        raise ValueError(
            f"Unknown region_aware_source.preservation_mode={mode!r}, expected one of {PRESERVATION_MODES}"
        )
