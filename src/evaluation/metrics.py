import numpy as np
import torch

import argparse
import os
import sys
import csv
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional
import random
import warnings
from src.evaluation.ssim import SSIM as _SSIM_torchmsssim
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.TaGeDiff import TaGeDiff
from src.data.dataset import PatientDataset
from src.data.preprocessing import ImageIntensityStats

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    from PIL import Image as PILImage

try:
    from pytorch_msssim import ms_ssim as ms_ssim_torch
    HAS_MS_SSIM = True
except ImportError:
    HAS_MS_SSIM = False

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    HAS_FID = True
except ImportError:
    HAS_FID = False


DATA_RANGE = 1.0

SSIM_DEFINITION = "gaussian_window_pytorch_msssim_win11_sigma1.5_data_range1.0"


def ssim_definition_string(data_range: float) -> str:
    return f"gaussian_window_pytorch_msssim_win11_sigma1.5_data_range{data_range:g}"


def resolve_data_range(
    normalization_policy: str,
    image_stats: Optional[Dict[int, "ImageIntensityStats"]] = None,
    flair_index: int = 2,
) -> float:
    if normalization_policy == "whole_volume":
        return DATA_RANGE
    if normalization_policy == "frozen_train_stats":
        if not image_stats or flair_index not in image_stats or not image_stats[flair_index].is_fit:
            raise ValueError(
                "[resolve_data_range] normalization_policy='frozen_train_stats' requires a fitted "
                f"image_stats[{flair_index}] (ImageIntensityStats) to compute the correct PSNR/SSIM "
                "data_range — pass the same frozen stats already loaded from the checkpoint/"
                "scripts/fit_image_stats.py, never refit here."
            )
        stats = image_stats[flair_index]
        std_safe = max(float(stats.std), stats.eps)
        return float((stats.clip_hi - stats.clip_lo) / std_safe)
    raise ValueError(f"[resolve_data_range] Unknown normalization_policy={normalization_policy!r}")


_SSIM_MODULES = {}
def _get_ssim_module(device: str, spatial_dims: int = 2, data_range: float = DATA_RANGE):
    key = (device, spatial_dims, data_range)
    if key not in _SSIM_MODULES:
        _SSIM_MODULES[key] = _SSIM_torchmsssim(
            win_size=11,
            win_sigma=1.5,
            data_range=data_range,
            size_average=True,
            channel=1,
            spatial_dims=spatial_dims,
        ).to(device)
    return _SSIM_MODULES[key]


def _to_metric_tensor(img: np.ndarray, device: str, spatial_dims: int = 2) -> torch.Tensor:
    ndim_no_channel = spatial_dims
    ndim_with_channel = spatial_dims + 1
    if img.ndim == ndim_no_channel:
        arr = img[None, ...]
    elif img.ndim == ndim_with_channel:
        arr = img[:1, ...]
    else:
        raise ValueError(
            f"Unexpected image shape {img.shape} for spatial_dims={spatial_dims} "
            f"(expected ndim {ndim_no_channel} or {ndim_with_channel})"
        )
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    t = t.unsqueeze(0)
    return t.to(device)
 

def compute_mse(gt: np.ndarray, pred: np.ndarray) -> float:
    gt32 = np.ascontiguousarray(gt, dtype=np.float32)
    pr32 = np.ascontiguousarray(pred, dtype=np.float32)
    if gt32.shape != pr32.shape:
        raise ValueError(f"compute_mse: shape mismatch gt={gt32.shape} vs pred={pr32.shape}")
    return float(np.mean((gt32 - pr32) ** 2))
 
 
def compute_psnr(gt: np.ndarray, pred: np.ndarray,
                 data_range: float = DATA_RANGE) -> float:
    mse = compute_mse(gt, pred)
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))
 
 
def compute_ssim(gt: np.ndarray, pred: np.ndarray, device: str = "cpu", spatial_dims: int = 2,
                  data_range: float = DATA_RANGE) -> float:
    ssim_mod = _get_ssim_module(device, spatial_dims=spatial_dims, data_range=data_range)
    with torch.no_grad():
        x = _to_metric_tensor(pred, device, spatial_dims=spatial_dims)
        y = _to_metric_tensor(gt, device, spatial_dims=spatial_dims)
        val = ssim_mod(x, y)
    return float(val.detach().cpu())


def compute_ms_ssim(gt: np.ndarray, pred: np.ndarray, device: str = "cpu", spatial_dims: int = 2,
                     data_range: float = DATA_RANGE) -> float:
    try:
        from pytorch_msssim import ms_ssim as _ms_ssim_fn
    except ImportError:
        return float("nan")

    x = _to_metric_tensor(pred, device, spatial_dims=spatial_dims)
    y = _to_metric_tensor(gt, device, spatial_dims=spatial_dims)
    try:
        with torch.no_grad():
            val = _ms_ssim_fn(x, y, data_range=data_range, size_average=True)
        return float(val.detach().cpu().item())
    except AssertionError:
        return float("nan")


def image_for_lpips(img: np.ndarray, device: str) -> torch.Tensor:
    t = _to_metric_tensor(img, device).clamp(0.0, 1.0)
    t = t.repeat(1, 3, 1, 1)
    return t * 2.0 - 1.0


def compute_lpips(gt: np.ndarray, pred: np.ndarray, lpips_model, device: str = "cpu") -> float:
    if lpips_model is None:
        return float("nan")
    gt_t = image_for_lpips(gt, device)
    pred_t = image_for_lpips(pred, device)
    with torch.no_grad():
        val = lpips_model(gt_t, pred_t)
    return float(val.detach().cpu().item())
 