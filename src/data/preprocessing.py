import collections
import hashlib
import inspect
import os
import warnings
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import SimpleITK as sitk
    N4_AVAILABLE = True
except ImportError:
    sitk = None
    N4_AVAILABLE = False

GENOMICS_PREPROC_VERSION = 1
IMAGE_INTENSITY_STATS_VERSION = 1

NORMALIZATION_POLICIES = ("whole_volume", "frozen_train_stats")

REPRESENTATION_MODES = ("latent", "voxel")


class GenomicsPreprocessor:

    def __init__(
        self,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        zero_variance_mask: Optional[np.ndarray] = None,
        dim: Optional[int] = None,
        eps: float = 1e-6,
        version: int = GENOMICS_PREPROC_VERSION,
    ):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        self.zero_variance_mask = (
            None if zero_variance_mask is None else np.asarray(zero_variance_mask, dtype=bool)
        )
        self.dim = dim
        self.eps = eps
        self.version = version

    @property
    def is_fit(self) -> bool:
        return self.mean is not None and self.std is not None

    def fit(self, train_geno: np.ndarray) -> "GenomicsPreprocessor":
        arr = np.asarray(train_geno, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"expected a (N_patients, G) array, got shape {arr.shape}")
        if arr.shape[0] == 0:
            raise ValueError("cannot fit GenomicsPreprocessor on zero training patients")
        if not np.isfinite(arr).all():
            raise ValueError("non-finite values in training genomics; cannot fit preprocessor")

        mean = arr.mean(axis=0)
        std_raw = arr.std(axis=0)
        zero_variance = std_raw < self.eps
        safe_std = std_raw.copy()
        safe_std[zero_variance] = 1.0

        self.mean = mean.astype(np.float32)
        self.std = safe_std.astype(np.float32)
        self.zero_variance_mask = zero_variance
        self.dim = int(arr.shape[1])
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("GenomicsPreprocessor.transform() called before fit()/from_dict()")
        x = np.asarray(x, dtype=np.float32)
        out = (x - self.mean) / (self.std + self.eps)
        out = out.astype(np.float32)
        if not np.isfinite(out).all():
            raise ValueError("GenomicsPreprocessor.transform produced non-finite output")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
            "zero_variance_mask": (
                None if self.zero_variance_mask is None else self.zero_variance_mask.tolist()
            ),
            "dim": self.dim,
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "GenomicsPreprocessor":
        if not d:
            return cls()
        mean = d.get("mean")
        std = d.get("std")
        zvm = d.get("zero_variance_mask")
        return cls(
            mean=None if mean is None else np.asarray(mean, dtype=np.float32),
            std=None if std is None else np.asarray(std, dtype=np.float32),
            zero_variance_mask=None if zvm is None else np.asarray(zvm, dtype=bool),
            dim=d.get("dim"),
            eps=d.get("eps", 1e-6),
            version=d.get("version", GENOMICS_PREPROC_VERSION),
        )


def crop_pad_volume_3d(arr: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
    D, H, W = arr.shape[-3:]
    tD, tH, tW = target_size

    sd = max(0, (D - tD) // 2)
    sh = max(0, (H - tH) // 2)
    sw = max(0, (W - tW) // 2)
    arr = arr[..., sd:sd + min(D, tD), sh:sh + min(H, tH), sw:sw + min(W, tW)]

    pD = max(0, tD - arr.shape[-3])
    pH = max(0, tH - arr.shape[-2])
    pW = max(0, tW - arr.shape[-1])
    if pD > 0 or pH > 0 or pW > 0:
        pad_widths = [(0, 0)] * (arr.ndim - 3) + [(0, pD), (0, pH), (0, pW)]
        arr = np.pad(arr, pad_widths, mode="constant", constant_values=0)
    return arr


def resize_volume_3d(arr: np.ndarray, target_size: Tuple[int, int, int], mode: str = "trilinear") -> np.ndarray:
    if mode not in ("trilinear", "nearest"):
        raise ValueError(f"resize_volume_3d: unsupported mode={mode!r}, expected 'trilinear' or 'nearest'")

    lead_shape = arr.shape[:-3]
    n_lead = int(np.prod(lead_shape)) if lead_shape else 1
    x = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).reshape(1, n_lead, *arr.shape[-3:])

    interp_kwargs = {} if mode == "nearest" else {"align_corners": False}
    with torch.no_grad():
        out = F.interpolate(x, size=tuple(target_size), mode=mode, **interp_kwargs)

    out_np = out.reshape(*lead_shape, *target_size).numpy()
    return out_np


def to_model_shape_3d(
    arr: np.ndarray,
    target_size: Tuple[int, int, int],
    representation: str = "latent",
    is_label: bool = False,
) -> np.ndarray:
    if representation == "latent":
        return crop_pad_volume_3d(arr, target_size)
    elif representation == "voxel":
        mode = "nearest" if is_label else "trilinear"
        return resize_volume_3d(arr, target_size, mode=mode)
    else:
        raise ValueError(
            f"Unknown representation={representation!r}, expected one of {REPRESENTATION_MODES}"
        )


def normalize_volume_whole(
    volume: np.ndarray, clip_pct: Tuple[float, float] = (0.5, 99.5), eps: float = 1e-6
) -> np.ndarray:
    volume = np.asarray(volume).astype(np.float32)
    mask = volume != 0
    out = np.zeros_like(volume, dtype=np.float32)

    if mask.sum() < 50:
        return out

    vals = volume[mask]
    lo, hi = np.percentile(vals, clip_pct[0]), np.percentile(vals, clip_pct[1])
    if (hi - lo) < eps:
        return out

    clipped = np.clip(volume, lo, hi)
    normalized = (clipped - lo) / (hi - lo + eps)
    out[mask] = normalized[mask]
    return out


def compute_foreground_mask(volume: np.ndarray) -> np.ndarray:
    return np.asarray(volume) != 0


def sanitize_non_finite(volume: np.ndarray, fill_value: float = 0.0) -> Tuple[np.ndarray, int]:
    volume = np.asarray(volume, dtype=np.float32)
    bad = ~np.isfinite(volume)
    n_bad = int(bad.sum())
    if n_bad:
        volume = volume.copy()
        volume[bad] = fill_value
    return volume, n_bad


def apply_n4_bias_correction(
    volume: np.ndarray,
    mask: np.ndarray,
    shrink_factor: int = 4,
    max_iterations: Tuple[int, ...] = (50, 50, 50, 50),
    convergence_threshold: float = 0.001,
) -> np.ndarray:
    if not N4_AVAILABLE:
        raise RuntimeError(
            "apply_n4_bias_correction requires SimpleITK, which is not importable in this "
            "environment. Install it (`pip install SimpleITK`) or set "
            "data.apply_n4_bias_correction=false."
        )
    volume = np.asarray(volume, dtype=np.float32)
    mask = np.asarray(mask).astype(np.uint8)

    img = sitk.GetImageFromArray(volume)
    mask_img = sitk.GetImageFromArray(mask)

    shrunk_img = sitk.Shrink(img, [shrink_factor] * img.GetDimension())
    shrunk_mask = sitk.Shrink(mask_img, [shrink_factor] * mask_img.GetDimension())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(max_iterations))
    corrector.SetConvergenceThreshold(convergence_threshold)
    corrector.Execute(shrunk_img, shrunk_mask)

    log_bias_field = corrector.GetLogBiasFieldAsImage(img)
    corrected_img = img / sitk.Exp(log_bias_field)
    corrected = sitk.GetArrayFromImage(corrected_img).astype(np.float32)

    corrected[mask == 0] = 0.0
    return corrected


def _resolve_n4_kwargs(n4_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    sig = inspect.signature(apply_n4_bias_correction)
    resolved = {
        name: param.default
        for name, param in sig.parameters.items()
        if name not in ("volume", "mask") and param.default is not inspect.Parameter.empty
    }
    resolved.update(n4_kwargs or {})
    return resolved


def compute_n4_cache_key(
    patient_id: str,
    session_idx: int,
    modality_idx: int,
    source_path: str,
    n4_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    stat = os.stat(source_path)
    payload = (
        ("patient_id", patient_id),
        ("session_idx", int(session_idx)),
        ("modality_idx", int(modality_idx)),
        ("source_path", os.path.abspath(source_path)),
        ("source_size", stat.st_size),
        ("source_mtime_ns", stat.st_mtime_ns),
        ("n4_kwargs", tuple(sorted(_resolve_n4_kwargs(n4_kwargs).items()))),
    )
    digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:16]
    return f"{patient_id}_s{session_idx}_m{modality_idx}_{digest}"


def n4_correct_with_cache(
    volume: np.ndarray,
    mask: np.ndarray,
    cache_dir: Optional[str],
    cache_key: Any,
    require_cache: bool = False,
    **n4_kwargs,
) -> np.ndarray:
    if cache_dir is None:
        if require_cache:
            raise RuntimeError(
                "[n4_correct_with_cache] require_cache=True but no n4_cache_dir is configured. "
                "N4 correction must be precomputed to a persistent cache before training/eval — "
                "set data.n4_cache_dir and run `python scripts/precompute_n4.py --config <config>` "
                "first."
            )
        return apply_n4_bias_correction(volume, mask, **n4_kwargs)

    missing_msg = (
        f"[n4_correct_with_cache] require_cache=True but no cached N4 output found at "
        f"'{os.path.join(cache_dir, f'{cache_key}.npy')}'. Run "
        "`python scripts/precompute_n4.py --config <config>` first to precompute N4 for every "
        "patient/session in every split — training/eval never computes N4 inline."
    )
    return _disk_array_cache(
        cache_dir, cache_key, lambda: apply_n4_bias_correction(volume, mask, **n4_kwargs),
        require_cache=require_cache, missing_cache_error=missing_msg,
    )


def _disk_array_cache(
    cache_dir: str, cache_key: str, compute_fn, require_cache: bool, missing_cache_error: str
) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_key}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    if require_cache:
        raise RuntimeError(missing_cache_error)

    value = compute_fn()
    tmp_path = cache_path + f".tmp{os.getpid()}.npy"
    np.save(tmp_path, value)
    os.replace(tmp_path, cache_path)
    return value


def compute_normalized_volume_cache_key(
    patient_id: str,
    session_idx: int,
    modality_idx: int,
    source_path: str,
    normalization_policy: str,
    clip_pct: Tuple[float, float],
    stats: Optional["ImageIntensityStats"],
    apply_n4: bool,
    n4_kwargs: Optional[Dict[str, Any]],
) -> str:
    stat = os.stat(source_path)
    stats_dict = stats.to_dict() if stats is not None and stats.is_fit else {}
    payload = (
        ("patient_id", patient_id),
        ("session_idx", int(session_idx)),
        ("modality_idx", int(modality_idx)),
        ("source_path", os.path.abspath(source_path)),
        ("source_size", stat.st_size),
        ("source_mtime_ns", stat.st_mtime_ns),
        ("normalization_policy", normalization_policy),
        ("clip_pct", tuple(clip_pct)),
        ("stats", tuple(sorted(stats_dict.items()))),
        ("apply_n4", bool(apply_n4)),
        ("n4_kwargs", tuple(sorted(_resolve_n4_kwargs(n4_kwargs).items())) if apply_n4 else None),
    )
    digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:16]
    return f"{patient_id}_s{session_idx}_m{modality_idx}_norm_{digest}"


def normalize_volume_with_cache(
    cache_dir: Optional[str],
    cache_key: Optional[str],
    compute_fn,
    require_cache: bool = False,
) -> np.ndarray:
    if cache_dir is None or cache_key is None:
        return compute_fn()

    missing_msg = (
        f"[normalize_volume_with_cache] require_cache=True but no cached normalized volume "
        f"found for key='{cache_key}' under '{cache_dir}'. Run "
        "`python scripts/precompute_normalized_volumes.py --config <config>` first to "
        "precompute normalized volumes for every patient/session in every split — "
        "training/eval never computes this inline when data.normalized_volume_cache_dir is set."
    )
    return _disk_array_cache(cache_dir, cache_key, compute_fn, require_cache, missing_msg)


class WelfordAccumulator:

    def __init__(self):
        self.count = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return
        n_a, mean_a, m2_a = self.count, self._mean, self._m2
        n_b = values.size
        mean_b = float(values.mean())
        m2_b = float(((values - mean_b) ** 2).sum())

        n_ab = n_a + n_b
        delta = mean_b - mean_a
        mean_ab = mean_a + delta * n_b / n_ab
        m2_ab = m2_a + m2_b + delta * delta * n_a * n_b / n_ab

        self.count = n_ab
        self._mean = mean_ab
        self._m2 = m2_ab

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return float(np.sqrt(self._m2 / self.count))


class ReservoirSampler:

    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self._buf = np.empty((0,), dtype=np.float32)
        self._n_seen = 0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return

        if self._buf.shape[0] < self.capacity:
            n_take = min(self.capacity - self._buf.shape[0], values.size)
            self._buf = np.concatenate([self._buf, values[:n_take]])
            self._n_seen += n_take
            values = values[n_take:]
            if values.size == 0:
                return

        m = values.size
        stream_positions = self._n_seen + np.arange(1, m + 1)
        j = self.rng.integers(0, stream_positions)
        replace_mask = j < self.capacity
        self._buf[j[replace_mask]] = values[replace_mask]
        self._n_seen += m

    def samples(self) -> np.ndarray:
        return self._buf


class ImageIntensityStats:

    def __init__(
        self,
        clip_lo: Optional[float] = None,
        clip_hi: Optional[float] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        clip_pct: Tuple[float, float] = (0.5, 99.5),
        eps: float = 1e-6,
        voxel_weighted: bool = True,
        n_foreground_voxels: int = 0,
        n_scans: int = 0,
        seed: int = 0,
        zero_variance: bool = False,
        version: int = IMAGE_INTENSITY_STATS_VERSION,
    ):
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi
        self.mean = mean
        self.std = std
        self.clip_pct = tuple(clip_pct)
        self.eps = eps
        self.voxel_weighted = voxel_weighted
        self.n_foreground_voxels = int(n_foreground_voxels)
        self.n_scans = int(n_scans)
        self.seed = seed
        self.zero_variance = zero_variance
        self.version = version

    @property
    def is_fit(self) -> bool:
        return self.clip_lo is not None and self.mean is not None and self.std is not None

    def transform(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("ImageIntensityStats.transform() called before fit — no frozen stats loaded.")
        volume = np.asarray(volume, dtype=np.float32)
        if mask is None:
            mask = compute_foreground_mask(volume)
        out = np.zeros_like(volume, dtype=np.float32)
        if mask.sum() == 0:
            return out

        clipped = np.clip(volume, self.clip_lo, self.clip_hi)
        std_safe = max(float(self.std), self.eps)
        normalized = (clipped - self.mean) / std_safe
        out[mask] = normalized[mask]
        if not np.isfinite(out).all():
            raise ValueError("ImageIntensityStats.transform produced non-finite output")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "clip_lo": self.clip_lo,
            "clip_hi": self.clip_hi,
            "mean": self.mean,
            "std": self.std,
            "clip_pct": list(self.clip_pct),
            "eps": self.eps,
            "voxel_weighted": self.voxel_weighted,
            "n_foreground_voxels": self.n_foreground_voxels,
            "n_scans": self.n_scans,
            "seed": self.seed,
            "zero_variance": self.zero_variance,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ImageIntensityStats":
        if not d:
            return cls()
        return cls(
            clip_lo=d.get("clip_lo"),
            clip_hi=d.get("clip_hi"),
            mean=d.get("mean"),
            std=d.get("std"),
            clip_pct=tuple(d.get("clip_pct", (0.5, 99.5))),
            eps=d.get("eps", 1e-6),
            voxel_weighted=d.get("voxel_weighted", True),
            n_foreground_voxels=d.get("n_foreground_voxels", 0),
            n_scans=d.get("n_scans", 0),
            seed=d.get("seed", 0),
            zero_variance=d.get("zero_variance", False),
            version=d.get("version", IMAGE_INTENSITY_STATS_VERSION),
        )


def fit_image_intensity_stats(
    volume_iter_factory: Callable[[], Iterable[np.ndarray]],
    clip_pct: Tuple[float, float] = (0.5, 99.5),
    reservoir_size: int = 2_000_000,
    seed: int = 0,
    eps: float = 1e-6,
    apply_n4: bool = False,
    n4_cache_dir: Optional[str] = None,
    n4_kwargs: Optional[Dict[str, Any]] = None,
    n4_cache_keys: Optional[Iterable[Any]] = None,
    require_n4_cache: bool = False,
) -> ImageIntensityStats:
    n4_kwargs = n4_kwargs or {}

    n4_cache_keys_list = list(n4_cache_keys) if n4_cache_keys is not None else None

    def _n4_keys():
        if n4_cache_keys_list is None:
            return iter(lambda: None, object())
        return iter(n4_cache_keys_list)

    reservoir = ReservoirSampler(capacity=reservoir_size, seed=seed)
    n_scans = 0
    n_finite_replaced = 0
    keys_iter = _n4_keys()
    for volume in volume_iter_factory():
        mask = compute_foreground_mask(volume)
        volume, n_bad = sanitize_non_finite(volume)
        n_finite_replaced += n_bad
        if apply_n4:
            volume = n4_correct_with_cache(
                volume, mask, n4_cache_dir, next(keys_iter), require_cache=require_n4_cache, **n4_kwargs
            )
        reservoir.update(volume[mask])
        n_scans += 1

    if n_scans == 0:
        raise ValueError("fit_image_intensity_stats: received zero training volumes to fit on.")

    sample = reservoir.samples()
    if sample.size == 0:
        raise ValueError(
            "fit_image_intensity_stats: zero foreground voxels found across all training volumes "
            "— cannot fit clip percentiles/mean/std."
        )
    clip_lo = float(np.percentile(sample, clip_pct[0]))
    clip_hi = float(np.percentile(sample, clip_pct[1]))
    zero_variance = (clip_hi - clip_lo) < eps

    welford = WelfordAccumulator()
    keys_iter = _n4_keys()
    for volume in volume_iter_factory():
        mask = compute_foreground_mask(volume)
        volume, _ = sanitize_non_finite(volume)
        if apply_n4:
            volume = n4_correct_with_cache(
                volume, mask, n4_cache_dir, next(keys_iter), require_cache=require_n4_cache, **n4_kwargs
            )
        clipped = np.clip(volume, clip_lo, clip_hi)
        welford.update(clipped[mask])

    if n_finite_replaced:
        warnings.warn(
            f"[fit_image_intensity_stats] replaced {n_finite_replaced} non-finite voxel(s) "
            "across the training cohort with 0.0 before fitting statistics.",
            stacklevel=2,
        )

    return ImageIntensityStats(
        clip_lo=clip_lo,
        clip_hi=clip_hi,
        mean=welford.mean,
        std=welford.std,
        clip_pct=clip_pct,
        eps=eps,
        voxel_weighted=True,
        n_foreground_voxels=welford.count,
        n_scans=n_scans,
        seed=seed,
        zero_variance=zero_variance,
    )


def normalize_volume_frozen_stats(
    volume: np.ndarray,
    stats: ImageIntensityStats,
    apply_n4: bool = False,
    n4_cache_dir: Optional[str] = None,
    n4_cache_key: Optional[Any] = None,
    n4_kwargs: Optional[Dict[str, Any]] = None,
    require_n4_cache: bool = False,
) -> np.ndarray:
    mask = compute_foreground_mask(volume)
    volume, _ = sanitize_non_finite(volume)
    if apply_n4:
        volume = n4_correct_with_cache(
            volume, mask, n4_cache_dir, n4_cache_key, require_cache=require_n4_cache, **(n4_kwargs or {})
        )
    return stats.transform(volume, mask=mask)


class VolumeNormalizationCache:

    def __init__(self, maxsize: int = 16):
        self.maxsize = maxsize
        self._cache: "collections.OrderedDict" = collections.OrderedDict()

    def get(self, key, compute_fn):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        value = compute_fn()
        self._cache[key] = value
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return value

    def __len__(self):
        return len(self._cache)


def apply_normalization_policy(
    policy: str,
    *,
    raw_volume_fn=None,
    cache: Optional[VolumeNormalizationCache] = None,
    cache_key=None,
    clip_pct: Tuple[float, float] = (0.5, 99.5),
    eps: float = 1e-6,
    stats: Optional[ImageIntensityStats] = None,
    apply_n4: bool = False,
    n4_cache_dir: Optional[str] = None,
    n4_cache_key: Optional[Any] = None,
    n4_kwargs: Optional[Dict[str, Any]] = None,
    require_n4_cache: bool = False,
    normalized_cache_dir: Optional[str] = None,
    normalized_cache_key: Optional[str] = None,
    require_normalized_cache: bool = False,
) -> np.ndarray:
    if policy not in NORMALIZATION_POLICIES:
        raise ValueError(f"Unknown normalization_policy={policy!r}, expected one of {NORMALIZATION_POLICIES}")

    if raw_volume_fn is None:
        raise ValueError(f"normalization_policy={policy!r} requires raw_volume_fn")

    if policy == "whole_volume":
        def _compute_uncached():
            return normalize_volume_whole(raw_volume_fn(), clip_pct=clip_pct, eps=eps)
    else:
        if stats is None or not stats.is_fit:
            raise ValueError(
                "normalization_policy='frozen_train_stats' requires a fitted `stats` "
                "(ImageIntensityStats) — fit it on TRAIN patients via "
                "dataset.fit_train_image_stats/scripts/fit_image_stats.py first. This "
                "policy never fits/refits statistics itself (see module docstring: "
                "never recompute normalization statistics per session/patient/window)."
            )

        def _compute_uncached():
            return normalize_volume_frozen_stats(
                raw_volume_fn(), stats, apply_n4=apply_n4, n4_cache_dir=n4_cache_dir,
                n4_cache_key=n4_cache_key, n4_kwargs=n4_kwargs, require_n4_cache=require_n4_cache,
            )

    def compute():
        return normalize_volume_with_cache(
            normalized_cache_dir, normalized_cache_key, _compute_uncached,
            require_cache=require_normalized_cache,
        )

    if cache is not None and cache_key is not None:
        return cache.get(cache_key, compute)
    return compute()
