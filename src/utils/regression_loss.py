import math

import torch
import torch.nn.functional as F


CURRICULUM_SCHEDULES = ("linear", "cosine")


def standard_regression_loss(pred: torch.Tensor, target: torch.Tensor):
    mse = torch.mean((pred - target) ** 2)
    return mse, mse


def tumor_weighted_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    label: torch.Tensor,
    kernel_shape=(5, 11, 11),
    kernel_divisor: float = 10.0,
):
    label = label.float()
    if label.ndim == 4:
        label = label.unsqueeze(1)

    label_latent = F.interpolate(
        label,
        size=pred.shape[-3:],
        mode="nearest",
    )

    loss_weights = torch.sum(label_latent, dim=1, keepdim=True)
    loss_weights = loss_weights * torch.exp(-loss_weights)

    dilation_filters = torch.ones(1, 1, *kernel_shape) / kernel_divisor

    loss_weights = (
        F.conv3d(
            loss_weights,
            dilation_filters.to(loss_weights.device),
            padding="same",
        ) + 1.0
    )

    loss = torch.mean(loss_weights * (pred - target) ** 2)

    mse = torch.mean((pred - target) ** 2)

    return loss, mse


def compute_curriculum_progress(
    current_epoch,
    start_epoch: float = 0.0,
    ramp_epochs: float = 1.0,
) -> float:
    if current_epoch is None:
        raise ValueError(
            "compute_curriculum_progress requires the restored absolute training "
            "progress (current_epoch) — got None."
        )
    if ramp_epochs <= 0:
        raise ValueError(f"ramp_epochs must be > 0, got {ramp_epochs}")

    progress = (current_epoch - start_epoch) / ramp_epochs
    return min(max(progress, 0.0), 1.0)


def compute_curriculum_alpha(
    current_epoch,
    enabled: bool,
    schedule: str = "cosine",
    start_epoch: float = 0.0,
    ramp_epochs: float = 1.0,
) -> float:
    if not enabled:
        return 1.0

    if current_epoch is None:
        raise ValueError(
            "normalized_tumor_weighted's curriculum.enabled=True requires the restored "
            "absolute training progress (current_epoch, i.e. completed epochs) to be "
            "provided explicitly — got None."
        )
    if ramp_epochs <= 0:
        raise ValueError(
            "normalized_tumor_weighted's curriculum.ramp_epochs must be > 0 when "
            f"curriculum.enabled=True, got {ramp_epochs}"
        )
    if schedule not in CURRICULUM_SCHEDULES:
        raise ValueError(
            f"Unknown normalized_tumor_weighted.curriculum.schedule={schedule!r}, "
            f"expected one of {CURRICULUM_SCHEDULES}"
        )

    progress = compute_curriculum_progress(
        current_epoch, start_epoch=start_epoch, ramp_epochs=ramp_epochs,
    )

    if schedule == "linear":
        return progress
    else:
        return 0.5 * (1.0 - math.cos(math.pi * progress))


def normalized_tumor_weighted_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    label: torch.Tensor,
    curriculum_alpha: float,
    kernel_shape=(5, 11, 11),
    kernel_divisor: float = 10.0,
    normalize_per_sample: bool = True,
    eps: float = 1e-8,
    diagnostics: dict = None,
):
    if not (0.0 <= curriculum_alpha <= 1.0):
        raise ValueError(f"curriculum_alpha must be in [0.0, 1.0], got {curriculum_alpha}")
    if kernel_divisor <= 0:
        raise ValueError(f"kernel_divisor must be > 0, got {kernel_divisor}")
    if len(kernel_shape) != 3 or any(int(k) <= 0 for k in kernel_shape):
        raise ValueError(
            f"kernel_shape must be a (kD, kH, kW) triple of positive ints, got {kernel_shape}"
        )

    label = label.float()
    if label.ndim == 4:
        label = label.unsqueeze(1)

    label_latent = F.interpolate(
        label,
        size=pred.shape[-3:],
        mode="nearest",
    )

    loss_weights = torch.sum(label_latent, dim=1, keepdim=True)
    loss_weights = loss_weights * torch.exp(-loss_weights)

    dilation_filter = loss_weights.new_ones((1, 1, *kernel_shape))
    dilation_filter = dilation_filter / kernel_divisor

    tumor_boost = F.conv3d(
        loss_weights,
        dilation_filter,
        padding="same",
    )

    raw_weights = 1.0 + curriculum_alpha * tumor_boost

    if normalize_per_sample:
        mean_weights = raw_weights.mean(dim=(2, 3, 4), keepdim=True)
        weights = raw_weights / mean_weights.clamp_min(eps)
    else:
        weights = raw_weights

    squared_error = (pred - target).pow(2)
    loss = torch.mean(weights * squared_error)
    mse = torch.mean(squared_error)

    if diagnostics is not None:
        with torch.no_grad():
            diagnostics["curriculum_alpha"] = float(curriculum_alpha)
            diagnostics["normalized_weighted_loss"] = loss.detach().item()
            diagnostics["unweighted_mse"] = mse.detach().item()
            diagnostics["raw_weight_mean"] = raw_weights.detach().mean().item()
            diagnostics["raw_weight_max"] = raw_weights.detach().max().item()
            diagnostics["normalized_weight_mean"] = weights.detach().mean().item()
            diagnostics["normalized_weight_max"] = weights.detach().max().item()

    return loss, mse


def compute_flow_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    tumor_mask: torch.Tensor = None,
    kernel_shape=(5, 11, 11),
    kernel_divisor: float = 10.0,
    curriculum_alpha: float = None,
    normalize_per_sample: bool = True,
    diagnostics: dict = None,
):
    if loss_type == "standard":
        return standard_regression_loss(prediction, target)
    elif loss_type == "tumor_weighted":
        if tumor_mask is None:
            raise ValueError(
                "flow_loss='tumor_weighted' requires tumor_mask (the target-visit "
                "segmentation, batch['target_label']) — got None."
            )
        return tumor_weighted_regression_loss(
            prediction, target, tumor_mask,
            kernel_shape=kernel_shape, kernel_divisor=kernel_divisor,
        )
    elif loss_type == "normalized_tumor_weighted":
        if tumor_mask is None:
            raise ValueError(
                "flow_loss='normalized_tumor_weighted' requires tumor_mask (the "
                "target-visit segmentation, batch['target_label']) — got None."
            )
        if curriculum_alpha is None:
            raise ValueError(
                "flow_loss='normalized_tumor_weighted' requires curriculum_alpha "
                "(see compute_curriculum_alpha) — got None."
            )
        return normalized_tumor_weighted_regression_loss(
            prediction, target, tumor_mask,
            curriculum_alpha=curriculum_alpha,
            kernel_shape=kernel_shape,
            kernel_divisor=kernel_divisor,
            normalize_per_sample=normalize_per_sample,
            diagnostics=diagnostics,
        )
    else:
        raise ValueError(
            f"Unknown flow_loss={loss_type!r}, expected 'standard', 'tumor_weighted', "
            "or 'normalized_tumor_weighted'"
        )
