from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False


def _to_sitk_image(volume: np.ndarray) -> "sitk.Image":
    return sitk.GetImageFromArray(np.ascontiguousarray(volume.astype(np.float32)))


def rigid_register_to_fixed(
    fixed: np.ndarray,
    moving: np.ndarray,
    moving_label: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    if not HAS_SITK:
        return moving, moving_label, {"status": "skipped_no_sitk"}

    fixed_img = _to_sitk_image(fixed)
    moving_img = _to_sitk_image(moving)

    try:
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_img, moving_img, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.2)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=1e-4, numberOfIterations=200,
        )
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2, 1, 0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        reg.SetInitialTransform(initial_transform, inPlace=False)

        final_transform = reg.Execute(fixed_img, moving_img)
        metric_value = float(reg.GetMetricValue())
        stop_condition = reg.GetOptimizerStopConditionDescription()
    except Exception as exc:
        return moving, moving_label, {"status": "failed", "error": repr(exc)}

    registered_img = sitk.Resample(
        moving_img, fixed_img, final_transform, sitk.sitkLinear, 0.0, moving_img.GetPixelID(),
    )
    registered = sitk.GetArrayFromImage(registered_img).astype(np.float32)

    registered_label = None
    if moving_label is not None:
        label_img = _to_sitk_image(moving_label)
        registered_label_img = sitk.Resample(
            label_img, fixed_img, final_transform, sitk.sitkNearestNeighbor, 0.0, label_img.GetPixelID(),
        )
        registered_label = sitk.GetArrayFromImage(registered_label_img).astype(moving_label.dtype)

    info = {
        "status": "ok",
        "metric_value": metric_value,
        "stop_condition": stop_condition,
        "transform_params": tuple(final_transform.GetParameters()),
    }
    return registered, registered_label, info
