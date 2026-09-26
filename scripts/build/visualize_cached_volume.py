import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", required=True, help="Path to a cached (D, H, W) .npy volume.")
    parser.add_argument("--axial", type=int, default=None, help="Axis-0 slice index (default: center).")
    parser.add_argument("--coronal", type=int, default=None, help="Axis-1 slice index (default: center).")
    parser.add_argument("--sagittal", type=int, default=None, help="Axis-2 slice index (default: center).")
    parser.add_argument("--out_dir", default="viz")
    parser.add_argument("--cmap", default="gray")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.path):
        raise FileNotFoundError(args.path)
    vol = np.load(args.path)
    if vol.ndim != 3:
        raise ValueError(
            f"Expected a 3D (D, H, W) cached volume at {args.path}, got shape {vol.shape} "
            f"(ndim={vol.ndim}) — this script visualizes single-volume n4_cache/"
            "normalized_volume_cache_dir entries, not a full (S, C, D, H, W) patient array "
            "(use scripts/visualize_patient_slice_across_sessions.py for that)."
        )
    depth, height, width = vol.shape

    axial = args.axial if args.axial is not None else depth // 2
    coronal = args.coronal if args.coronal is not None else height // 2
    sagittal = args.sagittal if args.sagittal is not None else width // 2
    for flag, idx, bound in [("--axial", axial, depth), ("--coronal", coronal, height), ("--sagittal", sagittal, width)]:
        if not (0 <= idx < bound):
            raise ValueError(f"{flag}={idx} out of range [0, {bound})")

    slices = [
        (f"Axial (axis 0 = {axial}/{depth})", vol[axial, :, :]),
        (f"Coronal (axis 1 = {coronal}/{height})", vol[:, coronal, :]),
        (f"Sagittal (axis 2 = {sagittal}/{width})", vol[:, :, sagittal]),
    ]

    cmap = matplotlib.colormaps[args.cmap].copy()
    cmap.set_bad(color="black")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, sl) in zip(axes, slices):
        masked = np.ma.masked_equal(sl, 0.0)
        ax.imshow(masked, cmap=cmap, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fname = os.path.splitext(os.path.basename(args.path))[0]
    fig.suptitle(
        f"{fname}\nshape={vol.shape}  min={vol.min():.3f}  max={vol.max():.3f}  mean={vol.mean():.3f}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{fname}_slices.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
