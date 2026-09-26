import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

MODALITY_NAMES = ["T1", "T1c", "FLAIR", "T2"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="/home/alaa.mohamed/MIU2")
    parser.add_argument("--patient_id", default="PatientID_0070")
    parser.add_argument("--session", type=int, default=0,
                         help="Visit/session index into the S axis (default: first visit).")
    parser.add_argument("--slice_start", type=int, default=68)
    parser.add_argument("--slice_end", type=int, default=72,
                         help="Inclusive.")
    parser.add_argument("--out_dir", default="viz")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    img_path = os.path.join(args.data_dir, f"{args.patient_id}_image.npy")
    img = np.load(img_path)
    if img.ndim != 5:
        raise ValueError(f"Expected a 5D (S, C, D, H, W) array at {img_path}, got shape {img.shape}")

    num_sessions, num_modalities, depth = img.shape[:3]
    if not (0 <= args.session < num_sessions):
        raise ValueError(f"--session {args.session} out of range [0, {num_sessions})")
    if num_modalities != len(MODALITY_NAMES):
        raise ValueError(f"Expected {len(MODALITY_NAMES)} modalities, got {num_modalities}")

    slices = list(range(args.slice_start, args.slice_end + 1))
    for z in slices:
        if not (0 <= z < depth):
            raise ValueError(f"Slice index {z} out of range [0, {depth})")

    volume = img[args.session]

    fig, axes = plt.subplots(
        num_modalities, len(slices),
        figsize=(3 * len(slices), 3 * num_modalities),
        squeeze=False,
    )

    for c in range(num_modalities):
        for j, z in enumerate(slices):
            ax = axes[c, j]
            ax.imshow(volume[c, z, :, :], cmap="gray")
            ax.axis("off")
            if c == 0:
                ax.set_title(f"slice {z}", fontsize=10)
            if j == 0:
                ax.text(-0.1, 0.5, MODALITY_NAMES[c], fontsize=12, fontweight="bold",
                        ha="right", va="center", transform=ax.transAxes)

    fig.suptitle(
        f"{args.patient_id} — session {args.session} — slices {args.slice_start}-{args.slice_end}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir,
        f"{args.patient_id}_session{args.session}_slices{args.slice_start}-{args.slice_end}.png",
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
