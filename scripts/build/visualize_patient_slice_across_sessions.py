import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

MODALITY_NAMES = ["T1", "T1c", "FLAIR", "T2"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="/home/alaa.mohamed/MIU2")
    parser.add_argument("--patient_id", default="PatientID_0070")
    parser.add_argument("--slice", type=int, default=70, dest="slice_idx")
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
    if num_modalities != len(MODALITY_NAMES):
        raise ValueError(f"Expected {len(MODALITY_NAMES)} modalities, got {num_modalities}")
    if not (0 <= args.slice_idx < depth):
        raise ValueError(f"--slice {args.slice_idx} out of range [0, {depth})")

    days_path = os.path.join(args.data_dir, f"{args.patient_id}_days.npy")
    days = np.load(days_path) if os.path.exists(days_path) else None

    fig, axes = plt.subplots(
        num_modalities, num_sessions,
        figsize=(3 * num_sessions, 3 * num_modalities),
        squeeze=False,
    )

    for c in range(num_modalities):
        for s in range(num_sessions):
            ax = axes[c, s]
            ax.imshow(img[s, c, args.slice_idx, :, :], cmap="gray")
            ax.axis("off")
            if c == 0:
                title = f"session {s}"
                if days is not None:
                    title += f" (day {int(days[s])})"
                ax.set_title(title, fontsize=10)
            if s == 0:
                ax.text(-0.1, 0.5, MODALITY_NAMES[c], fontsize=12, fontweight="bold",
                        ha="right", va="center", transform=ax.transAxes)

    fig.suptitle(f"{args.patient_id} — slice {args.slice_idx} — all sessions", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.patient_id}_slice{args.slice_idx}_all_sessions.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
