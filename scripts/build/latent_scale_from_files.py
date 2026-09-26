import argparse
import glob
import os

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dir", default="/home/alaa.mohamed/MIU2/latents")
    parser.add_argument("--n", type=int, default=10, help="number of patients to sample")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.latent_dir, "*_latent.npy")))[: args.n]
    if not paths:
        raise FileNotFoundError(f"No *_latent.npy files found in {args.latent_dir!r}")

    print(f"{'patient_id':<20} {'shape':<22} {'mean':>10} {'std':>10} {'latent_scale':>14}")

    all_means, all_stds = [], []
    for path in paths:
        patient_id = os.path.basename(path).removesuffix("_latent.npy")
        z = np.load(path).astype(np.float32)

        mean = float(z.mean())
        std = float(z.std())
        latent_scale = 1.0 / std if std > 0 else float("nan")

        all_means.append(mean)
        all_stds.append(std)

        print(f"{patient_id:<20} {str(z.shape):<22} {mean:>10.4f} {std:>10.4f} {latent_scale:>14.4f}")

    mean_of_stds = float(np.mean(all_stds))
    mean_of_means = float(np.mean(all_means))
    print("\n--- summary across {} patients ---".format(len(paths)))
    print(f"mean of per-patient means : {mean_of_means:.4f}")
    print(f"mean of per-patient stds  : {mean_of_stds:.4f}")
    print(f"suggested latent_scale    : {1.0 / mean_of_stds:.4f}")
    print(f"suggested latent_mean     : {mean_of_means:.4f}")
    print("(paste BOTH into configs/default.yaml's vae.latent_scale and vae.latent_mean — "
          "latent_scale alone is a pure rescale with no centering)")


if __name__ == "__main__":
    main()
