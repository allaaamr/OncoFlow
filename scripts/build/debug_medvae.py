import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

MIU_DIR = "/home/alaa.mohamed/MIU2"
MODALITY_NAMES = ["T1", "T1c", "FLAIR", "T2"]


def find_sample_patient(miu_dir: str) -> str:
    for fname in sorted(os.listdir(miu_dir)):
        if fname.endswith("_image.npy"):
            return fname[: -len("_image.npy")]
    raise FileNotFoundError(f"No *_image.npy files found in {miu_dir}")


def load_patient_volume(miu_dir: str, patient_id: str, session: int, modality: int) -> np.ndarray:
    path = os.path.join(miu_dir, f"{patient_id}_image.npy")
    print(f"=> Loading {path}")
    img = np.load(path, mmap_mode="r")
    print(f"   full array shape (sessions, modalities, D, H, W) = {img.shape}, dtype={img.dtype}")

    n_sessions, n_modalities = img.shape[0], img.shape[1]
    if session >= n_sessions:
        raise ValueError(f"Patient {patient_id} only has {n_sessions} session(s); got --session {session}")
    if modality >= n_modalities:
        raise ValueError(f"Patient {patient_id} only has {n_modalities} modalities; got --modality {modality}")

    volume = np.array(img[session, modality]).astype(np.float32)
    print(f"   selected volume shape={volume.shape}, min={volume.min():.4f}, max={volume.max():.4f}")
    return volume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miu_dir", type=str, default=MIU_DIR)
    parser.add_argument("--patient", type=str, default=None, help="Patient id, e.g. PatientID_0006. Defaults to first patient found.")
    parser.add_argument("--session", type=int, default=0, help="Session/timepoint index.")
    parser.add_argument("--modality", type=int, default=0, help="Modality channel index (0=T1, 1=T1c, 2=FLAIR, 3=T2).")
    parser.add_argument("--model_name", type=str, default="medvae_4_1_3d",
                         help="MedVAE model id (medvae_4x1.yaml / vae_4x_1c_3D.ckpt = medvae_4_1_3d).")
    parser.add_argument("--modality_type", type=str, default="mri", choices=["mri", "ct"])
    parser.add_argument("--roi_size", type=int, default=160)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="scripts/medvae_debug_out")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"=> Device: {device}")

    patient_id = args.patient or find_sample_patient(args.miu_dir)
    print(f"=> Using patient: {patient_id} (session={args.session}, modality={MODALITY_NAMES[args.modality]})")
    volume = load_patient_volume(args.miu_dir, patient_id, args.session, args.modality)

    volume = (volume - 0.5) / 0.5

    img = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
    img = img.to(device)
    print(f"=> Model input tensor shape: {tuple(img.shape)}")

    from medvae import MVAE

    print(f"=> Building MedVAE model '{args.model_name}' (modality={args.modality_type}, roi_size={args.roi_size})")
    model = MVAE(args.model_name, args.modality_type, args.roi_size).to(device)
    model.requires_grad_(False)
    model.eval()

    with torch.no_grad():
        print("=> Encoding (compress) ...")
        latent = model.encode(img)
        print(f"   latent shape: {tuple(latent.shape)}")

        print("=> Decoding (decompress) ...")
        recon = model.decode(latent)
        print(f"   reconstruction shape: {tuple(recon.shape)}")

    compression_ratio = img.numel() / latent.numel()
    print(f"=> Voxel compression ratio (orig/latent): {compression_ratio:.2f}x")

    orig_np = img.squeeze().float().cpu().numpy()
    recon_np = recon.squeeze().float().cpu().numpy()
    latent_np = latent.squeeze().float().cpu().numpy()

    print(f"=> orig  : shape={orig_np.shape}, min={orig_np.min():.4f}, max={orig_np.max():.4f}")
    print(f"=> recon : shape={recon_np.shape}, min={recon_np.min():.4f}, max={recon_np.max():.4f}")

    common_shape = tuple(min(o, r) for o, r in zip(orig_np.shape, recon_np.shape))
    orig_crop = orig_np[: common_shape[0], : common_shape[1], : common_shape[2]]
    recon_crop = recon_np[: common_shape[0], : common_shape[1], : common_shape[2]]
    mse = float(np.mean((orig_crop - recon_crop) ** 2))
    print(f"=> Reconstruction MSE (normalized [-1,1] space, on common shape {common_shape}): {mse:.6f}")

    def to_display(x_slice):
        x_slice = x_slice * 0.5 + 0.5
        return np.clip(x_slice, 0, 1)

    z_orig = orig_np.shape[0] // 2
    z_recon = min(z_orig, recon_np.shape[0] - 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    axes[0].imshow(to_display(orig_np[z_orig]), cmap="gray")
    axes[0].set_title(f"Original\n{patient_id} sess={args.session} {MODALITY_NAMES[args.modality]}\nslice z={z_orig} {orig_np.shape}")
    axes[0].axis("off")

    mid_latent_c = latent_np.shape[0] // 2 if latent_np.ndim == 4 else None
    if latent_np.ndim == 4:
        latent_slice = latent_np[mid_latent_c, latent_np.shape[1] // 2]
    else:
        latent_slice = latent_np[latent_np.shape[0] // 2]
    axes[1].imshow(latent_slice, cmap="viridis")
    axes[1].set_title(f"Compressed latent\nshape={latent_np.shape}\nslice z={latent_np.shape[-3] // 2}")
    axes[1].axis("off")

    axes[2].imshow(to_display(recon_np[z_recon]), cmap="gray")
    axes[2].set_title(f"Decoded (reconstructed)\nslice z={z_recon} {recon_np.shape}\nMSE={mse:.5f}")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(
        args.out_dir,
        f"{patient_id}_sess{args.session}_{MODALITY_NAMES[args.modality]}_medvae_4x1_3d.png",
    )
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"=> Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
