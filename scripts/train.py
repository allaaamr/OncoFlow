import argparse, yaml, torch, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.trainer import Trainer


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, overrides):
    for o in overrides:
        key, val = o.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() in ("true", "false"):
                    val = val.lower() == "true"
        d[keys[-1]] = val
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--training_mode", choices=["diffusion", "flow_matching"], default=None,
        help="Training objective: 'diffusion' (default — original DDPM/DDIM recipe, "
             "unchanged) or 'flow_matching' (continuous-time conditional flow matching, "
             "ODE sampling). Overrides training.training_mode from --config if set.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)
    if args.training_mode is not None:
        cfg.setdefault("training", {})["training_mode"] = args.training_mode
    cfg["training"].setdefault("training_mode", "diffusion")

    representation = cfg["data"].get("representation", "latent")
    if representation == "voxel":
        shape_label = f"Voxel shape:   {tuple(cfg['data'].get('voxel_shape', (64, 96, 96)))}"
        vae_label = f"VAE:           none (representation=voxel — identity pass-through, see src/models/vae_wrapper.py)"
    else:
        shape_label = f"Volume size:   {cfg['data'].get('volume_size', (160, 240, 240))}"
        vae_label = (
            f"VAE:           {cfg['vae'].get('model_name', 'medvae_4_1_3d')} "
            f"(C_z={cfg['vae']['latent_channels']}, backend={cfg['vae'].get('backend', 'medvae')})"
        )

    print("=" * 60)
    print("3D Whole-Volume Longitudinal MRI Diffusion / Flow Matching")
    print("=" * 60)
    print(f"Device:        {args.device}")
    print(f"Representation:{representation}")
    print(shape_label)
    print(vae_label)
    print(f"UNet channels: {cfg['unet']['base_channels']}")
    print(f"Batch size:    {cfg['data']['batch_size']} (whole volumes)")
    print(f"Seg head:      {'ON' if cfg['seg_head']['enabled'] else 'OFF'}")
    print(f"Training mode: {cfg['training']['training_mode']}")
    print("=" * 60)

    Trainer(cfg, args.device,  resume_path=args.resume).fit()


if __name__ == "__main__":
    main()
