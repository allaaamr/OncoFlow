import argparse
import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.TaGeDiff import TaGeDiff
from src.data import preprocessing as preprocessing_utils
from src.data.dataset import n4_kwargs_from_dc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[Warning] matplotlib not found — PNGs will be saved as raw numpy grids via PIL.")
    from PIL import Image

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

FLAIR_INDEX = 2


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list) -> dict:
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


def read_checkpoint(path: str, device: str) -> dict:
    return torch.load(path, map_location=device)


def resolve_training_mode(cfg: dict, ckpt: dict, checkpoint_path: str) -> dict:
    ckpt_mode = ckpt.get("training_mode")
    cfg.setdefault("training", {})
    cfg_mode = cfg["training"].get("training_mode", "diffusion")

    if ckpt_mode is None:
        print(
            f"[InferViz] WARNING: checkpoint '{checkpoint_path}' has no saved training_mode "
            f"(pre-schema-v2). Trusting --config's training.training_mode={cfg_mode!r}."
        )
        return cfg

    if ckpt_mode != cfg_mode:
        print(
            f"[InferViz] CORRECTING training.training_mode: --config says {cfg_mode!r} but "
            f"checkpoint '{checkpoint_path}' was trained with {ckpt_mode!r}. Using the "
            "checkpoint's value."
        )
    cfg["training"]["training_mode"] = ckpt_mode

    objective_cfg = ckpt.get("objective_cfg")
    if objective_cfg:
        section = "flow_matching" if ckpt_mode == "flow_matching" else "diffusion"
        cfg.setdefault(section, {})
        cfg[section] = {**cfg[section], **objective_cfg}

    return cfg


def resolve_representation(cfg: dict, ckpt: dict, checkpoint_path: str) -> dict:
    ckpt_repr = ckpt.get("representation")
    cfg.setdefault("data", {})
    cfg_repr = cfg["data"].get("representation", "latent")

    if ckpt_repr is None:
        print(
            f"[InferViz] WARNING: checkpoint '{checkpoint_path}' has no saved representation "
            f"(pre-voxel-mode). Trusting --config's data.representation={cfg_repr!r}."
        )
        return cfg

    if ckpt_repr != cfg_repr:
        print(
            f"[InferViz] CORRECTING data.representation: --config says {cfg_repr!r} but "
            f"checkpoint '{checkpoint_path}' was trained with {ckpt_repr!r}. Using the "
            "checkpoint's value."
        )
    cfg["data"]["representation"] = ckpt_repr

    ckpt_voxel_shape = ckpt.get("voxel_shape")
    if ckpt_repr == "voxel" and ckpt_voxel_shape:
        cfg["data"]["voxel_shape"] = list(ckpt_voxel_shape)

    return cfg


def resolve_condition_on_treatment(cfg: dict, ckpt: dict, checkpoint_path: str) -> dict:
    ckpt_cot = ckpt.get("condition_on_treatment")
    cfg.setdefault("context", {})
    cfg_cot = cfg["context"].get("condition_on_treatment", True)

    if ckpt_cot is None:
        print(
            f"[InferViz] WARNING: checkpoint '{checkpoint_path}' has no saved "
            f"condition_on_treatment (pre-this-feature). Trusting --config's "
            f"context.condition_on_treatment={cfg_cot!r}."
        )
        return cfg

    if ckpt_cot != cfg_cot:
        print(
            f"[InferViz] CORRECTING context.condition_on_treatment: --config says {cfg_cot!r} "
            f"but checkpoint '{checkpoint_path}' was trained with {ckpt_cot!r}. Using the "
            "checkpoint's value."
        )
    cfg["context"]["condition_on_treatment"] = ckpt_cot

    return cfg


def resolve_temporal_aggregator_variant(cfg: dict, ckpt: dict, checkpoint_path: str) -> dict:
    ckpt_variant = ckpt.get("temporal_aggregator_variant")
    cfg.setdefault("temporal_aggregator", {})
    cfg_variant = cfg["temporal_aggregator"].get("variant", "temporal_spatial")

    if ckpt_variant is None:
        ckpt_variant = "patch_temporal_spatial"
        print(
            f"[InferViz] WARNING: checkpoint '{checkpoint_path}' has no saved "
            f"temporal_aggregator_variant (pre-this-feature) — every such checkpoint used "
            f"'patch_temporal_spatial'. Trusting that, not --config's temporal_aggregator."
            f"variant={cfg_variant!r}."
        )

    if ckpt_variant != cfg_variant:
        print(
            f"[InferViz] CORRECTING temporal_aggregator.variant: --config says {cfg_variant!r} "
            f"but checkpoint '{checkpoint_path}' was trained with {ckpt_variant!r}. Using the "
            "checkpoint's value."
        )
    cfg["temporal_aggregator"]["variant"] = ckpt_variant

    return cfg


def load_checkpoint(model, ckpt: dict, use_ema: bool, device: str):
    schema_version = ckpt.get("schema_version", 1)
    if schema_version < 3:
        raise ValueError(
            f"Checkpoint predates schema v3 (found v{schema_version}) — it was trained under "
            "the OLD 2D pipeline, architecturally incompatible with this 3D pipeline. Cannot "
            "run inference here — re-train under the 3D pipeline first."
        )
    weights = ckpt["ema"] if (use_ema and ckpt.get("ema")) else ckpt["model"]
    current = model.state_dict()
    loaded = skipped = 0
    for k, v in weights.items():
        if k in current:
            current[k].copy_(v)
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(current)
    tag = "EMA" if (use_ema and ckpt.get("ema")) else "model"
    print(f"[InferViz] Loaded {tag} weights — epoch={ckpt.get('epoch','?')}  "
          f"loaded={loaded}  skipped={skipped}")
    return {
        "epoch": ckpt.get("epoch", None),
        "geno_preproc": preprocessing_utils.GenomicsPreprocessor.from_dict(ckpt.get("geno_preproc")),
        "image_stats": (
            {int(k): preprocessing_utils.ImageIntensityStats.from_dict(v) for k, v in ckpt["image_stats"].items()}
            if ckpt.get("image_stats") else None
        ),
        "normalization_policy": ckpt.get("normalization_policy", "whole_volume"),
        "normalization_clip_pct": tuple(ckpt.get("normalization_clip_pct", (0.5, 99.5))),
        "volume_size": tuple(ckpt.get("volume_size", (160, 240, 240))),
        "representation": ckpt.get("representation", "latent"),
        "voxel_shape": tuple(ckpt["voxel_shape"]) if ckpt.get("voxel_shape") else None,
        "condition_on_treatment": bool(ckpt.get("condition_on_treatment", True)),
        "temporal_aggregator_variant": ckpt.get("temporal_aggregator_variant", "patch_temporal_spatial"),
    }


def load_patient(data_dir: str, patient_id: str, cfg: dict, geno_preproc=None) -> dict:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    def load(suffix: str):
        file_path = data_path / f"{patient_id}_{suffix}.npy"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")
        return np.load(str(file_path), mmap_mode="r")

    data = {
        "image": load("image"),
        "image_path": str(data_path / f"{patient_id}_image.npy"),
        "label": load("label"),
        "treatment": load("treatment"),
        "days": load("days"),
    }
    if data["image"].ndim != 5:
        raise ValueError(
            f"[InferViz] Patient '{patient_id}': expected image array (S,C,D,H,W), "
            f"got shape {data['image'].shape}"
        )

    geno_path = data_path / f"{patient_id}_geno.npy"
    real_geno = cfg["data"].get("real_geno", False)
    genomic_dim = cfg["data"].get("genomic_dim")

    if real_geno and geno_path.exists():
        raw_geno = np.asarray(np.load(str(geno_path)), dtype=np.float32).reshape(-1)
        if geno_preproc is not None and geno_preproc.is_fit:
            data["geno"] = geno_preproc.transform(raw_geno)
        else:
            data["geno"] = raw_geno
        data["geno_mask"] = True
    else:
        if genomic_dim is None:
            raise ValueError("Missing cfg['data']['genomic_dim']; needed for dummy genomics.")
        data["geno"] = np.zeros((genomic_dim,), dtype=np.float32)
        data["geno_mask"] = False

    S = data["image"].shape[0]
    print(
        f"[InferViz] Patient '{patient_id}': {S} session(s), "
        f"image shape {data['image'].shape}, label shape {data['label'].shape}, "
        f"geno_mask={data['geno_mask']}"
    )
    return data


def build_batch(patient_data: dict, t_idx: int, volume_size: tuple, device: str,
                 normalization_policy: str, normalization_clip_pct: tuple,
                 norm_cache: preprocessing_utils.VolumeNormalizationCache,
                 representation: str = "latent", image_stats: dict = None,
                 apply_n4: bool = False, n4_cache_dir: str = None, n4_kwargs: dict = None,
                 patient_id: str = None) -> dict:
    images = patient_data["image"]
    image_path = patient_data.get("image_path")
    labels = patient_data["label"]
    treatments = np.asarray(patient_data["treatment"], dtype=np.int64)
    days = np.asarray(patient_data["days"], dtype=np.float32)
    geno = patient_data["geno"]
    geno_mask = bool(patient_data["geno_mask"])

    def normalize_and_resize(session_idx):
        n4_cache_key = (
            preprocessing_utils.compute_n4_cache_key(patient_id, session_idx, FLAIR_INDEX, image_path, n4_kwargs)
            if apply_n4 and patient_id is not None and image_path is not None else None
        )
        vol = preprocessing_utils.apply_normalization_policy(
            normalization_policy,
            raw_volume_fn=lambda: np.asarray(images[session_idx, FLAIR_INDEX, :, :, :]),
            cache=norm_cache, cache_key=session_idx, clip_pct=normalization_clip_pct,
            stats=image_stats.get(FLAIR_INDEX) if image_stats else None,
            apply_n4=apply_n4, n4_cache_dir=n4_cache_dir,
            n4_cache_key=n4_cache_key, n4_kwargs=n4_kwargs, require_n4_cache=True,
        )
        return preprocessing_utils.to_model_shape_3d(vol, volume_size, representation=representation, is_label=False)

    target_vol = normalize_and_resize(t_idx)
    target_lbl = preprocessing_utils.to_model_shape_3d(
        np.asarray(labels[t_idx]), volume_size, representation=representation, is_label=True
    )

    ctx_idxs = list(range(0, t_idx))
    n_ctx = len(ctx_idxs)

    target_day_abs = float(days[t_idx])
    last_input_day_abs = float(days[ctx_idxs[-1]]) if n_ctx > 0 else target_day_abs

    ctx_vols, ctx_days_rel, ctx_treats = [], [], []
    for ct in ctx_idxs:
        v = normalize_and_resize(ct)
        ctx_vols.append(torch.from_numpy(v).float().unsqueeze(0))
        ctx_days_rel.append(float(days[ct] - target_day_abs))
        ctx_treats.append(int(treatments[ct]))

    target_day_rel = np.float32(target_day_abs - last_input_day_abs)
    target_treat = int(treatments[t_idx])

    D, H, W = volume_size
    input_images = (
        torch.stack(ctx_vols, dim=0).unsqueeze(0).to(device)
        if n_ctx > 0
        else torch.zeros((1, 0, 1, D, H, W), device=device)
    )

    batch = {
        "input_images": input_images,
        "input_days": (
            torch.tensor([ctx_days_rel], dtype=torch.float32, device=device)
            if n_ctx > 0 else torch.zeros((1, 0), dtype=torch.float32, device=device)
        ),
        "input_treatments": (
            torch.tensor([ctx_treats], dtype=torch.long, device=device)
            if n_ctx > 0 else torch.zeros((1, 0), dtype=torch.long, device=device)
        ),
        "input_mask": torch.ones((1, n_ctx), dtype=torch.bool, device=device),
        "target_day": torch.tensor([target_day_rel], dtype=torch.float32, device=device),
        "target_treatment": torch.tensor([target_treat], dtype=torch.long, device=device),
        "geno": torch.tensor([geno], dtype=torch.float32, device=device),
        "geno_mask": torch.tensor([geno_mask], dtype=torch.bool, device=device),
    }
    return batch, target_vol[None, ...], target_lbl


@torch.no_grad()
def generate_volume(model: TaGeDiff, batch: dict, num_steps: int, device: str, solver: str = None,
                     normalization_policy: str = "whole_volume") -> torch.Tensor:
    if model.training_mode == "diffusion":
        sampler_kwargs = dict(ddim_steps=num_steps, ddim_eta=0.0)
    else:
        sampler_kwargs = dict(fm_steps=num_steps, fm_solver=solver or model.fm_cfg.get("solver", "euler"))

    with torch.cuda.amp.autocast(enabled=str(device).startswith("cuda")):
        out = model.generate(
            input_images=batch["input_images"], input_days=batch["input_days"],
            input_treatments=batch["input_treatments"], input_mask=batch["input_mask"],
            genomics=batch["geno"], geno_mask=batch["geno_mask"],
            target_day=batch["target_day"], target_treatment=batch["target_treatment"],
            **sampler_kwargs,
        )
    gen = out["generated_mri"].float()
    if normalization_policy == "whole_volume":
        gen = gen.clamp(0.0, 1.0)
    return gen.cpu()


def _to_display(img_2d: np.ndarray, fg_mask: np.ndarray = None) -> np.ndarray:
    img_2d = np.asarray(img_2d, dtype=np.float32)
    fg = fg_mask if fg_mask is not None else (img_2d != 0.0)
    out = np.zeros_like(img_2d)
    if not fg.any():
        return out
    lo, hi = float(img_2d[fg].min()), float(img_2d[fg].max())
    if hi - lo < 1e-6:
        out[fg] = 0.5
        return out
    out[fg] = (img_2d[fg] - lo) / (hi - lo)
    return out


def save_volume_report(gt_vol: np.ndarray, gen_vol: np.ndarray, gt_lbl_vol: np.ndarray,
                        save_path: str, title: str = ""):
    D, H, W = gt_vol.shape
    zd, zh, zw = D // 2, H // 2, W // 2

    def views(vol):
        return [vol[zd], vol[:, zh, :], vol[:, :, zw]]

    gt_views_raw = views(gt_vol)
    gen_views_raw = views(gen_vol)
    bg_masks = [v != 0.0 for v in gt_views_raw]
    gt_views = [_to_display(v, fg_mask=m) for v, m in zip(gt_views_raw, bg_masks)]
    gen_views = [_to_display(v, fg_mask=m) for v, m in zip(gen_views_raw, bg_masks)]
    lbl_views = views(gt_lbl_vol)
    names = ["Axial", "Coronal", "Sagittal"]

    if HAS_MPL:
        fig = plt.figure(figsize=(10, 9))
        fig.suptitle(title, fontsize=10)
        gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.05, hspace=0.15)

        for c in range(3):
            ax = fig.add_subplot(gs[0, c])
            ax.set_title(f"GT {names[c]}")
            ax.imshow(gt_views[c], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

        for c in range(3):
            ax = fig.add_subplot(gs[1, c])
            ax.set_title(f"Generated {names[c]}")
            ax.imshow(gen_views[c], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

        for c in range(3):
            ax = fig.add_subplot(gs[2, c])
            ax.set_title(f"GT + Tumor {names[c]}")
            ax.imshow(gt_views[c], cmap="gray", vmin=0, vmax=1)
            lbl_2d = lbl_views[c]
            if lbl_2d.any():
                ax.imshow(np.ma.masked_where(lbl_2d == 0, lbl_2d), cmap="autumn", alpha=0.5, vmin=0, vmax=lbl_2d.max() or 1)
            ax.axis("off")

        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        from PIL import Image as PILImage
        def to_uint8(a):
            return (a * 255).clip(0, 255).astype(np.uint8)
        row = np.concatenate([to_uint8(gt_views[0]), to_uint8(gen_views[0])], axis=1)
        PILImage.fromarray(row).save(save_path)


def main():
    parser = argparse.ArgumentParser(
        description="Whole-volume 3D inference + visualization for a single patient."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patient-id", required=True, help="Patient folder name inside data_dir")
    parser.add_argument("--timepoint", type=int, default=None,
                         help="Generate only this target session index (default: every "
                              "follow-up session, i.e. all t_idx in [1, num_sessions)).")
    parser.add_argument("--save-dir", default="./viz", help="Directory to write outputs (default: ./viz)")
    parser.add_argument("--steps", type=int, default=100,
                         help="DDIM steps (diffusion) or ODE integration steps (flow_matching)")
    parser.add_argument("--solver", default=None, choices=[None, "euler", "heun"])
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights from checkpoint")
    parser.add_argument("--save-volumes", action="store_true", help="Also save generated/GT volumes as .npy")
    parser.add_argument("--save-nifti", action="store_true", help="Also save .nii.gz (identity affine — MIU data carries no real affine)")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    out_dir = Path(args.save_dir) / args.patient_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("3D Whole-Volume Longitudinal MRI — Inference & Visualization")
    print("=" * 60)
    print(f"  Device      : {args.device}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Patient ID  : {args.patient_id}")
    print(f"  Steps       : {args.steps}")
    print(f"  EMA         : {args.use_ema}")
    print(f"  Output dir  : {out_dir}")
    print("=" * 60)

    raw_ckpt = read_checkpoint(args.checkpoint, device=args.device)
    cfg = resolve_training_mode(cfg, raw_ckpt, args.checkpoint)
    cfg = resolve_representation(cfg, raw_ckpt, args.checkpoint)
    cfg = resolve_condition_on_treatment(cfg, raw_ckpt, args.checkpoint)
    cfg = resolve_temporal_aggregator_variant(cfg, raw_ckpt, args.checkpoint)

    model = TaGeDiff(cfg).to(args.device)
    model.eval()
    ckpt_info = load_checkpoint(model, raw_ckpt, use_ema=args.use_ema, device=args.device)
    print(f"  Training mode: {model.training_mode}")
    print(f"  Representation: {model.representation}")

    patient_data = load_patient(cfg["data"]["data_dir"], args.patient_id, cfg, geno_preproc=ckpt_info["geno_preproc"])
    if model.representation == "voxel":
        volume_size = tuple(cfg["data"].get("voxel_shape", ckpt_info["voxel_shape"] or (64, 96, 96)))
    else:
        volume_size = tuple(cfg["data"].get("volume_size", ckpt_info["volume_size"]))
    normalization_policy = ckpt_info["normalization_policy"]
    normalization_clip_pct = ckpt_info["normalization_clip_pct"]
    image_stats = ckpt_info["image_stats"]
    apply_n4 = bool(cfg["data"].get("apply_n4_bias_correction", False))
    n4_cache_dir = cfg["data"].get("n4_cache_dir")
    n4_kwargs = n4_kwargs_from_dc(cfg["data"])

    num_sessions = int(patient_data["label"].shape[0])
    if args.timepoint is not None:
        target_timepoints = [args.timepoint]
    else:
        target_timepoints = list(range(1, num_sessions))

    if not target_timepoints:
        print("[InferViz] No follow-up timepoints to generate for this patient (needs >=2 sessions). Exiting.")
        return

    norm_cache = preprocessing_utils.VolumeNormalizationCache(maxsize=32)
    for t_idx in tqdm(target_timepoints, desc="Generating"):
        tag = f"t{t_idx:02d}"
        label = f"patient={args.patient_id}  timepoint={t_idx}  steps={args.steps}"

        batch, gt_vol_np, gt_lbl_np = build_batch(
            patient_data, t_idx, volume_size, args.device,
            normalization_policy, normalization_clip_pct, norm_cache,
            representation=model.representation,
            image_stats=image_stats, apply_n4=apply_n4, n4_cache_dir=n4_cache_dir, n4_kwargs=n4_kwargs,
            patient_id=args.patient_id,
        )

        gen_tensor = generate_volume(model, batch, num_steps=args.steps, device=args.device, solver=args.solver,
                                      normalization_policy=normalization_policy)
        gen_vol_np = gen_tensor.squeeze(0).squeeze(0).numpy()
        gt_vol_2d = gt_vol_np[0]

        common = tuple(min(a, b) for a, b in zip(gt_vol_2d.shape, gen_vol_np.shape))
        gt_c = gt_vol_2d[:common[0], :common[1], :common[2]]
        gen_c = gen_vol_np[:common[0], :common[1], :common[2]]
        lbl_c = gt_lbl_np[:common[0], :common[1], :common[2]]

        png_path = out_dir / f"{tag}.png"
        save_volume_report(gt_c, gen_c, lbl_c, save_path=str(png_path), title=label)

        if args.save_volumes:
            np.save(out_dir / f"{tag}_gt.npy", gt_c)
            np.save(out_dir / f"{tag}_pred.npy", gen_c)
            if args.save_nifti:
                if not HAS_NIBABEL:
                    print("[InferViz] --save-nifti requested but nibabel is not installed; skipping.")
                else:
                    affine = np.eye(4)
                    nib.save(nib.Nifti1Image(gt_c, affine), str(out_dir / f"{tag}_gt.nii.gz"))
                    nib.save(nib.Nifti1Image(gen_c, affine), str(out_dir / f"{tag}_pred.nii.gz"))

    print(f"\n[InferViz] Done. {len(target_timepoints)} timepoint(s) saved to: {out_dir}")


if __name__ == "__main__":
    main()
