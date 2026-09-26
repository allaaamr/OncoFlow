#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from full_test import (
    load_config, read_checkpoint, resolve_training_mode, resolve_representation,
    resolve_condition_on_treatment, resolve_temporal_aggregator_variant,
    load_checkpoint, read_patient_ids_from_split, load_patient, build_batch,
    set_seed,
)
from src.models.TaGeDiff import TaGeDiff

QUANTILES = (0.001, 0.01, 0.5, 0.99, 0.999)


def spatial_gradient_energy(z: torch.Tensor) -> float:
    gd = z[:, 1:, :, :] - z[:, :-1, :, :]
    gh = z[:, :, 1:, :] - z[:, :, :-1, :]
    gw = z[:, :, :, 1:] - z[:, :, :, :-1]
    total = (gd.pow(2).sum() + gh.pow(2).sum() + gw.pow(2).sum())
    n = gd.numel() + gh.numel() + gw.numel()
    return float((total / max(n, 1)).item())


def high_freq_fourier_fraction(z: torch.Tensor) -> float:
    C, d, h, w = z.shape
    fracs = []
    zc, yc, xc = d / 2.0, h / 2.0, w / 2.0
    zz, yy, xx = torch.meshgrid(
        torch.arange(d, device=z.device) - zc,
        torch.arange(h, device=z.device) - yc,
        torch.arange(w, device=z.device) - xc,
        indexing="ij",
    )
    radius = torch.sqrt(zz.float() ** 2 + yy.float() ** 2 + xx.float() ** 2)
    r_max = min(zc, yc, xc)
    hf_mask = radius > (0.5 * r_max)
    for c in range(C):
        F = torch.fft.fftshift(torch.fft.fftn(z[c].float()))
        power = (F.real ** 2 + F.imag ** 2)
        total = power.sum()
        hf = power[hf_mask].sum()
        fracs.append(float((hf / total.clamp_min(1e-12)).item()))
    return float(np.mean(fracs))


def per_channel_stats(z: torch.Tensor) -> Dict[str, List[float]]:
    C = z.shape[0]
    stats = {"mean": [], "std": [], "min": [], "max": []}
    for q in QUANTILES:
        stats[f"q{q}"] = []
    for c in range(C):
        zc = z[c].float()
        stats["mean"].append(float(zc.mean().item()))
        stats["std"].append(float(zc.std().item()))
        stats["min"].append(float(zc.min().item()))
        stats["max"].append(float(zc.max().item()))
        flat = zc.flatten()
        qs = torch.quantile(flat, torch.tensor(QUANTILES, device=flat.device, dtype=flat.dtype))
        for q, val in zip(QUANTILES, qs.tolist()):
            stats[f"q{q}"].append(float(val))
    return stats


def full_summary(z: torch.Tensor) -> Dict[str, float]:
    pc = per_channel_stats(z)
    out = {f"mean_{k}": float(np.mean(v)) for k, v in pc.items()}
    out["latent_norm"] = float(z.float().flatten().norm(p=2).item())
    out["grad_energy"] = spatial_gradient_energy(z)
    out["hf_fourier_frac"] = high_freq_fourier_fraction(z)
    return out


def aggregate(rows: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    if not rows:
        return {}
    keys = rows[0].keys()
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = (float(vals.mean()), float(vals.std()))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patients-split", required=True)
    parser.add_argument("--patient-id-col", default="patient_id")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--split", default=None)
    parser.add_argument("--num-patients", type=int, default=8, help="Cap on number of patients (0 = all).")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--ema-policy", default="auto", choices=["auto", "force_ema", "force_raw"])
    parser.add_argument("--save-dir", default="./evaluations/latent_diagnostics")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    raw_ckpt = read_checkpoint(args.checkpoint)
    cfg = resolve_training_mode(cfg, raw_ckpt, args.checkpoint)
    cfg = resolve_representation(cfg, raw_ckpt, args.checkpoint)
    cfg = resolve_condition_on_treatment(cfg, raw_ckpt, args.checkpoint)
    cfg = resolve_temporal_aggregator_variant(cfg, raw_ckpt, args.checkpoint)

    if cfg["data"].get("representation", "latent") != "latent":
        raise ValueError(
            "[latent_diagnostics] this script compares MedVAE latent statistics — meaningless for "
            "data.representation='voxel' checkpoints (no VAE latent exists in voxel mode)."
        )

    model = TaGeDiff(cfg).to(args.device)
    model.eval()
    ckpt_info = load_checkpoint(model, raw_ckpt, ema_policy=args.ema_policy, device=args.device)
    volume_size = tuple(cfg["data"].get("volume_size", ckpt_info["volume_size"]))

    patient_ids = read_patient_ids_from_split(
        csv_path=args.patients_split, patient_id_col=args.patient_id_col,
        split_col=args.split_col, split=args.split,
    )
    if args.num_patients > 0:
        patient_ids = patient_ids[: args.num_patients]
    print(f"[LatentDiag] {len(patient_ids)} patient(s), latent_scale={model.latent_scale}, "
          f"latent_mean={model.latent_mean}, steps={args.steps}, device={args.device}")

    variants = ["real_raw", "real_norm", "real_inv", "gen_norm", "gen_inv"]
    all_rows: Dict[str, List[Dict[str, float]]] = {v: [] for v in variants}
    per_channel_examples: Dict[str, Dict[str, List[float]]] = {}

    for patient_id in patient_ids:
        try:
            patient_data = load_patient(cfg["data"]["data_dir"], patient_id, cfg, geno_preproc=ckpt_info["geno_preproc"])
        except Exception as exc:
            print(f"[LatentDiag] skip {patient_id}: {exc}")
            continue
        num_sessions = int(patient_data["label"].shape[0])
        if num_sessions < 2:
            continue

        for t_idx in range(1, num_sessions):
            batch, gt_vol_np, _ = build_batch(
                patient_data, t_idx, volume_size, args.device,
                normalization_policy=ckpt_info["normalization_policy"],
                normalization_clip_pct=ckpt_info["normalization_clip_pct"],
                representation="latent",
            )
            with torch.no_grad():
                z_real_raw = model.flair_encoder(
                    torch.from_numpy(gt_vol_np).unsqueeze(0).to(args.device)
                ).squeeze(0)
                z_real_norm = (z_real_raw - model.latent_mean) * model.latent_scale
                z_real_inv = z_real_norm / model.latent_scale + model.latent_mean

                out = model.generate(
                    input_images=batch["input_images"], input_days=batch["input_days"],
                    input_treatments=batch["input_treatments"], input_mask=batch["input_mask"],
                    genomics=batch["geno"], geno_mask=batch["geno_mask"],
                    target_day=batch["target_day"], target_treatment=batch["target_treatment"],
                    fm_steps=args.steps, ddim_steps=args.steps,
                )
                z_gen_norm = out["z_0"].squeeze(0)
                z_gen_inv = z_gen_norm / model.latent_scale + model.latent_mean

            tensors = {
                "real_raw": z_real_raw, "real_norm": z_real_norm, "real_inv": z_real_inv,
                "gen_norm": z_gen_norm, "gen_inv": z_gen_inv,
            }
            for name, z in tensors.items():
                all_rows[name].append(full_summary(z))
                if name not in per_channel_examples:
                    per_channel_examples[name] = per_channel_stats(z)

            print(f"  [{patient_id} t={t_idx}] "
                  f"real_raw std={all_rows['real_raw'][-1]['mean_std']:.4f} "
                  f"gen_inv std={all_rows['gen_inv'][-1]['mean_std']:.4f} | "
                  f"real_raw hf={all_rows['real_raw'][-1]['hf_fourier_frac']:.4f} "
                  f"gen_inv hf={all_rows['gen_inv'][-1]['hf_fourier_frac']:.4f}")

    print("\n" + "=" * 96)
    print(f"{'variant':<12}{'n':>5}  " + "".join(f"{k:>16}" for k in
          ["mean", "std", "min", "max", "norm", "grad_energy", "hf_frac"]))
    print("=" * 96)
    summary_out = {}
    for name in variants:
        agg = aggregate(all_rows[name])
        if not agg:
            continue
        summary_out[name] = {k: v for k, v in agg.items()}
        row = [
            agg["mean_mean"][0], agg["mean_std"][0], agg["mean_min"][0], agg["mean_max"][0],
            agg["latent_norm"][0], agg["grad_energy"][0], agg["hf_fourier_frac"][0],
        ]
        print(f"{name:<12}{len(all_rows[name]):>5}  " + "".join(f"{v:>16.5f}" for v in row))
    print("=" * 96)

    real_std = summary_out.get("real_raw", {}).get("mean_std", (float("nan"),))[0]
    gen_std = summary_out.get("gen_inv", {}).get("mean_std", (float("nan"),))[0]
    real_hf = summary_out.get("real_raw", {}).get("hf_fourier_frac", (float("nan"),))[0]
    gen_hf = summary_out.get("gen_inv", {}).get("hf_fourier_frac", (float("nan"),))[0]
    real_grad = summary_out.get("real_raw", {}).get("grad_energy", (float("nan"),))[0]
    gen_grad = summary_out.get("gen_inv", {}).get("grad_energy", (float("nan"),))[0]

    print(f"\ngen_inv / real_raw ratios (decoder-input space; <1 => generated latent is smoother/"
          f"lower-variance than real, consistent with blurry decoded output):")
    print(f"  std ratio          : {gen_std / real_std:.4f}")
    print(f"  spatial-grad ratio : {gen_grad / real_grad:.4f}")
    print(f"  high-freq ratio    : {gen_hf / real_hf:.4f}")

    real_inv_vs_real_raw = summary_out.get("real_inv", {}).get("mean_mean", (float("nan"),))[0] - \
        summary_out.get("real_raw", {}).get("mean_mean", (float("nan"),))[0]
    print(f"\nround-trip sanity (real_inv - real_raw mean, should be ~0): {real_inv_vs_real_raw:.6e}")

    with open(out_dir / "latent_diagnostics_summary.json", "w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "latent_scale": float(model.latent_scale),
                "latent_mean": float(model.latent_mean),
                "n_samples": {v: len(all_rows[v]) for v in variants},
                "aggregate": summary_out,
                "per_channel_last_example": per_channel_examples,
                "ratios_gen_inv_over_real_raw": {
                    "std": gen_std / real_std, "grad_energy": gen_grad / real_grad, "hf_fourier_frac": gen_hf / real_hf,
                },
            },
            f, indent=2,
        )
    print(f"\n[LatentDiag] wrote {out_dir / 'latent_diagnostics_summary.json'}")


if __name__ == "__main__":
    main()
