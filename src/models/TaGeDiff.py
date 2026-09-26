import math
import os

import torch
import torch.nn as nn
from typing import Dict

from ..utils.context_encoder import VisitContextBinder
from ..utils.temporal_aggregator import build_temporal_aggregator
from ..utils.diffusion import GaussianDiffusion
from ..utils.flow_matching import ConditionalFlowMatching, FLOW_PATHS, FLOW_LOSSES, BLEND_MODES, blend_source
from ..utils.preservation_map import (
    build_preservation_map, build_local_density_preservation_map,
    build_hard_dilation_preservation_map, build_horizon_decay_preservation_map,
    bg_retention_to_lambda, PRESERVATION_MODES,
)
from ..utils.regression_loss import (
    tumor_weighted_regression_loss, compute_flow_loss,
    compute_curriculum_alpha, compute_curriculum_progress, CURRICULUM_SCHEDULES,
)
from ..utils.seg_head import LatentSegmentationHead3D, SegmentationLoss
from .unet3d import ConditionalUNet3D, GlobalConditionEncoder, MetadataTokenBuilder
from .vae_wrapper import FlairOnlyEncoder3D, FrozenMedVAE3D
import torch.nn.functional as F


class TaGeDiff(nn.Module):

    VALID_TRAINING_MODES = ("diffusion", "flow_matching")

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        vae_cfg = cfg["vae"]
        ctx_cfg = cfg["context"]
        ta_cfg = cfg["temporal_aggregator"]
        unet_cfg = cfg["unet"]
        diff_cfg = cfg["diffusion"]
        fm_cfg = cfg.get("flow_matching", {})
        seg_cfg = cfg["seg_head"]

        self.training_mode = cfg["training"].get("training_mode", "diffusion")
        if self.training_mode not in self.VALID_TRAINING_MODES:
            raise ValueError(
                f"Unknown training_mode={self.training_mode!r}, expected one of {self.VALID_TRAINING_MODES}"
            )

        self.representation = cfg["data"].get("representation", "latent")
        if self.representation not in ("latent", "voxel"):
            raise ValueError(
                f"Unknown data.representation={self.representation!r}, expected 'latent' or 'voxel'"
            )

        self.condition_on_treatment = bool(ctx_cfg.get("condition_on_treatment", True))
        self.bind_context_to_images = bool(ctx_cfg.get("bind_context_to_images", True))

        self.latent_context = bool(ctx_cfg.get("latent_context_in_voxel", False))
        if self.latent_context and self.representation != "voxel":
            print("[TaGeDiff] context.latent_context_in_voxel=True ignored: data.representation != 'voxel'.")
            self.latent_context = False

        C_z = vae_cfg["latent_channels"]
        backend = vae_cfg.get("backend", "medvae")
        if self.representation == "voxel":
            if backend not in ("medvae", "identity"):
                print(
                    f"[TaGeDiff] data.representation='voxel': ignoring vae.backend={backend!r} "
                    "(no VAE runs in voxel mode) — using an identity pass-through."
                )
            from .vae_wrapper import build_identity_voxel_representation
            self.vae = build_identity_voxel_representation(vae_cfg)
        elif backend == "medvae":
            self.vae = FrozenMedVAE3D.from_config(vae_cfg)
        elif backend == "dummy":
            from .vae_wrapper import build_dummy_vae3d
            self.vae = build_dummy_vae3d(vae_cfg)
        elif backend == "identity":
            raise ValueError(
                "vae.backend='identity' is only valid together with data.representation='voxel' "
                f"(this model was constructed with data.representation={self.representation!r}). "
                "If you intended voxel mode, set data.representation='voxel' too (e.g. via "
                "configs/voxel.yaml); if you intended latent mode, set vae.backend to 'medvae' or 'dummy'."
            )
        else:
            raise ValueError(f"Unknown vae.backend={backend!r}, expected 'medvae' or 'dummy'")
        self.latent_scale = vae_cfg["latent_scale"]
        self.latent_mean = vae_cfg.get("latent_mean", 0.0) if vae_cfg.get("use_latent_mean", True) else 0.0
        self.ta_cfg = ta_cfg
        self.flair_encoder = FlairOnlyEncoder3D(
            vae=self.vae,
            flair_index=cfg["data"].get("flair_index", 2),
            strict=True,
        )

        C_ctx = C_z
        self.ctx_latent_scale, self.ctx_latent_mean = 1.0, 0.0
        if self.latent_context:
            cvae_cfg = cfg.get("context_vae")
            if cvae_cfg is None:
                raise ValueError(
                    "context.latent_context_in_voxel=True requires a top-level `context_vae:` block "
                    "(backend, latent_channels, downsample_factor, latent_scale, [latent_mean])."
                )
            cbackend = cvae_cfg.get("backend", "medvae")
            if cbackend == "medvae":
                self.context_vae = FrozenMedVAE3D.from_config(cvae_cfg)
            elif cbackend == "dummy":
                from .vae_wrapper import build_dummy_vae3d
                self.context_vae = build_dummy_vae3d(cvae_cfg)
            else:
                raise ValueError(f"Unknown context_vae.backend={cbackend!r}, expected 'medvae' or 'dummy'")
            C_ctx = int(cvae_cfg["latent_channels"])
            self.ctx_latent_scale = cvae_cfg.get("latent_scale", 1.0)
            self.ctx_latent_mean = cvae_cfg.get("latent_mean", 0.0) if cvae_cfg.get("use_latent_mean", True) else 0.0
            self.ctx_cache_dir = cfg["data"].get("voxel_latent_cache_dir")
            if self.ctx_cache_dir:
                os.makedirs(self.ctx_cache_dir, exist_ok=True)
            self.voxel_shape_tag = "x".join(str(int(v)) for v in cfg["data"].get("voxel_shape", ()))
            self.context_encoder3d = FlairOnlyEncoder3D(
                vae=self.context_vae, flair_index=cfg["data"].get("flair_index", 2), strict=True,
            )

        self.context_binder = VisitContextBinder(
            latent_channels=C_ctx,
            time_embed_dim=ctx_cfg["time_embed_dim"],
            treatment_vocab_size=ctx_cfg["treatment_vocab_size"],
            treatment_embed_dim=ctx_cfg["treatment_embed_dim"],
            treatment_continuous_dim=ctx_cfg.get("treatment_continuous_dim", 0),
            context_dim=ctx_cfg["context_dim"],
            binding_mode=ctx_cfg["binding_mode"],
            condition_on_treatment=self.condition_on_treatment,
            bind_context_to_images=self.bind_context_to_images,
        )

        self.temporal_aggregator_variant = ta_cfg.get("variant", "temporal_spatial")
        self.temporal_aggregator = build_temporal_aggregator(ta_cfg, channels=C_ctx)

        self.metadata_token_builder = MetadataTokenBuilder(
            treatment_vocab_size=ctx_cfg["treatment_vocab_size"],
            genomic_dim=ctx_cfg["genomic_dim"],
            token_dim=unet_cfg["global_cond_dim"],
            condition_on_treatment=self.condition_on_treatment,
        )

        self.global_cond_encoder = GlobalConditionEncoder(
            global_cond_dim=unet_cfg["global_cond_dim"],
            treatment_vocab_size=ctx_cfg["treatment_vocab_size"],
            genomic_dim=ctx_cfg["genomic_dim"],
            condition_on_treatment=self.condition_on_treatment,
        )

        self.unet = ConditionalUNet3D(
            in_channels=C_z,
            out_channels=C_z,
            base_channels=unet_cfg["base_channels"],
            channel_multipliers=tuple(unet_cfg["channel_multipliers"]),
            num_res_blocks=unet_cfg["num_res_blocks"],
            num_heads=unet_cfg["num_heads"],
            dropout=unet_cfg["dropout"],
            global_cond_dim=unet_cfg["global_cond_dim"],
            concat_temporal_summary=unet_cfg["concat_temporal_summary"],
            concat_last=unet_cfg["concat_last"],
            cross_attn_temporal_sequence=unet_cfg["cross_attn_temporal_sequence"],
            temporal_context_channels=C_ctx,
            concat_channels=C_ctx,
            sn_context_dim=ta_cfg["sn_context_dim"],
            metadata_cross_attn=unet_cfg.get("use_metadata_cross_attn", False),
            metadata_context_dim=unet_cfg["global_cond_dim"],
            attention_min_level=unet_cfg.get("attention_min_level", 1),
        )

        self.diffusion = GaussianDiffusion(
            num_timesteps=diff_cfg["num_timesteps"],
            schedule=diff_cfg["noise_schedule"],
            prediction_type=diff_cfg["prediction_type"],
        )

        self.flow_matching = ConditionalFlowMatching(
            sigma_min=fm_cfg.get("sigma_min", 0.0),
            time_scale=fm_cfg.get("time_scale", 999.0),
        )
        self.fm_cfg = fm_cfg

        self.flow_path = fm_cfg.get("flow_path", "standard")
        if self.flow_path not in FLOW_PATHS:
            raise ValueError(
                f"Unknown flow_matching.flow_path={self.flow_path!r}, expected one of {FLOW_PATHS}"
            )
        self.flow_loss = fm_cfg.get("flow_loss", "tumor_weighted")
        if self.flow_loss not in FLOW_LOSSES:
            raise ValueError(
                f"Unknown flow_matching.flow_loss={self.flow_loss!r}, expected one of {FLOW_LOSSES}"
            )

        ntw_cfg = fm_cfg.get("normalized_tumor_weighted", {})
        self.ntw_kernel_shape = tuple(int(k) for k in ntw_cfg.get("kernel_shape", (5, 11, 11)))
        self.ntw_kernel_divisor = float(ntw_cfg.get("kernel_divisor", 10.0))
        self.ntw_normalize_per_sample = bool(ntw_cfg.get("normalize_per_sample", True))

        ntw_curr_cfg = ntw_cfg.get("curriculum", {})
        self.ntw_curriculum_enabled = bool(ntw_curr_cfg.get("enabled", True))
        self.ntw_curriculum_schedule = ntw_curr_cfg.get("schedule", "cosine")
        self.ntw_curriculum_start_epoch = float(ntw_curr_cfg.get("start_epoch", 0))
        self.ntw_curriculum_ramp_epochs = float(ntw_curr_cfg.get("ramp_epochs", 1))

        if self.flow_loss == "normalized_tumor_weighted":
            if len(self.ntw_kernel_shape) != 3 or any(k <= 0 for k in self.ntw_kernel_shape):
                raise ValueError(
                    "flow_matching.normalized_tumor_weighted.kernel_shape must be a "
                    f"(kD,kH,kW) triple of positive ints, got {self.ntw_kernel_shape}"
                )
            if any(k % 2 == 0 for k in self.ntw_kernel_shape):
                raise ValueError(
                    "flow_matching.normalized_tumor_weighted.kernel_shape components must "
                    "all be odd (required for symmetric 'same'-shape conv3d padding — same "
                    f"convention as region_aware_source.kernel_size), got {self.ntw_kernel_shape}"
                )
            if self.ntw_kernel_divisor <= 0:
                raise ValueError(
                    "flow_matching.normalized_tumor_weighted.kernel_divisor must be > 0, "
                    f"got {self.ntw_kernel_divisor}"
                )
            if self.ntw_curriculum_schedule not in CURRICULUM_SCHEDULES:
                raise ValueError(
                    "Unknown flow_matching.normalized_tumor_weighted.curriculum.schedule="
                    f"{self.ntw_curriculum_schedule!r}, expected one of {CURRICULUM_SCHEDULES}"
                )
            if self.ntw_curriculum_enabled and self.ntw_curriculum_ramp_epochs <= 0:
                raise ValueError(
                    "flow_matching.normalized_tumor_weighted.curriculum.ramp_epochs must be "
                    f"> 0 when curriculum.enabled=True, got {self.ntw_curriculum_ramp_epochs}"
                )
            print(
                "[TaGeDiff] flow_matching.flow_loss='normalized_tumor_weighted': "
                f"kernel_shape={self.ntw_kernel_shape}, kernel_divisor={self.ntw_kernel_divisor}, "
                f"normalize_per_sample={self.ntw_normalize_per_sample}, curriculum("
                f"enabled={self.ntw_curriculum_enabled}, schedule={self.ntw_curriculum_schedule!r}, "
                f"start_epoch={self.ntw_curriculum_start_epoch}, "
                f"ramp_epochs={self.ntw_curriculum_ramp_epochs})"
            )

        ras_cfg = fm_cfg.get("region_aware_source", {})
        self.region_preservation_mode = ras_cfg.get("preservation_mode", "distance")
        if self.region_preservation_mode not in PRESERVATION_MODES:
            raise ValueError(
                f"Unknown region_aware_source.preservation_mode={self.region_preservation_mode!r}, "
                f"expected one of {PRESERVATION_MODES}"
            )

        self.region_source_sigma = float(ras_cfg.get("sigma", 10.0))
        if self.region_preservation_mode == "distance" and self.region_source_sigma <= 0:
            raise ValueError(
                f"region_aware_source.sigma must be > 0, got {self.region_source_sigma}"
            )

        self.region_kernel_size = tuple(int(k) for k in ras_cfg.get("kernel_size", (5, 11, 11)))
        if len(self.region_kernel_size) != 3:
            raise ValueError(
                f"region_aware_source.kernel_size must be a (kD,kH,kW) triple, got {self.region_kernel_size}"
            )
        if self.region_preservation_mode in ("local_density", "hard_dilation", "horizon_decay"):
            if any(k <= 0 for k in self.region_kernel_size):
                raise ValueError(
                    f"region_aware_source.kernel_size components must all be > 0, got {self.region_kernel_size}"
                )
            if any(k % 2 == 0 for k in self.region_kernel_size):
                raise ValueError(
                    "region_aware_source.kernel_size components must all be odd (required for "
                    f"symmetric 'same'-shape padding — see preservation_map.py), got {self.region_kernel_size}"
                )

        self.region_gamma = float(ras_cfg.get("gamma", 1.0))
        if self.region_preservation_mode == "local_density" and self.region_gamma <= 0:
            raise ValueError(f"region_aware_source.gamma must be > 0, got {self.region_gamma}")

        self.region_p_min = float(ras_cfg.get("p_min", 0.1))
        if not (0.0 <= self.region_p_min <= 1.0):
            raise ValueError(f"region_aware_source.p_min must be in [0, 1], got {self.region_p_min}")

        if self.region_preservation_mode == "hard_dilation":
            if "gamma" in ras_cfg and float(ras_cfg["gamma"]) != 1.0:
                print(
                    "[TaGeDiff] WARNING: region_aware_source.gamma is set but "
                    "preservation_mode='hard_dilation' ignores it (gamma is a "
                    "'local_density'-only knob) — no effect."
                )
            if "p_min" in ras_cfg and float(ras_cfg["p_min"]) != 0.1:
                print(
                    "[TaGeDiff] WARNING: region_aware_source.p_min is set but "
                    "preservation_mode='hard_dilation' ignores it (p_min is a "
                    "'local_density'-only knob) — no effect."
                )

        self.region_blend = ras_cfg.get("blend", "linear")
        if self.region_blend not in BLEND_MODES:
            raise ValueError(
                f"Unknown region_aware_source.blend={self.region_blend!r}, expected one of {BLEND_MODES}"
            )

        hd_cfg = ras_cfg.get("horizon_decay", {})
        self.region_horizon_gamma = float(hd_cfg.get("gamma", self.region_gamma))
        self.region_bg_retention = float(hd_cfg.get("bg_retention", 0.90))
        self.region_bg_retention_horizon_days = float(hd_cfg.get("bg_retention_horizon_days", 365.0))
        if self.region_preservation_mode == "horizon_decay" and self.region_horizon_gamma <= 0:
            raise ValueError(
                f"region_aware_source.horizon_decay.gamma must be > 0, got {self.region_horizon_gamma}"
            )
        self.region_lambda_bg = bg_retention_to_lambda(
            self.region_bg_retention, self.region_bg_retention_horizon_days
        )
        if self.region_preservation_mode == "horizon_decay":
            ref_days = sorted({90, 365, 730, int(round(self.region_bg_retention_horizon_days))})
            retention_str = ", ".join(
                f"{d}d={math.exp(-self.region_lambda_bg * d):.3f}" for d in ref_days
            )
            print(
                f"[TaGeDiff] region_aware_source.horizon_decay: lambda_bg={self.region_lambda_bg:.6g}/day "
                f"(bg_retention={self.region_bg_retention} @ {self.region_bg_retention_horizon_days:.0f} days); "
                f"implied distant-tissue temporal retention: {retention_str}; blend={self.region_blend!r}"
            )

        if self.flow_path == "region_aware_source" and self.training_mode != "flow_matching":
            print(
                "[TaGeDiff] WARNING: flow_matching.flow_path='region_aware_source' is set but "
                f"training.training_mode={self.training_mode!r} — region_aware_source only "
                "affects training_mode='flow_matching' and has no effect here."
            )

        self.seg_enabled = seg_cfg["enabled"]
        if self.seg_enabled:
            self.seg_head = LatentSegmentationHead3D(
                latent_channels=C_z,
                hidden_channels=seg_cfg["hidden_channels"],
                num_classes=seg_cfg["num_classes"],
                upsample_factor=vae_cfg["downsample_factor"],
            )
            self.seg_loss_fn = SegmentationLoss(
                loss_weight=seg_cfg["loss_weight"],
                noise_threshold=seg_cfg["noise_threshold"],
            )

    @torch.no_grad()
    def encode_visits(self, images, mask):
        B, N_max = images.shape[:2]
        latents = []
        for n in range(N_max):
            z_n = self.flair_encoder(images[:, n])
            z_n = (z_n - self.latent_mean) * self.latent_scale
            latents.append(z_n)
        return latents

    def _ctx_cache_path(self, key):
        return os.path.join(self.ctx_cache_dir, f"{key}_{self.voxel_shape_tag}.npy")

    def encode_context_visits(self, images, visit_keys=None, input_mask=None):
        import numpy as np
        use_cache = bool(self.ctx_cache_dir) and visit_keys is not None
        B, N = images.shape[:2]
        out = []
        for n in range(N):
            if not use_cache:
                z = self.context_encoder3d(images[:, n])
            else:
                per_b = []
                for b in range(B):
                    valid = input_mask is None or bool(input_mask[b, n])
                    if not valid or n >= len(visit_keys[b]):
                        per_b.append(None)
                        continue
                    path = self._ctx_cache_path(visit_keys[b][n])
                    z_b = None
                    if os.path.exists(path):
                        try:
                            z_b = torch.from_numpy(np.load(path)).to(images.device, images.dtype)[None]
                        except Exception:
                            z_b = None
                    if z_b is None:
                        z_b = self.context_encoder3d(images[b:b + 1, n])
                        tmp = f"{path}.{os.getpid()}.tmp.npy"
                        np.save(tmp, z_b[0].float().cpu().numpy())
                        os.replace(tmp, path)
                    per_b.append(z_b)
                ref = next((z for z in per_b if z is not None), None)
                if ref is None:
                    z = self.context_encoder3d(images[:, n])
                else:
                    z = torch.cat([zb if zb is not None else torch.zeros_like(ref) for zb in per_b], dim=0)
            out.append((z - self.ctx_latent_mean) * self.ctx_latent_scale)
        return out

    @staticmethod
    def _gather_last_valid(stacked: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        last_idx = (input_mask.sum(dim=1).clamp(min=1) - 1).to(stacked.device)
        batch_idx = torch.arange(stacked.shape[0], device=stacked.device)
        return stacked[batch_idx, last_idx]

    def _horizon_decay_dt_bucket_diag(self, preservation_map: torch.Tensor, dt: torch.Tensor) -> dict:
        dt = dt.detach().float().reshape(-1)
        short_thr = self.region_bg_retention_horizon_days / 4.0
        long_thr = self.region_bg_retention_horizon_days
        is_short = dt < short_thr
        is_long = dt >= long_thr
        is_medium = (~is_short) & (~is_long)

        def _bucket_mean(sel: torch.Tensor) -> float:
            if not bool(sel.any()):
                return float("nan")
            sel_full = sel.view(-1, 1, 1, 1, 1).expand_as(preservation_map)
            return preservation_map[sel_full].mean().item()

        diag = {
            "P_dt_short_mean": _bucket_mean(is_short),
            "P_dt_medium_mean": _bucket_mean(is_medium),
            "P_dt_long_mean": _bucket_mean(is_long),
        }
        populated = [v for v in diag.values() if not math.isnan(v)]
        if len(populated) >= 2 and (max(populated) - min(populated)) < 1e-4:
            print(
                "[TaGeDiff] WARNING: region_aware_source.horizon_decay's per-dt-bucket mean P "
                "values are nearly identical within this batch despite differing horizons — the "
                "horizon (dt) may not be reaching the preservation-map formula."
            )
        return diag

    def forward(
        self, batch: Dict[str, torch.Tensor], log_diagnostics: bool = False, epoch: int = None,
    ) -> Dict[str, torch.Tensor]:
        device = batch["target_day"].device
        B = batch["target_day"].shape[0]

        treatment_for_binder = None
        treatment_for_global = None
        if self.condition_on_treatment:
            treatment_for_binder = batch["input_treatments"]
            treatment_for_global = batch["target_treatment"]

            input_treatments = batch["input_treatments"]
            target_treatment = batch["target_treatment"]

            p = float(self.cfg["training"].get("cond_dropout_prob", 0.0))
            null_tx = int(self.cfg["training"].get("null_treatment_id", 3))

            if self.training and p > 0:
                drop_inputs = (torch.rand_like(input_treatments.float()) < p) & batch["input_mask"].bool()
                input_treatments = torch.where(
                    drop_inputs,
                    torch.full_like(input_treatments, null_tx),
                    input_treatments
                )

                drop_target = (torch.rand_like(target_treatment.float()) < p)
                target_treatment = torch.where(
                    drop_target,
                    torch.full_like(target_treatment, null_tx),
                    target_treatment
                )

        z_ctx = None
        if self.latent_context and "input_latents" in batch:
            z_ctx = [(batch["input_latents"][:, n] - self.ctx_latent_mean) * self.ctx_latent_scale
                     for n in range(batch["input_latents"].shape[1])]
        if not self.latent_context and "input_latents" in batch and "target_latent" in batch:
            N_max = batch["input_latents"].shape[1]
            z_target_raw = batch["target_latent"]
            z_inputs = [(batch["input_latents"][:, n] - self.latent_mean) * self.latent_scale for n in range(N_max)]
        else:
            z_inputs = self.encode_visits(batch["input_images"], batch["input_mask"])
            z_target_raw = self.flair_encoder(batch["target_image"])
        z_target = (z_target_raw - self.latent_mean) * self.latent_scale


        if self.latent_context and z_ctx is None:
            z_ctx = self.encode_context_visits(
                batch["input_images"], batch.get("input_visit_keys"), batch["input_mask"]
            )
        elif not self.latent_context:
            z_ctx = z_inputs
        z_bound = self.context_binder(
            z_ctx, batch["input_days"], treatment_for_binder, batch["input_mask"]
        )

        Z_agg, h_temporal = self.temporal_aggregator(z_bound, batch["input_mask"])
        s = None

        region_diag = None
        if self.training_mode == "diffusion":
            t = torch.randint(0, self.diffusion.T, (B,), device=device)
            z_t, noise = self.diffusion.q_sample(z_target, t)
            t_embed = t
            regression_target = noise
        else:
            t_cont = self.flow_matching.sample_t(B, device)

            if self.flow_path == "standard":
                z_t, source, velocity = self.flow_matching.construct_flow_path(
                    z_target, t_cont, path_type="standard",
                )
            else:
                if "input_labels" not in batch:
                    raise ValueError(
                        "flow_matching.flow_path='region_aware_source' requires the baseline "
                        "segmentation mask in every training batch (batch['input_labels']) to "
                        "build the preservation map — got a batch missing it."
                    )
                z_stack = torch.stack(z_inputs, dim=1)
                z_prev = self._gather_last_valid(z_stack, batch["input_mask"])
                m0_img = self._gather_last_valid(batch["input_labels"], batch["input_mask"])

                region_density_diag = None
                if log_diagnostics and self.region_preservation_mode == "local_density":
                    preservation_map, R = build_local_density_preservation_map(
                        m0_img, kernel_size=self.region_kernel_size, p_min=self.region_p_min,
                        gamma=self.region_gamma, target_shape=z_target.shape[-3:], return_density=True,
                    )
                    region_density_diag = {
                        "R_mean": R.mean().item(), "R_min": R.min().item(), "R_max": R.max().item(),
                    }
                elif log_diagnostics and self.region_preservation_mode == "hard_dilation":
                    preservation_map, D = build_hard_dilation_preservation_map(
                        m0_img, kernel_size=self.region_kernel_size,
                        target_shape=z_target.shape[-3:], return_dilated_mask=True,
                    )
                    region_density_diag = {
                        "D_mean": D.mean().item(),
                    }
                elif log_diagnostics and self.region_preservation_mode == "horizon_decay":
                    dt_vals = batch["target_day"].detach()
                    preservation_map, R = build_horizon_decay_preservation_map(
                        m0_img, dt=dt_vals, lambda_bg=self.region_lambda_bg,
                        kernel_size=self.region_kernel_size, gamma=self.region_horizon_gamma,
                        target_shape=z_target.shape[-3:], return_density=True,
                    )
                    region_density_diag = self._horizon_decay_dt_bucket_diag(preservation_map, dt_vals)
                    region_density_diag.update({
                        "R_mean": R.mean().item(), "R_min": R.min().item(), "R_max": R.max().item(),
                        "lambda_bg": self.region_lambda_bg,
                        "dt_mean": dt_vals.float().mean().item(),
                    })
                else:
                    horizon_decay_active = self.region_preservation_mode == "horizon_decay"
                    preservation_map = build_preservation_map(
                        m0_img, mode=self.region_preservation_mode, target_shape=z_target.shape[-3:],
                        sigma=self.region_source_sigma, kernel_size=self.region_kernel_size,
                        p_min=self.region_p_min,
                        gamma=self.region_horizon_gamma if horizon_decay_active else self.region_gamma,
                        dt=batch["target_day"] if horizon_decay_active else None,
                        lambda_bg=self.region_lambda_bg if horizon_decay_active else None,
                    )

                m0_latent = None
                if log_diagnostics:
                    m0 = m0_img.float()
                    if m0.ndim == 4:
                        m0 = m0.unsqueeze(1)
                    m0_latent = F.interpolate(m0, size=z_target.shape[-3:], mode="nearest")

                path_out = self.flow_matching.construct_flow_path(
                    z_target, t_cont, path_type="region_aware_source",
                    z_prev=z_prev, preservation_map=preservation_map,
                    baseline_mask=m0_latent,
                    return_diagnostics=log_diagnostics,
                    blend=self.region_blend,
                )
                if log_diagnostics:
                    z_t, source, velocity, region_diag = path_out
                    region_diag["preservation_mode"] = self.region_preservation_mode
                    region_diag["blend"] = self.region_blend
                    if region_density_diag is not None:
                        region_diag.update(region_density_diag)
                        region_diag["kernel_size"] = list(self.region_kernel_size)
                        if self.region_preservation_mode == "local_density":
                            region_diag["p_min"] = self.region_p_min
                            region_diag["gamma"] = self.region_gamma
                        elif self.region_preservation_mode == "horizon_decay":
                            region_diag["gamma"] = self.region_horizon_gamma
                            region_diag["bg_retention"] = self.region_bg_retention
                            region_diag["bg_retention_horizon_days"] = self.region_bg_retention_horizon_days
                else:
                    z_t, source, velocity = path_out

            t_embed = self.flow_matching.embed_time(t_cont)
            regression_target = velocity

        global_cond = self.global_cond_encoder(t_embed, batch["target_day"], treatment_for_global, genomics=batch["geno"], geno_mask=batch["geno_mask"])

        metadata_tokens = None

        if self.cfg["unet"]["use_metadata_cross_attn"]:
            metadata_tokens = self.metadata_token_builder(
                target_treatment=treatment_for_global,
                genomics=batch["geno"],
                geno_mask=batch["geno_mask"],
            )

        pred = self.unet(
            z_t, global_cond, h_temporal, Z_agg, s,
            metadata_tokens=metadata_tokens, input_mask=batch["input_mask"],
        )

        ntw_diag = None
        if self.training_mode == "flow_matching":
            if self.flow_loss == "normalized_tumor_weighted":
                completed_epochs = (epoch - 1) if epoch is not None else None
                curriculum_alpha = compute_curriculum_alpha(
                    current_epoch=completed_epochs,
                    enabled=self.ntw_curriculum_enabled,
                    schedule=self.ntw_curriculum_schedule,
                    start_epoch=self.ntw_curriculum_start_epoch,
                    ramp_epochs=self.ntw_curriculum_ramp_epochs,
                )
                ntw_diag = {} if log_diagnostics else None
                loss_diff, mse_noise = compute_flow_loss(
                    pred, regression_target, self.flow_loss,
                    tumor_mask=batch["target_label"],
                    curriculum_alpha=curriculum_alpha,
                    kernel_shape=self.ntw_kernel_shape,
                    kernel_divisor=self.ntw_kernel_divisor,
                    normalize_per_sample=self.ntw_normalize_per_sample,
                    diagnostics=ntw_diag,
                )
                if ntw_diag is not None:
                    ntw_diag["absolute_epoch"] = epoch
                    ntw_diag["curriculum_progress"] = (
                        compute_curriculum_progress(
                            completed_epochs,
                            start_epoch=self.ntw_curriculum_start_epoch,
                            ramp_epochs=self.ntw_curriculum_ramp_epochs,
                        )
                        if self.ntw_curriculum_enabled
                        else 1.0
                    )
            else:
                loss_diff, mse_noise = compute_flow_loss(
                    pred, regression_target, self.flow_loss,
                    tumor_mask=batch["target_label"],
                )
        else:
            loss_diff, mse_noise = tumor_weighted_regression_loss(
                pred, regression_target, batch["target_label"],
            )

        out = {
            "loss": loss_diff,
            "mse_noise": mse_noise,
        }
        if region_diag is not None:
            out["region_diag"] = region_diag
        if ntw_diag:
            out["ntw_diag"] = ntw_diag
        return out

    @torch.no_grad()
    def generate(self, input_images, input_days, input_treatments=None, input_mask=None,
                 genomics=None, geno_mask=None, target_day=None, target_treatment=None,
                 ddim_steps=50, ddim_eta=0.0, fm_steps=None, fm_solver=None,
                 input_latents=None, input_labels=None, input_visit_keys=None):
        treatment_for_binder = input_treatments if self.condition_on_treatment else None
        treatment_for_global = target_treatment if self.condition_on_treatment else None

        z_ctx = None
        if self.latent_context and input_latents is not None:
            z_ctx = [(input_latents[:, n] - self.ctx_latent_mean) * self.ctx_latent_scale
                     for n in range(input_latents.shape[1])]
        if input_latents is not None and not self.latent_context:
            N_max = input_latents.shape[1]
            z_inputs = [(input_latents[:, n] - self.latent_mean) * self.latent_scale for n in range(N_max)]
        else:
            z_inputs = self.encode_visits(input_images, input_mask)
        if self.latent_context and z_ctx is None:
            z_ctx = self.encode_context_visits(input_images, input_visit_keys, input_mask)
        elif not self.latent_context:
            z_ctx = z_inputs
        z_bound = self.context_binder(z_ctx, input_days, treatment_for_binder, input_mask)
        s = None
        Z_agg, h_temporal = self.temporal_aggregator(z_bound, input_mask)

        z_grid = input_images.shape[-3:] if self.latent_context else h_temporal.shape[2:]
        z_shape = (input_images.shape[0], self.cfg["vae"]["latent_channels"], *z_grid)

        init_state = None
        if self.training_mode == "flow_matching" and self.flow_path == "region_aware_source":
            if input_labels is None:
                raise ValueError(
                    "flow_matching.flow_path='region_aware_source' requires input_labels "
                    "(the baseline/history segmentation, same shape as batch['input_labels']) "
                    "to be passed to generate() — the previous MRI's tumor segmentation is "
                    "needed to construct the region-aware source state. Never pass a "
                    "follow-up/target mask here — only information available at inference time."
                )
            horizon_decay_active = self.region_preservation_mode == "horizon_decay"
            if horizon_decay_active and target_day is None:
                raise ValueError(
                    "flow_matching.flow_path='region_aware_source' with preservation_mode="
                    "'horizon_decay' requires target_day (the horizon, in days, from the most "
                    "recent previous visit to the target — see src/data/dataset.py's "
                    "target_day_rel) to be passed to generate() — got None."
                )
            z_stack = torch.stack(z_inputs, dim=1)
            z_prev = self._gather_last_valid(z_stack, input_mask)
            m0_img = self._gather_last_valid(input_labels, input_mask)
            preservation_map = build_preservation_map(
                m0_img, mode=self.region_preservation_mode, target_shape=z_shape[-3:],
                sigma=self.region_source_sigma, kernel_size=self.region_kernel_size,
                p_min=self.region_p_min,
                gamma=self.region_horizon_gamma if horizon_decay_active else self.region_gamma,
                dt=target_day if horizon_decay_active else None,
                lambda_bg=self.region_lambda_bg if horizon_decay_active else None,
            )
            noise0 = torch.randn(z_shape, device=input_images.device)
            init_state = blend_source(preservation_map, z_prev, noise0, blend=self.region_blend)

        def build_metadata_tokens():
            if self.cfg["unet"]["use_metadata_cross_attn"]:
                return self.metadata_token_builder(
                    target_treatment=treatment_for_global,
                    genomics=genomics,
                    geno_mask=geno_mask,
                )
            return None

        if self.training_mode == "diffusion":
            def denoise_fn(z_t, t_tensor, **kw):
                gc = self.global_cond_encoder(t_tensor, target_day, treatment_for_global, genomics, geno_mask)
                metadata_tokens = build_metadata_tokens()
                return self.unet(z_t, gc, h_temporal, Z_agg, s, metadata_tokens, input_mask=input_mask)

            z_0 = self.diffusion.ddim_sample(denoise_fn, z_shape, ddim_steps, ddim_eta, input_images.device)
        else:
            steps = fm_steps if fm_steps is not None else self.fm_cfg.get("num_steps", 50)
            solver = fm_solver if fm_solver is not None else self.fm_cfg.get("solver", "euler")

            def velocity_fn(z_t, t_cont, **kw):
                t_embed = self.flow_matching.embed_time(t_cont)
                gc = self.global_cond_encoder(t_embed, target_day, treatment_for_global, genomics, geno_mask)
                metadata_tokens = build_metadata_tokens()
                return self.unet(z_t, gc, h_temporal, Z_agg, s, metadata_tokens, input_mask=input_mask)

            z_0 = self.flow_matching.sample(
                velocity_fn, z_shape, num_steps=steps, solver=solver,
                device=input_images.device, init_state=init_state,
            )

        mri = self.vae.decode(z_0 / self.latent_scale + self.latent_mean)
        seg = torch.sigmoid(self.seg_head(z_0)) if self.seg_enabled else None

        return {"generated_mri": mri, "generated_seg": seg, "z_0": z_0}
