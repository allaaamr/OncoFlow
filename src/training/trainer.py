import os
import time
import warnings
from typing import Dict, Optional

import wandb
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from ..models.TaGeDiff import TaGeDiff
from ..data.dataset import (
    PatientDataset,
    PatientLocalityBatchSampler,
    collate_variable_length,
    split_patients_by_id,
    collect_raw_genomics_for_patients,
    assert_disjoint_splits,
    load_image_stats,
)
from ..data.preprocessing import GenomicsPreprocessor, ImageIntensityStats
from ..utils.regression_loss import compute_curriculum_alpha, compute_curriculum_progress
import torch.nn.functional as F

CHECKPOINT_SCHEMA_VERSION = 3


def _resolve_mixed_precision(value) -> str:
    if isinstance(value, bool):
        return "fp16" if value else "no"
    value = str(value).strip().lower()
    if value == "true":
        return "fp16"
    if value == "false":
        return "no"
    if value in ("no", "fp16", "bf16", "fp8"):
        return value
    raise ValueError(
        f"Unknown training.mixed_precision={value!r}; expected a bool (legacy) or one "
        "of 'no', 'fp16', 'bf16', 'fp8'."
    )


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {n: p.clone() for n, p in model.named_parameters() if n in self.shadow}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


class Trainer:
    def __init__(self, cfg, device="cuda", resume_path=None):
        self.cfg = cfg
        tc = cfg["training"]
        self.training_mode = tc.get("training_mode", "diffusion")

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            cpu=(device == "cpu"),
            mixed_precision=_resolve_mixed_precision(tc.get("mixed_precision", True)),
            gradient_accumulation_steps=int(tc.get("gradient_accumulation_steps", 1)),
            dataloader_config=DataLoaderConfiguration(split_batches=False),
            kwargs_handlers=[ddp_kwargs],
        )
        self.device = self.accelerator.device
        set_seed(int(cfg.get("data", {}).get("rng_seed", 42)), device_specific=True)

        self.model = TaGeDiff(cfg).to(self.device)

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.accelerator.is_main_process:
            print(f"[Trainer] Params: {total:,} total | {trainable:,} trainable | {total - trainable:,} frozen (VAE)")

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=float(tc["epochs"]), eta_min=float(tc["lr"]) * 0.01)
        self.use_amp = self.accelerator.mixed_precision != "no"
        self.grad_accum = int(tc.get("gradient_accumulation_steps", 1))
        self.ema = EMA(self.model, tc.get("ema_decay", 0.9999))

        self.representation = cfg.get("data", {}).get("representation", "latent")
        mode_dir = os.path.join(tc.get("log_dir", "./runs"), self.training_mode)
        self.log_dir = os.path.join(mode_dir, self.representation) if self.representation != "latent" else mode_dir
        if self.accelerator.is_main_process:
            print(
                f"[Trainer] Checkpoints for this run (training_mode={self.training_mode!r}, "
                f"representation={self.representation!r}) -> {self.log_dir}"
            )
            print(
                f"[Trainer] Accelerate: {self.accelerator.num_processes} process(es), "
                f"mixed_precision={self.accelerator.mixed_precision!r}, "
                f"gradient_accumulation_steps={self.grad_accum}, device={self.device}"
            )
        self.start_epoch = 1

        self.geno_preproc = None
        self.image_stats = None
        self._resumed_normalization_policy = None
        self._resumed_context_sampling = None

        if resume_path is not None:
            self.load_checkpoint(resume_path)

        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        if self.accelerator.is_main_process:
            os.makedirs(self.log_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self.use_wandb = tc.get("use_wandb", True) and self.accelerator.is_main_process
        self.wandb_project = tc.get("wandb_project", "TaGeDiff")
        self.wandb_run_name = tc.get("wandb_run_name", None)

        self.best_val_mse = float("inf")
        self.best_val_dice = -float("inf")

        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config=self.cfg,
            )
            wandb.watch(self.accelerator.unwrap_model(self.model), log=None)

    def build_dataloaders(self):
        dc = self.cfg["data"]
        data_dir = dc["data_dir"]
        volume_size = tuple(dc.get("volume_size", (160, 240, 240)))

        split_csv_path = dc.get("split_csv_path")
        if not split_csv_path:
            raise ValueError(
                "[Trainer] data.split_csv_path is required in the config — it must point at "
                "the patient split CSV this run should train on (e.g. 'patient_split.csv'). "
                "There is no default; this prevents accidentally training on whichever split "
                "file happens to already exist in the current working directory."
            )
        train_ids, val_ids, test_ids = split_patients_by_id(
            data_dir=data_dir,
            train_frac=0.70,
            val_frac=0.15,
            test_frac=0.15,
            seed=int(dc.get("split_seed", 0)),
            save_csv_path=split_csv_path,
            require_existing=True,
        )

        assert_disjoint_splits(train_ids, val_ids, test_ids)
        print(
            f"[Trainer][SPLIT] train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} "
            f"patients (disjoint, verified)"
        )

        if self.geno_preproc is not None and self.geno_preproc.is_fit:
            geno_preproc = self.geno_preproc
            print("[Trainer] Reusing genomics preprocessing statistics restored from checkpoint.")
        else:
            geno_preproc = GenomicsPreprocessor()
            train_raw_geno = collect_raw_genomics_for_patients(data_dir, train_ids)
            if train_raw_geno.shape[0] > 0 and train_raw_geno.shape[1] == int(dc.get("genomic_dim", 13)):
                geno_preproc.fit(train_raw_geno)
                print(
                    f"[Trainer] Fit GenomicsPreprocessor on {train_raw_geno.shape[0]} train patients "
                    f"({int(geno_preproc.zero_variance_mask.sum())} zero-variance features guarded)."
                )
            else:
                print(
                    "[Trainer] WARNING: no usable training-patient genomics found "
                    f"(found shape {train_raw_geno.shape}, expected genomic_dim={dc.get('genomic_dim', 13)}); "
                    "genomics will be used RAW/unnormalized."
                )
            self.geno_preproc = geno_preproc

        normalization_policy = dc.get("normalization_policy", "whole_volume")
        context_sampling = dc.get("context_sampling", "all_windows")

        image_stats: Optional[Dict[int, ImageIntensityStats]] = None
        if normalization_policy == "frozen_train_stats":
            if self.image_stats is not None:
                image_stats = self.image_stats
                print("[Trainer] Reusing frozen image intensity stats restored from checkpoint.")
            else:
                image_stats_path = dc.get("image_stats_path")
                if not image_stats_path:
                    raise ValueError(
                        "[Trainer] data.normalization_policy='frozen_train_stats' requires "
                        "data.image_stats_path to be set — it must point at a stats file already "
                        "produced by `python scripts/fit_image_stats.py --config <this config>`. "
                        "There is no default and Trainer never fits this on the fly, so the exact "
                        "same frozen stats are guaranteed to be used by training, "
                        "scripts/precompute_latents.py (if used), and eval/inference."
                    )
                image_stats, image_stats_meta = load_image_stats(image_stats_path)
                fitted_ids = set(image_stats_meta.get("train_patient_ids", []))
                if fitted_ids and fitted_ids != set(train_ids):
                    warnings.warn(
                        f"[Trainer] data.image_stats_path='{image_stats_path}' was fit on a "
                        f"different train-patient set ({len(fitted_ids)} patients) than the "
                        f"current split ({len(train_ids)} patients). The frozen stats are still "
                        "applied as-is (never refit here) but may no longer reflect the intended "
                        "train cohort — re-run scripts/fit_image_stats.py if this split changed "
                        "intentionally.",
                        stacklevel=2,
                    )
                self.image_stats = image_stats
                print(
                    f"[Trainer] Loaded frozen image intensity stats from '{image_stats_path}' "
                    f"(fit on {len(fitted_ids)} train patients)."
                )
        if self._resumed_normalization_policy is not None and self._resumed_normalization_policy != normalization_policy:
            print(
                f"[Trainer] WARNING: resuming with data.normalization_policy="
                f"'{normalization_policy}' but checkpoint was trained with "
                f"'{self._resumed_normalization_policy}'. This is not a byte-identical "
                "resume — the model will now see a different input distribution."
            )
        if self._resumed_context_sampling is not None and self._resumed_context_sampling != context_sampling:
            print(
                f"[Trainer] WARNING: resuming with data.context_sampling="
                f"'{context_sampling}' but checkpoint was trained with "
                f"'{self._resumed_context_sampling}'."
            )

        latent_dir = dc.get("latent_dir")
        representation = dc.get("representation", "latent")
        latent_context_in_voxel = bool(self.cfg.get("context", {}).get("latent_context_in_voxel", False))
        if representation == "voxel" and latent_dir is not None and not latent_context_in_voxel:
            print(
                "[Trainer] data.representation='voxel': ignoring data.latent_dir "
                f"({latent_dir!r}) — there is no VAE step in voxel mode, so no latent "
                "cache is possible. Loading raw images for every split."
            )
            latent_dir = None

        def make_ds(patient_ids, rng_seed, require_images=True):
            return PatientDataset(
                data_dir=data_dir,
                patient_ids=patient_ids,
                volume_size=volume_size,
                merge_labels=dc.get("merge_labels", True),
                geno_mean=geno_preproc.mean,
                geno_std=geno_preproc.std,
                rng_seed=rng_seed,
                mmap=dc.get("mmap", True),
                dc=dc,
                latent_dir=latent_dir,
                latent_context_in_voxel=latent_context_in_voxel,
                require_images=require_images,
                image_stats=image_stats,
            )

        train_ds = make_ds(train_ids, int(dc.get("rng_seed", 42)), require_images=(latent_dir is None or representation == "voxel"))
        val_ds = make_ds(val_ids, int(dc.get("rng_seed", 84)))
        test_ds = make_ds(test_ids, int(dc.get("rng_seed", 123)))

        sampler_policy = dc.get("sampler_policy", "none")
        train_sampler = self._build_train_sampler(train_ds, sampler_policy, dc)

        batch_size = int(dc.get("batch_size", 4))
        num_workers = int(dc.get("num_workers", 4))
        drop_last = bool(dc.get("drop_last", True))

        if train_sampler is None:
            train_loader = DataLoader(
                train_ds,
                batch_sampler=PatientLocalityBatchSampler(
                    train_ds, group_size=batch_size, drop_last=drop_last,
                    seed=int(dc.get("rng_seed", 42)),
                ),
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_variable_length,
                persistent_workers=(num_workers > 0),
            )
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_variable_length,
                drop_last=drop_last,
                persistent_workers=(num_workers > 0),
            )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_variable_length,
            drop_last=False,
            persistent_workers=(num_workers > 0),
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_variable_length,
            drop_last=False,
            persistent_workers=(num_workers > 0),
        )

        return train_loader, val_loader, test_loader

    @staticmethod
    def _build_train_sampler(train_ds, sampler_policy, dc):
        if sampler_policy == "none":
            return None

        n = len(train_ds.index)
        if n == 0:
            return None

        if sampler_policy == "full_history_weighted":
            target_fraction = float(dc.get("sampler_target_fraction", 0.3))
            is_full = train_ds.is_full_history_flags
            n_full = sum(is_full)
            n_other = n - n_full
            if n_full == 0 or n_other == 0:
                print("[Trainer] sampler_policy='full_history_weighted' requested but "
                      "no mix of full/partial-history samples exists; skipping sampler.")
                return None
            w_full = target_fraction * n_other / ((1.0 - target_fraction) * n_full)
            weights = [w_full if f else 1.0 for f in is_full]
            return WeightedRandomSampler(weights, num_samples=n, replacement=True)

        if sampler_policy == "tumor_balanced":
            is_pos = [e[4] for e in train_ds.index]
            n_pos = sum(is_pos)
            n_neg = n - n_pos
            if n_pos == 0 or n_neg == 0:
                print("[Trainer] sampler_policy='tumor_balanced' requested but train split "
                      "has only one class of target; skipping sampler.")
                return None
            w_pos, w_neg = 0.5 / n_pos, 0.5 / n_neg
            weights = [w_pos if p else w_neg for p in is_pos]
            return WeightedRandomSampler(weights, num_samples=n, replacement=True)

        raise ValueError(
            f"Unknown data.sampler_policy={sampler_policy!r}, "
            "expected 'none', 'full_history_weighted', or 'tumor_balanced'"
        )

    def train_epoch(self, loader, epoch):
        self.model.train()
        metrics = {
            "loss": 0.0,
            "mse_noise": 0.0,
            "n": 0,
        }

        unwrapped_for_flags = self.accelerator.unwrap_model(self.model)
        log_region_diag = (
            self.training_mode == "flow_matching"
            and getattr(unwrapped_for_flags, "flow_path", "standard") == "region_aware_source"
            and self.use_wandb
        )
        log_ntw_diag = (
            self.training_mode == "flow_matching"
            and getattr(unwrapped_for_flags, "flow_loss", "tumor_weighted") == "normalized_tumor_weighted"
            and self.use_wandb
        )

        for step, batch in enumerate(loader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            want_diag = (log_region_diag or log_ntw_diag) and (step + 1) % 100 == 0

            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()

                with self.accelerator.autocast():
                    out = self.model(batch, log_diagnostics=want_diag, epoch=epoch)

                self.accelerator.backward(out["loss"])

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 1.0
                    )
                self.optimizer.step()

                if self.accelerator.sync_gradients:
                    self.ema.update(self.accelerator.unwrap_model(self.model))

            if want_diag and "region_diag" in out:
                wandb.log({f"train/region/{k}": v for k, v in out["region_diag"].items()})
            if want_diag and "ntw_diag" in out:
                ntw_log = {f"train/ntw/{k}": v for k, v in out["ntw_diag"].items()}
                ntw_log["train/ntw/resumed_from_epoch"] = self.start_epoch
                wandb.log(ntw_log)

            metrics["loss"] += out["loss"].detach().item()
            metrics["mse_noise"] += out["mse_noise"].detach().item()
            metrics["n"] += 1

            if self.accelerator.is_main_process and (step + 1) % 100 == 0:
                n = max(metrics["n"], 1)
                print(
                    f"[Train][E{epoch}][local] "
                    f"step {step+1}/{len(loader)} "
                    f"loss={metrics['loss']/n:.4f} "
                    f"mse={metrics['mse_noise']/n:.4f} ",
                    flush=True,
                )

        n = max(metrics["n"], 1)
        local_avg = {k: v / n for k, v in metrics.items() if k != "n"}
        return self._sync_epoch_metrics(local_avg)

    def _sync_epoch_metrics(self, local_metrics: dict) -> dict:
        return {
            k: self.accelerator.reduce(torch.tensor(float(v), device=self.device), reduction="mean").item()
            for k, v in local_metrics.items()
        }
    
    @torch.no_grad()
    def validate_epoch(self, loader, epoch=None, use_ema=True):
        unwrapped = self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
            print(f"[Trainer.validate_epoch] weights_used={'ema' if use_ema else 'raw'}")
        if use_ema:
            self.ema.apply(unwrapped)

        unwrapped.eval()
        loss_chunks = []
        mse_chunks = []

        for step, batch in enumerate(loader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with self.accelerator.autocast():
                out = unwrapped(batch, epoch=epoch)

            loss_chunks.append(self.accelerator.gather_for_metrics(out["loss"].detach().reshape(1)))
            mse_chunks.append(self.accelerator.gather_for_metrics(out["mse_noise"].detach().reshape(1)))

            if self.accelerator.is_main_process and (step + 1) % 100 == 0:
                seen_loss = torch.cat(loss_chunks)
                seen_mse = torch.cat(mse_chunks)
                print(
                    f"[Val][global so far] "
                    f"step {step+1}/{len(loader)} "
                    f"loss={seen_loss.mean().item():.4f} "
                    f"mse={seen_mse.mean().item():.4f}",
                    flush=True,
                )

        if use_ema:
            self.ema.restore(unwrapped)

        all_loss = torch.cat(loss_chunks) if loss_chunks else torch.zeros(0, device=self.device)
        all_mse = torch.cat(mse_chunks) if mse_chunks else torch.zeros(0, device=self.device)
        n = max(int(all_loss.numel()), 1)

        return {
            "loss": all_loss.sum().item() / n,
            "mse_noise": all_mse.sum().item() / n,
        }
    

    def save_checkpoint(self, path, epoch, train_metrics=None, val_metrics=None):
        if self.training_mode == "diffusion":
            objective_cfg = self.cfg.get("diffusion", {})
        else:
            objective_cfg = self.cfg.get("flow_matching", {})

        dc = self.cfg.get("data", {})
        unwrapped = self.accelerator.unwrap_model(self.model)
        model_state = self.accelerator.get_state_dict(self.model)

        payload = {
            "epoch": epoch,
            "model": {k: v for k, v in model_state.items()
                      if not k.startswith(("vae.", "context_vae.", "context_encoder3d."))},
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.accelerator.scaler.state_dict() if self.accelerator.scaler is not None else None,
            "ema": self.ema.shadow,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "cfg": self.cfg,
            "training_mode": self.training_mode,
            "objective_cfg": objective_cfg,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "geno_preproc": self.geno_preproc.to_dict() if self.geno_preproc is not None else None,
            "image_stats": (
                {str(k): v.to_dict() for k, v in self.image_stats.items()}
                if self.image_stats is not None else None
            ),
            "normalization_policy": dc.get("normalization_policy", "whole_volume"),
            "normalization_clip_pct": tuple(dc.get("normalization_clip_pct", (0.5, 99.5))),
            "context_sampling": dc.get("context_sampling", "all_windows"),
            "volume_size": tuple(dc.get("volume_size", (160, 240, 240))),
            "ssim_definition": "gaussian_window_pytorch_msssim_win11_sigma1.5_data_range1.0",
            "representation": dc.get("representation", "latent"),
            "voxel_shape": tuple(dc.get("voxel_shape", (64, 96, 96))) if dc.get("representation") == "voxel" else None,
            "condition_on_treatment": unwrapped.condition_on_treatment,
            "temporal_aggregator_variant": unwrapped.temporal_aggregator_variant,
        }

        if self.accelerator.is_main_process:
            torch.save(payload, path)
            print(f"[Trainer] Saved: {path}")
        self.accelerator.wait_for_everyone()

    def load_checkpoint(self, path):
        print(f"[Trainer] Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location="cpu")

        ckpt_mode = ckpt.get("training_mode", "diffusion")
        if ckpt_mode != self.training_mode:
            raise ValueError(
                f"[Trainer] training_mode mismatch on resume: checkpoint '{path}' was trained "
                f"with training_mode='{ckpt_mode}', but this run was started with "
                f"training_mode='{self.training_mode}'. Diffusion and flow-matching checkpoints "
                f"are not interchangeable — the model predicts a different regression target "
                f"(epsilon vs. velocity) and uses a different sampler. Resume with "
                f"--training_mode {ckpt_mode}, or start a new run instead."
            )

        ckpt_schema_version = ckpt.get("schema_version", 1)
        if ckpt_schema_version < 3:
            raise ValueError(
                f"[Trainer] Checkpoint '{path}' predates schema v3 (found schema "
                f"v{ckpt_schema_version}) — it was trained under the OLD 2D pipeline "
                "(a 2D MONAI AutoencoderKL + Conv2d UNet). The 3D migration replaced the "
                "autoencoder with medvae_4_1_3d and the UNet with ConditionalUNet3D "
                "(Conv3d throughout) — these are architecturally incompatible with the old "
                "Conv2d weights (different parameter shapes/ranks entirely, not just a "
                "renamed key), so this checkpoint CANNOT be resumed. Start a fresh 3D "
                "training run instead."
            )

        ckpt_representation = ckpt.get("representation", "latent")
        self_representation = self.cfg.get("data", {}).get("representation", "latent")
        if ckpt_representation != self_representation:
            raise ValueError(
                f"[Trainer] representation mismatch on resume: checkpoint '{path}' was trained "
                f"with data.representation='{ckpt_representation}', but this run was started with "
                f"data.representation='{self_representation}'. Latent and voxel checkpoints are not "
                f"interchangeable — the model's VAE (real vs. identity pass-through), input spatial "
                f"resolution, and every conv weight shape at the finest UNet level(s) differ. Resume "
                f"with data.representation={ckpt_representation!r}, or start a new run instead."
            )

        ckpt_ta_variant = ckpt.get("temporal_aggregator_variant", "patch_temporal_spatial")
        if ckpt_ta_variant != self.model.temporal_aggregator_variant:
            raise ValueError(
                f"[Trainer] temporal_aggregator.variant mismatch on resume: checkpoint '{path}' "
                f"was trained with temporal_aggregator.variant={ckpt_ta_variant!r}, but this run "
                f"was started with temporal_aggregator.variant={self.model.temporal_aggregator_variant!r}. "
                "TemporalSpatialAggregator3D and PatchTemporalSpatialAggregator3D are not "
                "interchangeable — different parameter shapes and token layouts. Resume with "
                f"temporal_aggregator.variant={ckpt_ta_variant!r}, or start a new run instead."
            )

        ckpt_condition_on_treatment = bool(ckpt.get("condition_on_treatment", True))
        if ckpt_condition_on_treatment != self.model.condition_on_treatment:
            print(
                f"[Trainer] WARNING: condition_on_treatment mismatch on resume: checkpoint "
                f"'{path}' was trained with context.condition_on_treatment="
                f"{ckpt_condition_on_treatment}, but this run was started with "
                f"context.condition_on_treatment={self.model.condition_on_treatment}. This is "
                "allowed (e.g. deliberately ablating treatment conditioning mid-resume) — "
                "continuing with this run's configured value."
            )

        if self.training_mode == "flow_matching" and getattr(self.model, "flow_loss", None) == "normalized_tumor_weighted":
            ckpt_ntw_curriculum = (
                (ckpt.get("objective_cfg") or {}).get("normalized_tumor_weighted", {}).get("curriculum", {})
            )
            if ckpt_ntw_curriculum:
                ckpt_curr = (
                    bool(ckpt_ntw_curriculum.get("enabled", True)),
                    ckpt_ntw_curriculum.get("schedule", "cosine"),
                    float(ckpt_ntw_curriculum.get("start_epoch", 0)),
                    float(ckpt_ntw_curriculum.get("ramp_epochs", 1)),
                )
                active_curr = (
                    self.model.ntw_curriculum_enabled,
                    self.model.ntw_curriculum_schedule,
                    self.model.ntw_curriculum_start_epoch,
                    self.model.ntw_curriculum_ramp_epochs,
                )
                if ckpt_curr != active_curr:
                    print(
                        f"[Trainer] WARNING: normalized_tumor_weighted.curriculum config changed on "
                        f"resume: checkpoint '{path}' was trained with curriculum(enabled={ckpt_curr[0]}, "
                        f"schedule={ckpt_curr[1]!r}, start_epoch={ckpt_curr[2]:g}, ramp_epochs={ckpt_curr[3]:g}), "
                        f"but this run is configured with curriculum(enabled={active_curr[0]}, "
                        f"schedule={active_curr[1]!r}, start_epoch={active_curr[2]:g}, "
                        f"ramp_epochs={active_curr[3]:g}). curriculum_alpha is always recomputed from the "
                        "restored absolute epoch using THIS run's config, not stored in the checkpoint, so "
                        "this change can cause an abrupt jump in curriculum_alpha at the next epoch. This is "
                        "allowed (e.g. deliberately re-tuning the ramp), but make sure it's intentional."
                    )

        missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"[Trainer] Missing model keys: {missing}")
        if unexpected:
            print(f"[Trainer] Unexpected model keys: {unexpected}")

        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])

        if "scheduler" in ckpt and ckpt["scheduler"] is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])

        if "scaler" in ckpt and ckpt["scaler"] is not None and self.accelerator.scaler is not None:
            self.accelerator.scaler.load_state_dict(ckpt["scaler"])

        if "ema" in ckpt and ckpt["ema"] is not None:
            self.ema.shadow = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in ckpt["ema"].items()
            }

        self.geno_preproc = GenomicsPreprocessor.from_dict(ckpt.get("geno_preproc"))
        ckpt_image_stats = ckpt.get("image_stats")
        self.image_stats = (
            {int(k): ImageIntensityStats.from_dict(v) for k, v in ckpt_image_stats.items()}
            if ckpt_image_stats else None
        )
        self._resumed_normalization_policy = ckpt.get("normalization_policy")
        self._resumed_context_sampling = ckpt.get("context_sampling")

        self.start_epoch = int(ckpt["epoch"]) + 1
        print(f"[Trainer] Resuming from epoch {self.start_epoch}")

        if self.training_mode == "flow_matching" and getattr(self.model, "flow_loss", None) == "normalized_tumor_weighted":
            resumed_completed_epochs = self.start_epoch - 1
            resumed_progress = (
                compute_curriculum_progress(
                    resumed_completed_epochs,
                    start_epoch=self.model.ntw_curriculum_start_epoch,
                    ramp_epochs=self.model.ntw_curriculum_ramp_epochs,
                )
                if self.model.ntw_curriculum_enabled
                else 1.0
            )
            resumed_alpha = compute_curriculum_alpha(
                resumed_completed_epochs,
                enabled=self.model.ntw_curriculum_enabled,
                schedule=self.model.ntw_curriculum_schedule,
                start_epoch=self.model.ntw_curriculum_start_epoch,
                ramp_epochs=self.model.ntw_curriculum_ramp_epochs,
            )
            print(
                f"[Trainer] Resumed at absolute epoch {self.start_epoch}. Tumor-weight curriculum: "
                f"start={self.model.ntw_curriculum_start_epoch:g}, ramp={self.model.ntw_curriculum_ramp_epochs:g}, "
                f"progress={resumed_progress:.4f}, alpha={resumed_alpha:.4f}."
            )

    def fit(self):
        tc = self.cfg["training"]
        train_loader, val_loader, _test_loader = self.build_dataloaders()
        train_loader, val_loader = self.accelerator.prepare(train_loader, val_loader)
        per_gpu_bs = int(self.cfg.get("data", {}).get("batch_size", 1))
        effective_bs = per_gpu_bs * self.accelerator.num_processes * self.grad_accum
        if self.accelerator.is_main_process:
            print(
                f"[Trainer] {tc['epochs']} epochs, {len(train_loader)} batches/epoch (per-process) "
                f"| effective batch size = {per_gpu_bs} (per-GPU) x {self.accelerator.num_processes} "
                f"(processes) x {self.grad_accum} (accumulation steps) = {effective_bs}"
            )

        best_mode = tc.get("best_metric", "dice")

        for epoch in range(self.start_epoch, tc["epochs"] + 1):
            t0 = time.time()

            train_metrics = self.train_epoch(train_loader, epoch)
            self.scheduler.step()

            val_metrics = self.validate_epoch(val_loader, epoch, use_ema=True)

            lr = self.scheduler.get_last_lr()[0]
            dt = time.time() - t0

            if self.accelerator.is_main_process:
                print(
                    f"[E{epoch}/{tc['epochs']}] "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"train_mse={train_metrics['mse_noise']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_mse={val_metrics['mse_noise']:.4f} "
                    f"time={dt:.1f}s lr={lr:.2e}"
                )

            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "lr": lr,

                    "train/loss": train_metrics["loss"],
                    "train/mse": train_metrics["mse_noise"],

                    "val/loss": val_metrics["loss"],
                    "val/mse": val_metrics["mse_noise"],

                    "time/epoch_sec": dt,
                })


            if epoch % tc.get("checkpoint_every", 10) == 0:
                numbered_path = os.path.join(self.log_dir, f"ckpt_e{epoch:04d}.pt")
                self.save_checkpoint(numbered_path, epoch, train_metrics, None)


        if self.use_wandb:
            wandb.finish()
            
    @torch.no_grad()
    def evaluate(self, loader, ema_policy: str = "auto"):
        unwrapped = self.accelerator.unwrap_model(self.model)
        has_ema = bool(self.ema.shadow)
        if ema_policy == "force_ema" and not has_ema:
            raise RuntimeError(
                "[Trainer.evaluate] ema_policy='force_ema' but this model has no EMA shadow "
                "state (empty). Refusing to silently fall back to raw weights — pass "
                "ema_policy='force_raw' explicitly if that's intended."
            )
        use_ema = has_ema and ema_policy in ("auto", "force_ema")
        if self.accelerator.is_main_process:
            print(f"[Trainer.evaluate] weights_used={'ema' if use_ema else 'raw'} (ema_policy={ema_policy})")

        if use_ema:
            self.ema.apply(unwrapped)
        unwrapped.eval()

        mse_sum = 0.0
        dice_sum = 0.0
        n = 0

        if self.training_mode == "diffusion":
            sampler_kwargs = dict(
                ddim_steps=self.cfg["training"].get("eval_ddim_steps", 50),
                ddim_eta=0.0,
            )
        else:
            fm_cfg = self.cfg.get("flow_matching", {})
            sampler_kwargs = dict(
                fm_steps=self.cfg["training"].get("eval_fm_steps", fm_cfg.get("num_steps", 50)),
                fm_solver=self.cfg["training"].get("eval_fm_solver", fm_cfg.get("solver", "euler")),
            )

        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            out = unwrapped.generate(
                input_images=batch["input_images"],
                input_days=batch["input_days"],
                input_treatments=batch["input_treatments"],
                input_mask=batch["input_mask"],
                genomics=batch.get("geno"),
                geno_mask=batch.get("geno_mask"),
                target_day=batch["target_day"],
                target_treatment=batch["target_treatment"],
                input_visit_keys=batch.get("input_visit_keys"),
                **sampler_kwargs,
            )

            pred_img = out["generated_mri"]
            gt_img = batch["target_image"]
            mse = torch.mean((pred_img - gt_img) ** 2).item()

            dice = 0.0
            if unwrapped.seg_enabled and out.get("generated_seg", None) is not None:
                pred_seg = out["generated_seg"]
                gt_seg = batch["target_label"]
                if gt_seg.ndim == 3:
                    gt_seg = gt_seg.unsqueeze(1)
                gt_seg = gt_seg.float()

                pred_bin = (pred_seg > 0.5).float()
                inter = (pred_bin * gt_seg).sum(dim=(1, 2, 3))
                denom = pred_bin.sum(dim=(1, 2, 3)) + gt_seg.sum(dim=(1, 2, 3))
                dice = ((2 * inter + 1.0) / (denom + 1.0)).mean().item()

            mse_sum += mse
            dice_sum += dice
            n += 1

        if use_ema:
            self.ema.restore(unwrapped)

        if n == 0:
            return {"mse": 0.0, "dice": 0.0, "weights_used": "ema" if use_ema else "raw"}
        return {
            "mse": mse_sum / n,
            "dice": dice_sum / n,
            "weights_used": "ema" if use_ema else "raw",
        }

    def final_test(self, test_loader, ema_policy: str = "auto"):
        opt_state_before = self.optimizer.state_dict()
        sched_state_before = self.scheduler.state_dict()
        ema_shadow_before = {k: v.clone() for k, v in self.ema.shadow.items()}

        if self.accelerator.is_main_process:
            print("[Trainer.final_test] Evaluating held-out TEST split (read-only).")
        results = self.evaluate(test_loader, ema_policy=ema_policy)

        assert self.optimizer.state_dict().keys() == opt_state_before.keys()
        assert self.scheduler.state_dict() == sched_state_before
        assert set(self.ema.shadow.keys()) == set(ema_shadow_before.keys())
        for k in ema_shadow_before:
            assert torch.equal(self.ema.shadow[k], ema_shadow_before[k]), (
                f"[Trainer.final_test] EMA shadow mutated by evaluation for key {k} — this must never happen."
            )

        if self.accelerator.is_main_process:
            print(f"[Trainer.final_test] Results: {results}")
        return results