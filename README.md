# 2D Longitudinal MRI Prediction via Conditional Latent Diffusion

## What Changed from 3D → 2D

| Aspect | 3D Version | 2D Version |
|---|---|---|
| Input | Full (C, D, H, W) volumes | (C, H, W) axial slices from tumor region |
| VAE | Custom 3D VAE (train yourself) | **Pretrained MONAI BraTS 2D** (just download) |
| Convolutions | Conv3d everywhere | Conv2d everywhere |
| Batch size | 1-2 (memory limited) | 8-32 (much faster training) |
| Spatial attention | 3D tokens (d×h×w) | 2D tokens (h×w) — much cheaper |
| Slice selection | N/A | Tumor-region axial slices only |

## Architecture

```
Input: N patient visits, each with (MRI, date, treatment)
Target: Future MRI at (target_date, target_treatment)

Pipeline:
  ┌──────────────┐
  │ Frozen 3D VAE│──→ z_1, z_2, ..., z_N  (latent representations)
  │   (encoder)  │──→ z_target             (only during training)
  └──────────────┘
         │
         ▼
  ┌───────────────────┐
  │ Per-Visit Context │  Each z_i modulated by (Δt_i, treatment_i)
  │ Binding (AdaGN)   │  via Adaptive Group Normalization
  └───────────────────┘
         │
         ▼
  ┌───────────────────┐
  │ Spatio-Temporal   │  Factored attention:
  │ Aggregator (SADM) │  spatial self-attn + temporal self-attn
  └───────┬───────────┘
          │
    ┌─────┴─────┐
    │           │
 h_temporal   Z_agg
 (summary)  (full seq)
    │           │
    ▼           │
  ┌─────────────┴──────────────┐
  │   Conditional 3D UNet      │  ← Global cond: (t, target_day, target_treatment)
  │                            │
  │  • Input: [z_t; h_temporal]│  (concat)
  │  • Cross-attn kv: Z_agg    │ (attention)
  │  • AdaGN: global_cond      │ (modulation)
  └────────────┬───────────────┘
               │
         ε_pred / z_0_pred
               │
        ┌──────┴──────┐
        │             │
   L_diffusion    Seg Head → L_seg
```

## Quick Start

### Step 1: Install dependencies
```bash
pip install torch monai monai-generative einops pyyaml huggingface_hub accelerate
```

### Step 2: Download the pretrained VAE
```bash
python setup_vae.py
```
This downloads the MONAI BraTS 2D AutoencoderKL from HuggingFace (~200MB)
and verifies it works. It produces:
```
Input:  (1, 1, 240, 240)
Latent: (1, 3, 60, 60)    ← 4× spatial downsampling, 3 latent channels
Recon:  (1, 1, 240, 240)
```

### Step 3: Update config
Edit `configs/default.yaml`:
```yaml
data:
  data_dir: "/path/to/your/patient_npy_files"
vae:
  checkpoint: "./monai_vae_bundle/models/model_autoencoder.pt"
```

### Step 4: Train

Two training objectives are supported via `--training_mode` (default: `diffusion`,
so all existing commands/configs/checkpoints keep working unchanged):

```bash
# Diffusion (default — original DDPM/DDIM recipe, unchanged)
python train.py --config configs/default.yaml --training_mode diffusion

# Flow matching (continuous-time conditional flow matching)
python train.py --config configs/default.yaml --training_mode flow_matching

# Override from command line:
python train.py --config configs/default.yaml --override data.batch_size=16 training.lr=3e-5
```

Resuming: `--resume <ckpt>` requires the checkpoint's `training_mode` to match the
current run's — diffusion and flow-matching checkpoints are not interchangeable
(different regression target, different sampler), and a mismatch raises a clear error
instead of silently loading incompatible weights.

See **Training Modes: Diffusion vs. Flow Matching** below for the sampling commands,
solver settings, and the underlying math.

## Multi-GPU Training (Hugging Face Accelerate)

`scripts/train.py`/`src/training/trainer.py` use [Accelerate](https://huggingface.co/docs/accelerate)
for data-parallel (DDP) training: the SAME entry point and config/`--override`
mechanism as single-GPU, launched via `accelerate launch` (or `torchrun`)
instead of plain `python`. No code changes are needed to go from 1 GPU to N —
GPU count/precision/accumulation are all driven by the launcher flags and the
existing `training:` config block, never hardcoded.

```bash
# 1 GPU (unchanged — accelerate launch with num_processes=1 is equivalent to
# plain `python scripts/train.py ...`):
accelerate launch --num_processes 1 scripts/train.py --config configs/mu_voxel_curric.yaml --training_mode flow_matching

# 4 GPUs, one whole volume per GPU (data.batch_size stays 1 — PER-PROCESS batch
# size, unchanged from the single-GPU config), no gradient accumulation ->
# effective batch size = 1 x 4 x 1 = 4:
accelerate launch --multi_gpu --num_processes 4 scripts/train.py \
    --config configs/mu_voxel_curric.yaml --training_mode flow_matching \
    --override training.gradient_accumulation_steps=1

# BF16 mixed precision (any GPU count) — training.mixed_precision accepts the
# legacy bool (true -> fp16, unchanged default) or "no"/"fp16"/"bf16":
accelerate launch --multi_gpu --num_processes 4 scripts/train.py \
    --config configs/mu_voxel_curric.yaml --training_mode flow_matching \
    --override training.mixed_precision=bf16

# Resuming a checkpoint (same --resume flag, any GPU count):
accelerate launch --multi_gpu --num_processes 4 scripts/train.py \
    --config configs/mu_voxel_curric.yaml --training_mode flow_matching \
    --resume ./new_ckpts/.../ckpt_e0100.pt
```

Data parallelism only — no model/pipeline/spatial (patch) parallelism, each
process trains the same full model on whole volumes. Accelerate shards
DataLoader batches across processes directly (no extra `DistributedSampler`),
keeps flow-matching noise/timestep draws independent per process, and the
LR scheduler still steps once per EPOCH exactly as it did on a single GPU
(unaffected by process count or accumulation). See `src/training/trainer.py`'s
`Trainer.__init__`/`train_epoch`/`validate_epoch` for the full rationale.

## Training Modes: Diffusion vs. Flow Matching

Both modes train the *same* conditional model (frozen VAE → per-visit context
binding → spatio-temporal aggregation → conditional UNet) on the *same*
conditioning inputs (longitudinal imaging history, treatment metadata, genomics)
— only the objective the UNet regresses onto, and the sampler used to invert it,
differ. Set the mode with `training.training_mode` in the config or `--training_mode`
on the CLI; `diffusion` is the default and its recipe (schedule, loss weighting,
sampling, checkpoint format) is unchanged from before this feature was added.

### Diffusion (default) — `src/utils/diffusion.py`

Standard DDPM forward process with a cosine β-schedule and ε-prediction:

```
t ~ Uniform{0, ..., T-1}                         (T = diffusion.num_timesteps, default 1000)
x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε,  ε ~ N(0, I)
loss = E[ w(x_0) * || ε_θ(x_t, t, cond) - ε ||² ]
```
Sampling: DDIM (`diffusion.ddim_sample`), steps/η exposed via `--override diffusion.ddim_steps=...`
or `eval_ddim_steps` in `training:`.

### Flow Matching — `src/utils/flow_matching.py`

Continuous-time conditional flow matching (Lipman et al., 2022) with the
straight-line conditional optimal-transport / rectified-flow probability path
between Gaussian noise and the target VAE latent:

```
t ~ Uniform[0, 1]
x_0 ~ N(0, I)                                    (noise)
x_1 = target latent (VAE-encoded FLAIR slice)
x_t = (1 - t) x_0 + t x_1                        (straight-line path)
u_t = x_1 - x_0                                  (target velocity, constant along the path)
loss = E[ w(x_0) * || v_θ(x_t, t, cond) - u_t ||² ]
```
The same tumor-region loss weighting `w(x_0)` used by the diffusion path is reused
here for parity between the two objectives.

`t ∈ [0, 1]` is rescaled to `flow_matching.time_scale` (default 999, matching
diffusion's integer-timestep range) before being fed to the existing sinusoidal
timestep embedding — no architecture change, just a numeric-range match so the
shared `GlobalConditionEncoder`/UNet see time values in the range they were designed for.

Sampling integrates the learned ODE `dx/dt = v_θ(x_t, t, cond)` from `x_0 ~ N(0, I)`
at `t=0` to `x_1` at `t=1`, via `flow_matching.num_steps` steps (default 50) of
`flow_matching.solver` (`"euler"` or 2nd-order `"heun"`), overridable per-run
via `fm_steps`/`fm_solver`.

```bash
python train.py --config configs/default.yaml --training_mode flow_matching \
    --override flow_matching.num_steps=50 flow_matching.solver=euler
```

Config keys (`configs/default.yaml`):
```yaml
training:
  training_mode: "diffusion"   # or "flow_matching"
flow_matching:
  sigma_min: 0.0        # 0.0 = pure conditional-OT / rectified-flow path
  time_scale: 999.0      # rescales t in [0,1] into the shared embedding's numeric range
  num_steps: 50           # ODE integration steps for sampling
  solver: "euler"         # "euler" or "heun"
```

### Caveats / assumptions

- The straight-line conditional-OT path (`sigma_min=0`) was chosen as the standard,
  simplest flow-matching path per the task; a positive `sigma_min` (min-variance path)
  is implemented but untested against this dataset.
- Guidance behavior (classifier-free conditioning dropout, `cond_dropout_prob`) is
  applied identically upstream of both objectives — dropped conditioning is baked
  into the training batch before either sampler-specific code runs.
- Euler is a first-order ODE solver; Heun (2nd-order) roughly doubles per-step
  compute for a lower discretization error at the same step count — prefer Heun
  when `num_steps` is small.

## About the MONAI BraTS 2D VAE

The pretrained VAE from `MONAI/brats_mri_axial_slices_generative_diffusion`:

- **Architecture**: `AutoencoderKL(spatial_dims=2, in_channels=1, out_channels=1, num_channels=(128,128,256), latent_channels=3)`
- **Trained on**: BraTS 2016+2017 axial Flair slices (38,800 slices from 388 volumes)
- **Input**: Single-channel 240×240 axial MRI slices
- **Latent**: 3 channels × 60×60 spatial (4× downsampling)
- **Losses used**: L1 + perceptual (ResNet50) + KL divergence + adversarial (PatchGAN)

### Multi-Modality Handling

The VAE expects 1 channel. For multi-modality (T1, T1ce, FLAIR), we:
1. Encode each modality separately: `(B, 1, 240, 240) → (B, 3, 60, 60)` × 3 modalities
2. Concatenate: `(B, 9, 60, 60)`
3. Fuse with learned 1×1 conv: `(B, 3, 60, 60)`

The fusion conv is the only trainable part touching latents — the VAE stays frozen.

## Tumor-Region Slice Extraction

Instead of random slices, the dataset finds slices that contain tumor:

1. Load segmentation masks for ALL sessions of a patient
2. Union the masks: `tumor_any = (labels > 0).any(axis=sessions)`
3. Find axial indices where `tumor_any[:, :].any()` is True
4. Sample from these indices

This ensures training focuses on the clinically relevant region where tumor evolution happens.

## Data Format

Same `.npy` files as before:
```
PatientID_XXXX_image.npy     : (C*S, D, H, W)
PatientID_XXXX_label.npy     : (S, D, H, W)
PatientID_XXXX_days.npy      : (S,)
PatientID_XXXX_treatment.npy : (S,)
PatientID_XXXX_geno.npy      : (G,)  optional
```

The dataset handles the 3D→2D slice extraction internally.

## Project Structure

```
longitudinal_diffusion/
├── configs/default.yaml           # All hyperparameters
├── setup_vae.py                   # Download + verify MONAI VAE
├── train.py                       # Entry point
└── src/
    ├── data/dataset.py            # PatientSliceDataset (tumor-region 2D extraction)
    ├── models/
    │   ├── vae_wrapper.py         # FrozenVAE2D + MultiModalityEncoder2D
    │   ├── pipeline.py            # LongitudinalDiffusion2DPipeline
    │   └── unet2d.py              # Conditional 2D UNet
    ├── utils/
    │   ├── context_encoder.py     # Time/treatment → AdaGN binding (2D)
    │   ├── temporal_aggregator.py # Spatio-temporal attention (2D)
    │   ├── seg_head.py            # 2D segmentation head + loss
    │   ├── diffusion.py           # DDPM/DDIM (dimension-agnostic)
    │   └── flow_matching.py       # Conditional flow matching (dimension-agnostic)
    └── training/trainer.py        # Training loop (mode-aware checkpointing)

tests/test_training_modes.py       # Smoke tests for both training modes
```
