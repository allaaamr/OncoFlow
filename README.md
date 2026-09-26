

## Quick Start

###  Install dependencies
```bash
pip install torch monai monai-generative einops pyyaml huggingface_hub accelerate
```


### Train
Re
Two training objectives are supported via `--training_mode` 
```bash
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


## Multi-GPU Training (Hugging Face Accelerate)

`scripts/train.py`/`src/training/trainer.py` use [Accelerate](https://huggingface.co/docs/accelerate)
for data-parallel (DDP) training: the SAME entry point and config/`--override`
mechanism as single-GPU, launched via `accelerate launch` (or `torchrun`)
instead of plain `python`. No code changes are needed to go from 1 GPU to N —
GPU count/precision/accumulation are all driven by the launcher flags and the
existing `training:` config block, never hardcoded.

```bash
accelerate launch --num_processes 1 scripts/train.py --config configs/mu_voxel_curric.yaml --training_mode flow_matching

accelerate launch --multi_gpu --num_processes 4 scripts/train.py \
    --config configs/mu_voxel_curric.yaml --training_mode flow_matching \
    --override training.gradient_accumulation_steps=1
```

Data parallelism only — no model/pipeline/spatial (patch) parallelism, each
process trains the same full model on whole volumes. Accelerate shards
DataLoader batches across processes directly (no extra `DistributedSampler`),
keeps flow-matching noise/timestep draws independent per process, and the
LR scheduler still steps once per EPOCH exactly as it did on a single GPU


## Training Modes: Diffusion vs. Flow Matching

Both modes train the *same* conditional model (frozen VAE → per-visit context
binding → spatio-temporal aggregation → conditional UNet) on the *same*
conditioning inputs (longitudinal imaging history, treatment metadata, genomics)
— only the objective the UNet regresses onto, and the sampler used to invert it,
differ. Set the mode with `training.training_mode` in the config or `--training_mode`
on the CLI; `diffusion` is the default and its recipe (schedule, loss weighting,
sampling, checkpoint format) is unchanged from before this feature was added.

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

## Data Format

Same `.npy` files as before:
```
PatientID_XXXX_image.npy     : (S*C, D, H, W)
PatientID_XXXX_label.npy     : (S, D, H, W)
PatientID_XXXX_days.npy      : (S,)
PatientID_XXXX_treatment.npy : (S,)
PatientID_XXXX_geno.npy      : (G,)  optional
```
