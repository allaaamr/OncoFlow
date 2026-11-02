import torch
import torch.nn as nn


class FrozenMedVAE3D(nn.Module):

    def __init__(self, mvae, latent_channels: int = 1, downsample_factor: int = 4):
        super().__init__()
        self.mvae = mvae
        self.latent_channels = latent_channels
        self.downsample_factor = downsample_factor
        self._freeze()

    def _freeze(self):
        for param in self.mvae.parameters():
            param.requires_grad = False
        self.mvae.eval()

    def train(self, mode=True):
        super().train(mode)
        self.mvae.eval()
        return self

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        from monai.inferers import sliding_window_inference
        from medvae.utils.extras import roi_size_calc

        x = (x - 0.5) / 0.5

        def predict_mode(patch: torch.Tensor) -> torch.Tensor:
            posterior = self.mvae.model.encode(patch)
            return posterior.mode()

        roi_size = roi_size_calc(x.shape[-3:], target_gpu_dim=self.mvae.gpu_dim)
        z = sliding_window_inference(
            inputs=x,
            roi_size=roi_size,
            sw_batch_size=1,
            mode="gaussian",
            predictor=predict_mode,
        )
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        recon = self.mvae.decode(z)
        return recon * 0.5 + 0.5

    @classmethod
    def from_config(cls, vae_cfg: dict) -> "FrozenMedVAE3D":
        from medvae import MVAE

        model_name = vae_cfg.get("model_name", "medvae_4_1_3d")
        modality = vae_cfg.get("modality", "mri")
        gpu_dim = int(vae_cfg.get("gpu_dim", 160))

        mvae = MVAE(model_name, modality, gpu_dim)
        mvae.requires_grad_(False)
        mvae.eval()

        print(f"[VAE] Loaded medvae '{model_name}' (modality={modality}, gpu_dim={gpu_dim})")

        latent_channels = int(vae_cfg.get("latent_channels", 1))
        downsample_factor = int(vae_cfg.get("downsample_factor", int(model_name.split("_")[1])))
        return cls(mvae, latent_channels=latent_channels, downsample_factor=downsample_factor)


class FlairOnlyEncoder3D(nn.Module):

    def __init__(self, vae: FrozenMedVAE3D, flair_index: int = 0, strict: bool = True):
        super().__init__()
        self.vae = vae
        self.flair_index = flair_index
        self.strict = strict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected 5D tensor (B,C,D,H,W), got shape {tuple(x.shape)}")

        B, C, D, H, W = x.shape

        if C == 1:
            x_flair = x
        else:
            if self.strict and not (0 <= self.flair_index < C):
                raise ValueError(
                    f"flair_index={self.flair_index} out of range for input with C={C}"
                )
            x_flair = x[:, self.flair_index:self.flair_index + 1, :, :, :]

        return self.vae.encode(x_flair)


class _DummyVAE3D(nn.Module):

    def __init__(self, latent_channels: int = 1, downsample_factor: int = 4):
        super().__init__()
        self.latent_channels = latent_channels
        self.downsample_factor = downsample_factor
        self.encoder = nn.Conv3d(1, latent_channels, kernel_size=downsample_factor, stride=downsample_factor)
        self.decoder = nn.ConvTranspose3d(latent_channels, 1, kernel_size=downsample_factor, stride=downsample_factor)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def build_dummy_vae3d(vae_cfg: dict) -> "_DummyVAE3D":
    return _DummyVAE3D(
        latent_channels=int(vae_cfg.get("latent_channels", 1)),
        downsample_factor=int(vae_cfg.get("downsample_factor", 4)),
    )


class IdentityVoxelRepresentation3D(nn.Module):

    def __init__(self, latent_channels: int = 1, downsample_factor: int = 1):
        super().__init__()
        self.latent_channels = latent_channels
        self.downsample_factor = downsample_factor

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z


def build_identity_voxel_representation(vae_cfg: dict) -> "IdentityVoxelRepresentation3D":
    return IdentityVoxelRepresentation3D(
        latent_channels=int(vae_cfg.get("latent_channels", 1)),
        downsample_factor=int(vae_cfg.get("downsample_factor", 1)),
    )
