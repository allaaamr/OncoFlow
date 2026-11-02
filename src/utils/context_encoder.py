import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.ndim == 2:
            t = t.squeeze(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TreatmentEncoder(nn.Module):
    def __init__(self, vocab_size: int = 20, embed_dim: int = 64, continuous_dim: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.continuous_dim = continuous_dim
        if continuous_dim > 0:
            self.continuous_proj = nn.Linear(continuous_dim, embed_dim)
            self.fuse = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, treatment_id, continuous_feats=None):
        e = self.embed(treatment_id)
        if self.continuous_dim > 0 and continuous_feats is not None:
            e = self.fuse(torch.cat([e, self.continuous_proj(continuous_feats)], dim=-1))
        return e


class PerVisitContextEncoder(nn.Module):

    def __init__(
        self,
        time_embed_dim=128,
        treatment_vocab_size=20,
        treatment_embed_dim=64,
        treatment_continuous_dim=0,
        context_dim=256,
        condition_on_treatment=True,
    ):
        super().__init__()

        self.condition_on_treatment = condition_on_treatment
        self.treatment_embed_dim = treatment_embed_dim
        self.time_emb = SinusoidalTimeEmbedding(time_embed_dim)
        self.treat_enc = TreatmentEncoder(
            treatment_vocab_size,
            treatment_embed_dim,
            treatment_continuous_dim,
        )

        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim + treatment_embed_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )

    def forward(self, delta_t, treatment=None, continuous_feats=None):
        t_emb = self.time_emb(delta_t)
        if self.condition_on_treatment:
            tr_emb = self.treat_enc(treatment, continuous_feats)
        else:
            tr_emb = t_emb.new_zeros(t_emb.shape[0], self.treatment_embed_dim)
        return self.mlp(torch.cat([t_emb, tr_emb], dim=-1))


class AdaptiveGroupNorm3D(nn.Module):

    def __init__(self, num_channels: int, context_dim: int, num_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(min(num_groups, num_channels), num_channels)
        self.proj = nn.Linear(context_dim, 2 * num_channels)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.norm(z)
        scale, shift = self.proj(c).chunk(2, dim=-1)

        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]

        return h * (1.0 + scale) + shift


class VisitContextBinder(nn.Module):

    def __init__(
        self,
        latent_channels,
        time_embed_dim=128,
        treatment_vocab_size=20,
        treatment_embed_dim=64,
        treatment_continuous_dim=0,
        context_dim=256,
        binding_mode="adagn",
        num_heads=4,
        dropout=0.0,
        condition_on_treatment=True,
        bind_context_to_images=True,
    ):
        super().__init__()

        if binding_mode != "adagn":
            raise ValueError(
                f"binding_mode={binding_mode!r} is not supported post-3D-migration — only "
                "'adagn' was ported (see module docstring); 'cross_attn'/'adagn_cross_attn' "
                "were dropped because configs/default.yaml never exercised them."
            )
        self.binding_mode = binding_mode
        self.treatment_continuous_dim = treatment_continuous_dim
        self.condition_on_treatment = condition_on_treatment
        self.bind_context_to_images = bind_context_to_images

        if self.bind_context_to_images:
            self.context_encoder = PerVisitContextEncoder(
                time_embed_dim=time_embed_dim,
                treatment_vocab_size=treatment_vocab_size,
                treatment_embed_dim=treatment_embed_dim,
                treatment_continuous_dim=treatment_continuous_dim,
                context_dim=context_dim,
                condition_on_treatment=condition_on_treatment,
            )

            self.ada_gn = AdaptiveGroupNorm3D(
                num_channels=latent_channels,
                context_dim=context_dim,
            )

    def forward(
        self,
        z_list,
        days,
        treatments=None,
        mask=None,
        continuous_feats=None,
    ):

        N = len(z_list)
        z_bound = []

        for i in range(N):
            z_i = z_list[i]

            if self.bind_context_to_images:
                cont_i = None
                if continuous_feats is not None:
                    cont_i = continuous_feats[:, i]

                treatment_i = None
                if self.condition_on_treatment and treatments is not None:
                    treatment_i = treatments[:, i]

                c_i = self.context_encoder(
                    days[:, i],
                    treatment_i,
                    cont_i,
                )
                z_i = self.ada_gn(z_i, c_i)

            if mask is not None:
                valid = mask[:, i].float()[:, None, None, None, None]
                z_i = z_i * valid

            z_bound.append(z_i)

        return z_bound
