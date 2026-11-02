import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SpatialSelfAttention3D(nn.Module):

    def __init__(self, channels: int, num_heads: int = 8, spatial_downsample: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.spatial_ds = spatial_downsample
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BN, C, d, h, w = x.shape
        residual = x
        x = self.norm(x)

        if self.spatial_ds > 1:
            ds, hs, ws = max(1, d // self.spatial_ds), max(1, h // self.spatial_ds), max(1, w // self.spatial_ds)
            x_ds = F.adaptive_avg_pool3d(x, (ds, hs, ws))
        else:
            x_ds = x
            ds, hs, ws = d, h, w

        tokens = rearrange(x_ds, "b c d h w -> b (d h w) c")
        qkv = self.qkv(tokens)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "b s (nh d) -> b nh s d", nh=self.num_heads)
        k = rearrange(k, "b s (nh d) -> b nh s d", nh=self.num_heads)
        v = rearrange(v, "b s (nh d) -> b nh s d", nh=self.num_heads)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = rearrange(attn, "b nh s d -> b s (nh d)")
        out = self.proj_out(attn)
        out = rearrange(out, "b (d h w) c -> b c d h w", d=ds, h=hs, w=ws)

        if self.spatial_ds > 1:
            out = F.interpolate(out, size=(d, h, w), mode="trilinear", align_corners=False)
        return residual + out


class TemporalSelfAttention(nn.Module):

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj_out = nn.Linear(channels, channels)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C, d, h, w = x.shape
        residual = x

        x_flat = rearrange(x, "b n c d h w -> (b n) c d h w")
        x_flat = self.norm(x_flat)
        x = rearrange(x_flat, "(b n) c d h w -> b n c d h w", b=B, n=N)

        x = rearrange(x, "b n c d h w -> (b d h w) n c")
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "bs n (nh d) -> bs nh n d", nh=self.num_heads)
        k = rearrange(k, "bs n (nh d) -> bs nh n d", nh=self.num_heads)
        v = rearrange(v, "bs n (nh d) -> bs nh n d", nh=self.num_heads)

        attn_mask = None
        if mask is not None:
            S = d * h * w
            attn_mask = mask[:, None, None, :].float()
            attn_mask = attn_mask.repeat_interleave(S, dim=0)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf"))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = rearrange(attn, "bs nh n d -> bs n (nh d)")
        out = self.proj_out(attn)
        out = rearrange(out, "(b d h w) n c -> b n c d h w", b=B, d=d, h=h, w=w)
        return residual + out


class SpatioTemporalBlock3D(nn.Module):

    def __init__(self, channels: int, num_heads: int = 8, spatial_downsample: int = 4):
        super().__init__()
        self.spatial_attn = SpatialSelfAttention3D(channels, num_heads, spatial_downsample)
        self.temporal_attn = TemporalSelfAttention(channels, num_heads)
        self.ffn = nn.Sequential(
            nn.GroupNorm(min(32, channels), channels),
            nn.Conv3d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Conv3d(channels * 4, channels, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C, d, h, w = x.shape
        x_flat = rearrange(x, "b n c d h w -> (b n) c d h w")
        x_flat = self.spatial_attn(x_flat)
        x = rearrange(x_flat, "(b n) c d h w -> b n c d h w", b=B, n=N)
        x = self.temporal_attn(x, mask=mask)
        x_flat = rearrange(x, "b n c d h w -> (b n) c d h w")
        x_flat = x_flat + self.ffn(x_flat)
        return rearrange(x_flat, "(b n) c d h w -> b n c d h w", b=B, n=N)


class TemporalSpatialAggregator3D(nn.Module):

    def __init__(self, channels, num_layers=4, num_heads=8,
                 spatial_downsample=4, aggregation="learned_query"):
        super().__init__()
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock3D(channels, num_heads, spatial_downsample)
            for _ in range(num_layers)
        ])
        self.aggregation = aggregation
        if aggregation == "learned_query":
            self.temporal_query = nn.Parameter(torch.randn(1, 1, channels))
            self.query_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
            self.query_norm = nn.LayerNorm(channels)

    def forward(self, z_bound_list, mask=None):
        Z = torch.stack(z_bound_list, dim=1)
        for block in self.blocks:
            Z = block(Z, mask=mask)
        Z_agg = Z
        B, N, C, D, H, W = Z.shape

        if self.aggregation == "mean_pool":
            if mask is not None:
                m = mask[:, :, None, None, None, None].float()
                h_temporal = (Z * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                h_temporal = Z.mean(1)
        elif self.aggregation == "learned_query":
            Z_spatial = rearrange(Z, "b n c d h w -> (b d h w) n c")
            query = self.temporal_query.expand(Z_spatial.shape[0], -1, -1)
            key_pad = None
            if mask is not None:
                S = D * H * W
                key_pad = ~mask
                key_pad = key_pad.repeat_interleave(S, dim=0)
            pooled, _ = self.query_attn(query, Z_spatial, Z_spatial, key_padding_mask=key_pad)
            pooled = self.query_norm(pooled.squeeze(1))
            h_temporal = rearrange(pooled, "(b d h w) c -> b c d h w", b=B, d=D, h=H, w=W)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return Z_agg, h_temporal


class PatchTemporalSpatialAggregator3D(nn.Module):

    def __init__(
        self,
        channels,
        num_layers=4,
        num_heads=8,
        patch_size=(4, 16, 16),
        spatial_downsample=1,
        aggregation="learned_query",
    ):
        super().__init__()

        self.channels = channels
        self.patch_size = tuple(patch_size)
        self.aggregation = aggregation

        self.patch_embed = nn.Conv3d(
            channels,
            channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.patch_decode = nn.ConvTranspose3d(
            channels,
            channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.blocks = nn.ModuleList([
            SpatioTemporalBlock3D(
                channels=channels,
                num_heads=num_heads,
                spatial_downsample=spatial_downsample,
            )
            for _ in range(num_layers)
        ])

        if aggregation == "learned_query":
            self.temporal_query = nn.Parameter(torch.randn(1, 1, channels))
            self.query_attn = nn.MultiheadAttention(
                channels,
                num_heads,
                batch_first=True,
            )
            self.query_norm = nn.LayerNorm(channels)

    def _pad_to_patch_size(self, x):
        B, C, D, H, W = x.shape
        pd, ph, pw = self.patch_size

        pad_d = (pd - D % pd) % pd
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        return x, D, H, W

    def _decode_and_crop(self, x, D, H, W):
        x = self.patch_decode(x)
        return x[..., :D, :H, :W]

    def forward(self, z_bound_list, mask=None):
        z_patches = []
        for z in z_bound_list:
            z, D_orig, H_orig, W_orig = self._pad_to_patch_size(z)
            z = self.patch_embed(z)
            z_patches.append(z)

        Z = torch.stack(z_patches, dim=1)

        for block in self.blocks:
            Z = block(Z, mask=mask)

        Z_patch_agg = Z
        B, N, C, Dp, Hp, Wp = Z.shape

        if self.aggregation == "mean_pool":
            if mask is not None:
                m = mask[:, :, None, None, None, None].float()
                h_patch = (Z * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                h_patch = Z.mean(1)

        elif self.aggregation == "learned_query":
            Z_spatial = rearrange(Z, "b n c dp hp wp -> (b dp hp wp) n c")

            query = self.temporal_query.expand(Z_spatial.shape[0], -1, -1)

            key_pad = None
            if mask is not None:
                S = Dp * Hp * Wp
                key_pad = ~mask
                key_pad = key_pad.repeat_interleave(S, dim=0)

            pooled, _ = self.query_attn(
                query,
                Z_spatial,
                Z_spatial,
                key_padding_mask=key_pad,
            )

            pooled = self.query_norm(pooled.squeeze(1))
            h_patch = rearrange(
                pooled,
                "(b dp hp wp) c -> b c dp hp wp",
                b=B,
                dp=Dp,
                hp=Hp,
                wp=Wp,
            )

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        Z_flat = rearrange(Z_patch_agg, "b n c dp hp wp -> (b n) c dp hp wp")
        Z_flat = self._decode_and_crop(Z_flat, D_orig, H_orig, W_orig)
        Z_agg = rearrange(Z_flat, "(b n) c d h w -> b n c d h w", b=B, n=N)

        h_temporal = self._decode_and_crop(h_patch, D_orig, H_orig, W_orig)

        return Z_agg, h_temporal


TEMPORAL_AGGREGATOR_VARIANTS = ("temporal_spatial", "patch_temporal_spatial")


def build_temporal_aggregator(ta_cfg: dict, channels: int) -> nn.Module:
    variant = ta_cfg.get("variant", "temporal_spatial")
    common_kwargs = dict(
        channels=channels,
        num_layers=ta_cfg["num_layers"],
        num_heads=ta_cfg["num_heads"],
        spatial_downsample=ta_cfg["spatial_downsample"],
        aggregation=ta_cfg["aggregation"],
    )
    if variant == "temporal_spatial":
        return TemporalSpatialAggregator3D(**common_kwargs)
    elif variant == "patch_temporal_spatial":
        return PatchTemporalSpatialAggregator3D(
            patch_size=tuple(ta_cfg.get("patch_size", (4, 16, 16))),
            **common_kwargs,
        )
    else:
        raise ValueError(
            f"Unknown temporal_aggregator.variant={variant!r}, expected one of "
            f"{TEMPORAL_AGGREGATOR_VARIANTS}"
        )
