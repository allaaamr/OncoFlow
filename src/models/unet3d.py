import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _group_norm_groups(channels: int, max_groups: int = 32) -> int:
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class OptionalGenomicConditioner(nn.Module):

    def __init__(self, global_cond_dim=512, genomic_dim=13, geno_token_dim=512):
        super().__init__()

        self.geno_proj = nn.Linear(1, geno_token_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=global_cond_dim,
            num_heads=8,
            batch_first=True,
        )

        self.gate = nn.Sequential(
            nn.Linear(global_cond_dim + geno_token_dim, global_cond_dim),
            nn.Sigmoid(),
        )

    def forward(self, base_cond, genomics=None, geno_mask=None):
        if genomics is None:
            return base_cond

        B, G = genomics.shape

        q = base_cond.unsqueeze(1)
        geno_tokens = genomics.unsqueeze(-1)
        geno_tokens = self.geno_proj(geno_tokens)

        attn_out, _ = self.cross_attn(
            query=q,
            key=geno_tokens,
            value=geno_tokens,
        )

        attn_out = attn_out.squeeze(1)

        gate = self.gate(
            torch.cat([base_cond, attn_out], dim=-1)
        )

        updated = base_cond + gate * attn_out

        if geno_mask is not None:
            geno_mask = geno_mask.float().view(B, 1)
            updated = geno_mask * updated + (1.0 - geno_mask) * base_cond

        return updated


class MetadataTokenBuilder(nn.Module):

    def __init__(
        self,
        treatment_vocab_size=4,
        treatment_embed_dim=64,
        genomic_dim=13,
        token_dim=512,
        condition_on_treatment=True,
    ):
        super().__init__()

        self.condition_on_treatment = condition_on_treatment
        self.treatment_embed_dim = treatment_embed_dim
        self.treat_emb = nn.Embedding(treatment_vocab_size, treatment_embed_dim)

        self.treat_proj = nn.Sequential(
            nn.Linear(treatment_embed_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )

        self.genomic_proj = nn.Sequential(
            nn.Linear(genomic_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )

        self.interaction_proj = nn.Sequential(
            nn.Linear(treatment_embed_dim + genomic_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, target_treatment=None, genomics=None, geno_mask=None):
        if genomics is None:
            return None

        B = genomics.shape[0]

        if self.condition_on_treatment:
            treat_raw = self.treat_emb(target_treatment)
        else:
            treat_raw = genomics.new_zeros(B, self.treatment_embed_dim)
        treat_token = self.treat_proj(treat_raw)
        genomic_token = self.genomic_proj(genomics)

        interaction_input = torch.cat([treat_raw, genomics], dim=-1)
        interaction_token = self.interaction_proj(interaction_input)

        tokens = torch.stack(
            [treat_token, genomic_token, interaction_token],
            dim=1,
        )

        if geno_mask is not None:
            mask = geno_mask.float().view(B, 1, 1)
            tokens = tokens * mask

        return tokens


class GlobalConditionEncoder(nn.Module):

    def __init__(
        self,
        global_cond_dim=512,
        time_embed_dim=256,
        target_time_embed_dim=128,
        treatment_vocab_size=4,
        treatment_embed_dim=64,
        genomic_dim=13,
        condition_on_treatment=True,
    ):
        super().__init__()

        self.condition_on_treatment = condition_on_treatment
        self.treatment_embed_dim = treatment_embed_dim

        self.timestep_emb = SinusoidalTimestepEmbedding(time_embed_dim)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.target_time_emb = SinusoidalTimestepEmbedding(target_time_embed_dim)
        self.target_treat_emb = nn.Embedding(
            treatment_vocab_size,
            treatment_embed_dim,
        )

        base_total = (
            time_embed_dim
            + target_time_embed_dim
            + treatment_embed_dim
        )

        self.base_fuse = nn.Sequential(
            nn.Linear(base_total, global_cond_dim),
            nn.SiLU(),
            nn.Linear(global_cond_dim, global_cond_dim),
        )

        self.genomic_conditioner = OptionalGenomicConditioner(
            global_cond_dim=global_cond_dim,
            genomic_dim=genomic_dim,
            geno_token_dim=global_cond_dim,
        )

    def forward(
        self,
        t,
        target_day,
        target_treatment=None,
        genomics=None,
        geno_mask=None,
    ):
        time_part = self.timestep_mlp(self.timestep_emb(t))
        day_part = self.target_time_emb(target_day)
        if self.condition_on_treatment:
            treat_part = self.target_treat_emb(target_treatment)
        else:
            treat_part = time_part.new_zeros(time_part.shape[0], self.treatment_embed_dim)

        base_cond = self.base_fuse(
            torch.cat([time_part, day_part, treat_part], dim=-1)
        )

        global_cond = self.genomic_conditioner(
            base_cond,
            genomics=genomics,
            geno_mask=geno_mask,
        )

        return global_cond


class ConditionalResBlock3D(nn.Module):

    def __init__(self, in_ch, out_ch, global_cond_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_norm_groups(in_ch), in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_group_norm_groups(out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(global_cond_dim, 2 * out_ch))
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, global_cond):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.cond_proj(global_cond).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        h = self.dropout(self.act(h))
        h = self.conv2(h)
        return h + self.skip(x)


class CrossAttentionBlock3D(nn.Module):

    def __init__(self, channels, context_channels, num_heads=8):
        super().__init__()
        self.norm_q = nn.GroupNorm(_group_norm_groups(channels), channels)
        self.norm_kv = nn.LayerNorm(context_channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(context_channels, channels)
        self.to_v = nn.Linear(context_channels, channels)
        self.proj_out = nn.Linear(channels, channels)
        self.num_heads = num_heads

    def forward(self, x, context):
        B, C, d, h, w = x.shape
        residual = x
        x_tok = rearrange(self.norm_q(x), "b c d h w -> b (d h w) c")
        context = self.norm_kv(context)
        q = rearrange(self.to_q(x_tok), "b s (nh d) -> b nh s d", nh=self.num_heads)
        k = rearrange(self.to_k(context), "b t (nh d) -> b nh t d", nh=self.num_heads)
        v = rearrange(self.to_v(context), "b t (nh d) -> b nh t d", nh=self.num_heads)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = rearrange(attn, "b nh s d -> b s (nh d)")
        out = rearrange(self.proj_out(attn), "b (d h w) c -> b c d h w", d=d, h=h, w=w)
        return residual + out


class Downsample3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False))


class ConditionalUNet3D(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=2,
        num_heads=4,
        dropout=0.1,
        global_cond_dim=512,
        concat_temporal_summary=False,
        concat_last=True,
        cross_attn_temporal_sequence=True,
        temporal_context_channels=1,
        sn_context_dim=512,
        metadata_cross_attn=False,
        metadata_context_dim=512,
        attention_min_level=1,
        concat_channels=None,
    ):
        super().__init__()
        concat_channels = in_channels if concat_channels is None else concat_channels
        self.concat_temporal = concat_temporal_summary
        self.concat_last = concat_last
        self.cross_attn = cross_attn_temporal_sequence
        self.metadata_cross_attn = metadata_cross_attn
        self.metadata_context_dim = metadata_context_dim
        self.attention_min_level = attention_min_level
        self.num_downsamplers = len(channel_multipliers) - 1

        actual_in = in_channels + (concat_channels if concat_temporal_summary or concat_last else 0)
        self.input_conv = nn.Conv3d(actual_in, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.down_meta_attns = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        ch = base_channels
        channels_list = [ch]
        attn_context_channels = temporal_context_channels

        for level, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            block_list = nn.ModuleList()
            attn_list = nn.ModuleList()
            meta_attn_list = nn.ModuleList()
            level_has_attn = cross_attn_temporal_sequence and level >= attention_min_level
            for _ in range(num_res_blocks):
                block_list.append(ConditionalResBlock3D(ch, out_ch, global_cond_dim, dropout))
                ch = out_ch
                channels_list.append(ch)
                attn_list.append(CrossAttentionBlock3D(ch, attn_context_channels, num_heads) if level_has_attn else None)
                if metadata_cross_attn and level_has_attn:
                    meta_attn_list.append(CrossAttentionBlock3D(ch, metadata_context_dim, num_heads))
                else:
                    meta_attn_list.append(None)
            self.down_blocks.append(block_list)
            self.down_attns.append(attn_list)
            self.down_meta_attns.append(meta_attn_list)
            if level < len(channel_multipliers) - 1:
                self.downsamplers.append(Downsample3D(ch))
                channels_list.append(ch)
            else:
                self.downsamplers.append(None)

        self.mid_block1 = ConditionalResBlock3D(ch, ch, global_cond_dim, dropout)
        self.mid_attn = CrossAttentionBlock3D(ch, attn_context_channels, num_heads) if cross_attn_temporal_sequence else None
        self.mid_meta_attn = CrossAttentionBlock3D(ch, metadata_context_dim, num_heads) if metadata_cross_attn else None
        self.mid_block2 = ConditionalResBlock3D(ch, ch, global_cond_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.up_meta_attns = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            block_list = nn.ModuleList()
            attn_list = nn.ModuleList()
            meta_attn_list = nn.ModuleList()
            level_has_attn = cross_attn_temporal_sequence and level >= attention_min_level
            for _ in range(num_res_blocks + 1):
                skip_ch = channels_list.pop()
                block_list.append(ConditionalResBlock3D(ch + skip_ch, out_ch, global_cond_dim, dropout))
                ch = out_ch
                attn_list.append(CrossAttentionBlock3D(ch, attn_context_channels, num_heads) if level_has_attn else None)
                if metadata_cross_attn and level_has_attn:
                    meta_attn_list.append(CrossAttentionBlock3D(ch, metadata_context_dim, num_heads))
                else:
                    meta_attn_list.append(None)
            self.up_blocks.append(block_list)
            self.up_attns.append(attn_list)
            self.up_meta_attns.append(meta_attn_list)
            self.upsamplers.append(Upsample3D(ch) if level > 0 else None)

        self.out_norm = nn.GroupNorm(_group_norm_groups(ch), ch)
        self.out_conv = nn.Conv3d(ch, out_channels, 3, padding=1)

    def _prepare_context(self, Z_agg, target_dhw):
        if Z_agg is None:
            raise ValueError(
                "Cross-attention is enabled (attention_min_level reached) but Z_agg is None."
            )
        B, N, C, d, h, w = Z_agg.shape
        td, th, tw = target_dhw
        if (d, h, w) != (td, th, tw):
            Z_flat = rearrange(Z_agg, "b n c d h w -> (b n) c d h w")
            Z_flat = F.interpolate(Z_flat, size=(td, th, tw), mode="trilinear", align_corners=False)
            Z_agg = rearrange(Z_flat, "(b n) c d h w -> b n c d h w", b=B, n=N)
        return rearrange(Z_agg, "b n c d h w -> b (n d h w) c")

    @staticmethod
    def _pad_to_divisible(x: torch.Tensor, multiple: int):
        D, H, W = x.shape[-3:]
        pd = (-D) % multiple
        ph = (-H) % multiple
        pw = (-W) % multiple
        if pd or ph or pw:
            x = F.pad(x, (0, pw, 0, ph, 0, pd))
        return x, (D, H, W)

    @staticmethod
    def _match_spatial(x: torch.Tensor, dhw):
        if tuple(x.shape[-3:]) != tuple(dhw):
            x = F.interpolate(x, size=tuple(dhw), mode="trilinear", align_corners=False)
        return x

    @staticmethod
    def _crop_to(x: torch.Tensor, dhw):
        D, H, W = dhw
        return x[..., :D, :H, :W]

    def forward(
        self,
        z_t,
        global_cond,
        h_temporal=None,
        Z_agg=None,
        Sn=None,
        metadata_tokens=None,
        input_mask=None,
    ):
        in_shape = z_t.shape
        multiple = 2 ** self.num_downsamplers

        z_t, orig_dhw = self._pad_to_divisible(z_t, multiple)
        if h_temporal is not None:
            h_temporal = self._match_spatial(h_temporal, orig_dhw)
            h_temporal, _ = self._pad_to_divisible(h_temporal, multiple)

        if self.concat_temporal and h_temporal is not None:
            x = torch.cat([z_t, h_temporal], dim=1)
        elif self.concat_last and Z_agg is not None:
            if input_mask is not None:
                last_idx = (input_mask.sum(dim=1).long() - 1).clamp(min=0)
                batch_idx = torch.arange(Z_agg.shape[0], device=Z_agg.device)
                z_last_agg = Z_agg[batch_idx, last_idx]
            else:
                z_last_agg = Z_agg[:, -1]
            z_last_agg = self._match_spatial(z_last_agg, orig_dhw)
            z_last_agg, _ = self._pad_to_divisible(z_last_agg, multiple)
            x = torch.cat([z_t, z_last_agg], dim=1)
        else:
            x = z_t
        x = self.input_conv(x)

        skips = [x]
        for blocks, attns, meta_attns, down in zip(
            self.down_blocks,
            self.down_attns,
            self.down_meta_attns,
            self.downsamplers,
        ):
            for block, attn, meta_attn in zip(blocks, attns, meta_attns):
                x = block(x, global_cond)
                if attn is not None:
                    context = self._prepare_context(Z_agg, x.shape[2:])
                    x = attn(x, context)
                if meta_attn is not None and metadata_tokens is not None:
                    x = meta_attn(x, metadata_tokens)
                skips.append(x)
            if down is not None:
                x = down(x)
                skips.append(x)

        x = self.mid_block1(x, global_cond)
        if self.mid_attn is not None:
            context = self._prepare_context(Z_agg, x.shape[2:])
            x = self.mid_attn(x, context)
        if self.mid_meta_attn is not None and metadata_tokens is not None:
            x = self.mid_meta_attn(x, metadata_tokens)
        x = self.mid_block2(x, global_cond)

        for blocks, attns, meta_attns, up in zip(
            self.up_blocks,
            self.up_attns,
            self.up_meta_attns,
            self.upsamplers,
        ):
            for block, attn, meta_attn in zip(blocks, attns, meta_attns):
                skip = skips.pop()
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
                x = block(torch.cat([x, skip], dim=1), global_cond)
                if attn is not None:
                    context = self._prepare_context(Z_agg, x.shape[2:])
                    x = attn(x, context)
                if meta_attn is not None and metadata_tokens is not None:
                    x = meta_attn(x, metadata_tokens)
            if up is not None:
                x = up(x)

        out = self.out_conv(F.silu(self.out_norm(x)))
        out = self._crop_to(out, orig_dhw)

        assert out.shape == in_shape, (
            f"[ConditionalUNet3D] output shape {tuple(out.shape)} != input z_t shape {tuple(in_shape)}"
        )
        return out
