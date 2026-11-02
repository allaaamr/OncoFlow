import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = ab / ab[0]
    betas = 1 - ab[1:] / ab[:-1]
    return betas.clamp(0.0001, 0.999)


def linear_beta_schedule(T, start=1e-4, end=0.02):
    return torch.linspace(start, end, T)


class GaussianDiffusion(nn.Module):
    def __init__(self, num_timesteps=1000, schedule="cosine", prediction_type="epsilon",
                 beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.T = num_timesteps
        self.prediction_type = prediction_type

        betas = cosine_beta_schedule(num_timesteps) if schedule == "cosine" else linear_beta_schedule(num_timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        ab = torch.cumprod(alphas, 0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", ab)
        self.register_buffer("alpha_bar_prev", torch.cat([torch.ones(1), ab[:-1]]))
        self.register_buffer("sqrt_ab", ab.sqrt())
        self.register_buffer("sqrt_1m_ab", (1 - ab).sqrt())
        self.register_buffer("sqrt_recip_ab", (1.0 / ab).sqrt())
        self.register_buffer("sqrt_recip_ab_m1", (1.0 / ab - 1).sqrt())
        post_var = betas * (1 - self.alpha_bar_prev) / (1 - ab)
        self.register_buffer("post_var", post_var)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self._ext(self.sqrt_ab, t, x0.shape)
        sm = self._ext(self.sqrt_1m_ab, t, x0.shape)
        return sa * x0 + sm * noise, noise

    def compute_loss(self, pred, x0, noise, t):
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "x0":
            target = x0
        elif self.prediction_type == "v":
            sa = self._ext(self.sqrt_ab, t, x0.shape)
            sm = self._ext(self.sqrt_1m_ab, t, x0.shape)
            target = sa * noise - sm * x0
        else:
            raise ValueError(self.prediction_type)
        return F.mse_loss(pred, target)

    def predict_x0(self, pred, x_t, t):
        if self.prediction_type == "epsilon":
            r = self._ext(self.sqrt_recip_ab, t, x_t.shape)
            rm = self._ext(self.sqrt_recip_ab_m1, t, x_t.shape)
            return r * x_t - rm * pred
        elif self.prediction_type == "x0":
            return pred
        elif self.prediction_type == "v":
            sa = self._ext(self.sqrt_ab, t, x_t.shape)
            sm = self._ext(self.sqrt_1m_ab, t, x_t.shape)
            return sa * x_t - sm * pred

    @torch.no_grad()
    def ddim_sample(self, denoise_fn, shape, num_steps=50, eta=0.0, device="cuda", **kw):
        step_size = max(1, self.T // num_steps)
        timesteps = list(range(0, self.T, step_size))[::-1]
        x = torch.randn(shape, device=device)
        for i, t in enumerate(timesteps):
            tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
            out = denoise_fn(x, tt, **kw)
            x0 = self.predict_x0(out, x, tt).clamp(-3, 3)
            t_next = timesteps[i + 1] if i < len(timesteps) - 1 else 0
            ab_t = self.alpha_bar[t]
            ab_n = self.alpha_bar[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
            sigma = eta * ((1 - ab_n) / (1 - ab_t) * (1 - ab_t / ab_n)).sqrt()
            pred_dir = (1 - ab_n - sigma**2).sqrt() * ((x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt())
            x = ab_n.sqrt() * x0 + pred_dir
            if sigma > 0 and t_next > 0:
                x = x + sigma * torch.randn_like(x)
        return x

    @staticmethod
    def _ext(sched, t, shape):
        out = sched.gather(0, t)
        while out.ndim < len(shape):
            out = out.unsqueeze(-1)
        return out.expand(shape)
