import torch
import torch.nn as nn
import torch.nn.functional as F


FLOW_PATHS = ("standard", "region_aware_source")
FLOW_LOSSES = ("standard", "tumor_weighted", "normalized_tumor_weighted")
BLEND_MODES = ("linear", "vp")


def blend_source(
    preservation_map: torch.Tensor,
    z_prev: torch.Tensor,
    noise: torch.Tensor,
    blend: str = "linear",
) -> torch.Tensor:
    P = preservation_map
    if blend == "linear":
        return P * z_prev + (1.0 - P) * noise
    elif blend == "vp":
        noise_coef = (1.0 - P.pow(2)).clamp(min=0.0).sqrt()
        return P * z_prev + noise_coef * noise
    else:
        raise ValueError(f"Unknown region_aware_source.blend={blend!r}, expected one of {BLEND_MODES}")


def region_aware_source_path(
    z_future: torch.Tensor,
    z_prev: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    preservation_map: torch.Tensor,
    baseline_mask: torch.Tensor = None,
    return_diagnostics: bool = False,
    blend: str = "linear",
):
    t_ = ConditionalFlowMatching._ext(t, z_future.shape)

    P = preservation_map
    z0_ra = blend_source(P, z_prev, noise, blend=blend)
    z_t = (1 - t_) * z0_ra + t_ * z_future
    velocity = z_future - z0_ra

    if not return_diagnostics:
        return z_t, z0_ra, velocity

    with torch.no_grad():
        diag = {
            "P_mean": P.mean().item(),
            "P_min": P.min().item(),
            "P_max": P.max().item(),
        }
        if baseline_mask is not None:
            inside = baseline_mask > 0.5
            peritumoral = (~inside) & (P < 0.95)
            outside = P >= 0.95
            has_inside = bool(inside.any())
            has_peri = bool(peritumoral.any())
            has_outside = bool(outside.any())
            diag.update({
                "P_inside_tumor_mean": P[inside].mean().item() if has_inside else float("nan"),
                "P_peritumoral_mean": P[peritumoral].mean().item() if has_peri else float("nan"),
                "P_outside_tumor_mean": P[outside].mean().item() if has_outside else float("nan"),
            })
            inside_v = inside.expand_as(velocity)
            outside_v = (~inside).expand_as(velocity)
            has_inside_v = bool(inside_v.any())
            has_outside_v = bool(outside_v.any())
            diag.update({
                "velocity_abs_inside_tumor_mean": velocity.abs()[inside_v].mean().item() if has_inside_v else float("nan"),
                "velocity_abs_outside_tumor_mean": velocity.abs()[outside_v].mean().item() if has_outside_v else float("nan"),
            })
    return z_t, z0_ra, velocity, diag


class ConditionalFlowMatching(nn.Module):
    def __init__(self, sigma_min: float = 0.0, time_scale: float = 999.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.time_scale = time_scale

    def sample_t(self, batch_size: int, device) -> torch.Tensor:
        return torch.rand(batch_size, device=device)

    def sample_path(self, x1: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x1)
        t_ = self._ext(t, x1.shape)
        if self.sigma_min > 0:
            x_t = (1 - (1 - self.sigma_min) * t_) * noise + t_ * x1
        else:
            x_t = (1 - t_) * noise + t_ * x1
        velocity = x1 - noise
        return x_t, noise, velocity

    def construct_flow_path(
        self,
        z_future: torch.Tensor,
        t: torch.Tensor,
        path_type: str,
        noise: torch.Tensor = None,
        z_prev: torch.Tensor = None,
        preservation_map: torch.Tensor = None,
        baseline_mask: torch.Tensor = None,
        return_diagnostics: bool = False,
        blend: str = "linear",
    ):
        if noise is None:
            noise = torch.randn_like(z_future)

        if path_type == "standard":
            z_t, source, velocity = self.sample_path(z_future, t, noise=noise)
            diag = {}
        elif path_type == "region_aware_source":
            if z_prev is None:
                raise ValueError(
                    "flow_path='region_aware_source' requires z_prev (the previous/"
                    "baseline-visit latent) — got None. This path needs the previous "
                    "MRI's latent to construct the source state z_0^RA."
                )
            if preservation_map is None:
                raise ValueError(
                    "flow_path='region_aware_source' requires preservation_map (see "
                    "src/utils/preservation_map.py::build_preservation_map) — got None. "
                    "This path needs the baseline tumor mask to construct P(r)."
                )
            result = region_aware_source_path(
                z_future, z_prev, noise, t, preservation_map,
                baseline_mask=baseline_mask, return_diagnostics=return_diagnostics, blend=blend,
            )
            if return_diagnostics:
                z_t, source, velocity, diag = result
            else:
                z_t, source, velocity = result
                diag = {}
        else:
            raise ValueError(f"Unknown flow_path={path_type!r}, expected one of {FLOW_PATHS}")

        if return_diagnostics:
            return z_t, source, velocity, diag
        return z_t, source, velocity

    def compute_loss(self, pred_v: torch.Tensor, target_v: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_v, target_v)

    def embed_time(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.time_scale

    @torch.no_grad()
    def sample(self, velocity_fn, shape, num_steps: int = 50, solver: str = "euler",
               device: str = "cuda", init_state: torch.Tensor = None, **kw) -> torch.Tensor:
        if solver not in ("euler", "heun"):
            raise ValueError(f"Unknown flow-matching solver: {solver!r} (expected 'euler' or 'heun')")

        x = init_state if init_state is not None else torch.randn(shape, device=device)
        ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_cur = ts[i].expand(shape[0])
            v1 = velocity_fn(x, t_cur, **kw)
            if solver == "euler":
                x = x + v1 * dt
            else:
                t_next = ts[i + 1].expand(shape[0])
                x_euler = x + v1 * dt
                v2 = velocity_fn(x_euler, t_next, **kw)
                x = x + 0.5 * (v1 + v2) * dt

        return x

    @staticmethod
    def _ext(t: torch.Tensor, shape) -> torch.Tensor:
        out = t
        while out.ndim < len(shape):
            out = out.unsqueeze(-1)
        return out.expand(shape)
