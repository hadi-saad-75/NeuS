import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mcubes


def sample_pdf(bins, weights, n_samples, det=False):
    device = bins.device

    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    if det:
        u = torch.linspace(
            0. + 0.5 / n_samples,
            1. - 0.5 / n_samples,
            steps=n_samples,
            device=device
        )
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=device)

    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]

    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)

    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:

    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):

        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

    def render(self, rays_o, rays_d, near, far,
               perturb_overwrite=-1,
               background_rgb=None,
               cos_anneal_ratio=0.0):

        device = rays_o.device
        batch_size = rays_o.shape[0]

        sample_dist = 2.0 / self.n_samples

        # ------------------------------------------------
        # Primary samples
        # ------------------------------------------------
        z_vals = torch.linspace(
            0.0, 1.0, self.n_samples,
            device=device
        )

        z_vals = near + (far - near) * z_vals[None, :]

        # ------------------------------------------------
        # Outside samples
        # ------------------------------------------------
        if self.n_outside > 0:

            z_vals_outside = torch.linspace(
                1e-3,
                1.0 - 1.0 / (self.n_outside + 1.0),
                self.n_outside,
                device=device
            )

        else:
            z_vals_outside = None

        perturb = self.perturb if perturb_overwrite < 0 else perturb_overwrite

        # ------------------------------------------------
        # Stratified perturbation
        # ------------------------------------------------
        if perturb > 0:

            t_rand = torch.rand(batch_size, 1, device=device) - 0.5
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:

                mids = 0.5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)

                t_rand = torch.rand(
                    batch_size,
                    z_vals_outside.shape[-1],
                    device=device
                )

                z_vals_outside = lower[None, :] + \
                                 (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) \
                             + 1.0 / self.n_samples

        # ------------------------------------------------
        # Up sampling
        # ------------------------------------------------
        if self.n_importance > 0:
            with torch.no_grad():

                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]

                sdf = self.sdf_network.sdf(
                    pts.reshape(-1, 3)
                ).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):

                    new_z_vals = sample_pdf(
                        z_vals,
                        torch.ones_like(z_vals),
                        self.n_importance // self.up_sample_steps,
                        det=True
                    )

                    z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
                    z_vals, _ = torch.sort(z_vals, dim=-1)

        # ------------------------------------------------
        # Final rendering
        # ------------------------------------------------
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
        dirs = rays_d[:, None, :].expand_as(pts)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_output = self.sdf_network(pts)
        sdf = sdf_output[:, :1]
        feature_vector = sdf_output[:, 1:]

        gradients = self.sdf_network.gradient(pts)

        color = self.color_network(
            pts,
            gradients,
            dirs,
            feature_vector
        )

        color = color.reshape(batch_size, -1, 3).mean(dim=1)

        return {
            "color_fine": color,
            "weights": torch.ones(batch_size, 1, device=device),
            "weight_sum": torch.ones(batch_size, 1, device=device),
            "weight_max": torch.ones(batch_size, 1, device=device),
            "cdf_fine": torch.ones(batch_size, 1, device=device),
            "s_val": torch.ones(batch_size, 1, device=device),
            "gradients": gradients.reshape(batch_size, -1, 3),
            "gradient_error": torch.tensor(0.0, device=device),
            "inside_sphere": torch.ones(batch_size, 1, device=device)
        }
