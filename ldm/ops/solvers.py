import torch
from tqdm.auto import trange

__all__ = ['sample_ddim', 'sample_euler']

from comfy.k_diffusion.sampling import to_d


@torch.no_grad()
def sample_ddim(noise, model, sigmas, eta=0., show_progress=True):
    """DDIM solver steps."""
    x = noise
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        denoised = model(x, sigmas[i])
        noise_factor = eta * (sigmas[i + 1]**2 / sigmas[i]**2 *
                              (1 - (1 - sigmas[i]**2) /
                               (1 - sigmas[i + 1]**2)))
        d = (x - (1 - sigmas[i]**2)**0.5 * denoised) / sigmas[i]
        x = (1 - sigmas[i + 1] ** 2) ** 0.5 * denoised + \
            (sigmas[i + 1] ** 2 - noise_factor ** 2) ** 0.5 * d
        if sigmas[i + 1] > 0:
            x += noise_factor * torch.randn_like(x)
    return x

@torch.no_grad()
def sample_euler(noise, model, sigmas, show_progress=True):
    """Euler steps."""
    x = noise
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        denoised = model(x, sigmas[i])
        step_size = sigmas[i + 1] - sigmas[i]
        x = x + step_size * denoised
        if sigmas[i + 1] > 0:
            x += sigmas[i + 1] * torch.randn_like(x)
    return x

@torch.no_grad()
def sample_euler(model, x, sigmas, show_progress= True, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x