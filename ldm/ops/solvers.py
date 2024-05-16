import torch
from tqdm.auto import trange

__all__ = ['sample_ddim', 'sample_dpm2pp', 'sample_euler']


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
    """Euler solver steps."""
    x = noise
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        denoised = model(x, sigmas[i])
        sigma_up, sigma_down = sigmas[i], sigmas[i + 1]
        d = (x - denoised) / sigma_up
        dt = sigma_down - sigma_up
        x = x + d * dt
        if sigma_down > 0:
            x += torch.randn_like(x) * sigma_down
    return x


@torch.no_grad()
def sample_dpm2pp(noise, model, sigmas, show_progress=True):
    """DPM2++ solver steps."""
    x = noise
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        denoised = model(x, sigmas[i])
        sigma_up, sigma_down = sigmas[i], sigmas[i + 1]
        gamma = min(sigma_down / sigma_up, 0.999)  # Clip gamma to avoid numerical instability
        eps = torch.randn_like(x)
        sigma_mid = sigma_up * (gamma ** 2)
        dt_1 = (sigma_down ** 2 - sigma_mid ** 2) ** 0.5
        dt_2 = (sigma_mid ** 2 - sigma_down ** 2) ** 0.5
        x_1 = x + eps * dt_1
        denoised_2 = model(x_1, sigma_down)
        if sigma_down > 0:
            d_2 = (eps - (denoised_2 - denoised) / sigma_mid) / (1 - sigma_mid / sigma_up)
            x = x_1 + d_2 * dt_2
        else:
            x = denoised_2
    return x
