"""GaussianDiffusion wraps denoising, diffusion, sampling, and loss computation
operators for diffusion models. We consider a variance preserving (VP) process
where the diffusion process is:

q(x_t | x_0) = N(x_t | alpha_t x_0, sigma_t^2 I),

where alpha_t^2 = 1 - sigma_t^2.
"""
import random

import torch

from .schedules import karras_schedule, ve_to_vp, vp_to_ve
from .solvers import sample_ddim

__all__ = ['GaussianDiffusion']


def _i(tensor, t, x):
    """Index tensor using t and format the output according to x."""
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    return tensor[t.to(tensor.device)].view(shape).float().to(x.device)


class GaussianDiffusion(object):

    def __init__(self, sigmas, prediction_type='eps'):
        assert prediction_type in {'x0', 'eps', 'v'}
        self.sigmas = sigmas  # noise coefficients
        self.alphas = torch.sqrt(1 - sigmas**2)  # signal coefficients
        self.num_timesteps = len(sigmas)
        self.prediction_type = prediction_type

    def diffuse(self, x0, t, noise=None):
        """Add Gaussian noise to signal x0 according to:

        q(x_t | x_0) = N(x_t | alpha_t x_0, sigma_t^2 I).
        """
        noise = torch.randn_like(x0) if noise is None else noise
        xt = _i(self.alphas, t, x0) * x0 + _i(self.sigmas, t, x0) * noise
        return xt

    def denoise(self,
                xt,
                t,
                s,
                model,
                model_kwargs={},
                guide_scale=None,
                guide_rescale=None,
                clamp=None,
                percentile=None):
        """Apply one step of denoising from the posterior distribution q(x_s |
        x_t, x0).

        Since x0 is not available, estimate the denoising results using the
        learned distribution p(x_s | x_t, \hat{x}_0 == f(x_t)).
        """
        s = t - 1 if s is None else s

        # hyperparams
        sigmas = _i(self.sigmas, t, xt)
        alphas = _i(self.alphas, t, xt)
        alphas_s = _i(self.alphas, s.clamp(0), xt)
        alphas_s[s < 0] = 1.
        sigmas_s = torch.sqrt(1 - alphas_s**2)

        # precompute variables
        betas = 1 - (alphas / alphas_s)**2
        coef1 = betas * alphas_s / sigmas**2
        coef2 = (alphas * sigmas_s**2) / (alphas_s * sigmas**2)
        var = betas * (sigmas_s / sigmas)**2
        log_var = torch.log(var).clamp_(-20, 20)

        # prediction
        if guide_scale is None:
            assert isinstance(model_kwargs, dict)
            out = model(xt, t=t, **model_kwargs)
        else:
            # classifier-free guidance (arXiv:2207.12598)
            # model_kwargs[0]: conditional kwargs
            # model_kwargs[1]: non-conditional kwargs
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
            y_out = model(xt, t=t, **model_kwargs[0])
            if guide_scale == 1.:
                out = y_out
            else:
                u_out = model(xt, t=t, **model_kwargs[1])
                out = u_out + guide_scale * (y_out - u_out)

                # rescale the output according to arXiv:2305.08891
                if guide_rescale is not None:
                    assert guide_rescale >= 0 and guide_rescale <= 1
                    ratio = (y_out.flatten(1).std(dim=1) /
                             (out.flatten(1).std(dim=1) +
                              1e-12)).view((-1, ) + (1, ) * (y_out.ndim - 1))
                    out *= guide_rescale * ratio + (1 - guide_rescale) * 1.0

        # compute x0
        if self.prediction_type == 'x0':
            x0 = out
        elif self.prediction_type == 'eps':
            x0 = (xt - sigmas * out) / alphas
        elif self.prediction_type == 'v':
            x0 = alphas * xt - sigmas * out
        else:
            raise NotImplementedError(
                f'prediction_type {self.prediction_type} not implemented')

        # restrict the range of x0
        if percentile is not None:
            # NOTE: percentile should only be used when data is within range [-1, 1]
            assert percentile > 0 and percentile <= 1
            s = torch.quantile(x0.flatten(1).abs(), percentile, dim=1)
            s = s.clamp_(1.0).view((-1, ) + (1, ) * (xt.ndim - 1))
            x0 = torch.min(s, torch.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)

        eps = (xt - alphas * x0) / sigmas

        mu = coef1 * x0 + coef2 * xt
        return mu, var, log_var, x0, eps

    @torch.no_grad()
    def sample(self,
               noise,
               model,
               model_kwargs={},
               guide_scale=None,
               guide_rescale=None,
               clamp=None,
               percentile=None,
               solver='ddim',
               steps=20,
               t_max=None,
               t_min=None,
               discretization=None,
               discard_penultimate_step=None,
               return_intermediate=None,
               show_progress=False,
               seed=-1,
               **kwargs):
        # sanity check
        assert isinstance(steps, (int, torch.LongTensor))
        assert t_max is None or (t_max > 0 and t_max <= self.num_timesteps - 1)
        assert t_min is None or (t_min >= 0 and t_min < self.num_timesteps - 1)
        assert discretization in (None, 'leading', 'linspace', 'trailing')
        assert discard_penultimate_step in (None, True, False)
        assert return_intermediate in (None, 'x0', 'xt')
        assert solver == 'ddim'
        # function of diffusion solver
        solver_fn = {'ddim': sample_ddim}[solver]

        # options
        schedule = 'karras' if 'karras' in solver else None
        discretization = discretization or 'linspace'
        seed = seed if seed >= 0 else random.randint(0, 2**31)
        if isinstance(steps, torch.LongTensor):
            discard_penultimate_step = False
        if discard_penultimate_step is None:
            discard_penultimate_step = True if solver in (
                'dpm2', 'dpm2_ancestral', 'dpmpp_2m_sde', 'dpm2_karras',
                'dpm2_ancestral_karras', 'dpmpp_2m_sde_karras') else False

        # function for denoising xt to get x0
        intermediates = []

        def model_fn(xt, sigma):
            # denoising
            t = self._sigma_to_t(sigma).expand(len(xt)).round().long()
            x0 = self.denoise(xt, t, None, model, model_kwargs, guide_scale,
                              guide_rescale, clamp, percentile)[-2]

            # collect intermediate outputs
            if return_intermediate == 'xt':
                intermediates.append(xt)
            elif return_intermediate == 'x0':
                intermediates.append(x0)
            return x0

        # get timesteps
        if isinstance(steps, int):
            steps += 1 if discard_penultimate_step else 0
            t_max = self.num_timesteps - 1 if t_max is None else t_max
            t_min = 0 if t_min is None else t_min

            # discretize timesteps
            if discretization == 'leading':
                steps = torch.arange(t_min, t_max + 1,
                                     (t_max - t_min + 1) / steps).flip(0)
            elif discretization == 'linspace':
                steps = torch.linspace(t_max, t_min, steps)
            elif discretization == 'trailing':
                steps = torch.arange(t_max, t_min - 1,
                                     -((t_max - t_min + 1) / steps))
            else:
                raise NotImplementedError(
                    f'{discretization} discretization not implemented')
            steps = steps.clamp_(t_min, t_max)
        steps = torch.as_tensor(steps,
                                dtype=torch.float32,
                                device=noise.device)

        # get sigmas
        sigmas = self._t_to_sigma(steps)
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        if schedule == 'karras':
            if sigmas[0] == 1.:
                sigmas = karras_schedule(
                    n=len(steps) - 1,
                    sigma_min=sigmas[sigmas > 0].min().item(),
                    sigma_max=sigmas[sigmas < 1.].max().item(),
                    rho=7.).to(sigmas)
                sigmas = torch.cat(
                    [sigmas.new_ones([1]), sigmas,
                     sigmas.new_zeros([1])])
            else:
                sigmas = karras_schedule(
                    n=len(steps),
                    sigma_min=sigmas[sigmas > 0].min().item(),
                    sigma_max=sigmas.max().item(),
                    rho=7.).to(sigmas)
                sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        if discard_penultimate_step:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

        # sampling
        x0 = solver_fn(noise,
                       model_fn,
                       sigmas,
                       show_progress=show_progress,
                       **kwargs)
        return (x0, intermediates) if return_intermediate is not None else x0

    def _t_to_sigma(self, t):
        """Convert time steps (float) to sigmas by interpolating in the log-VE-
        sigma space."""
        t = t.float()
        i, j, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigmas = vp_to_ve(self.sigmas).log().to(t)
        log_sigma = (1 - w) * log_sigmas[i] + w * log_sigmas[j]
        log_sigma[torch.isnan(log_sigma)
                  | torch.isinf(log_sigma)] = float('inf')
        return ve_to_vp(log_sigma.exp())

    def _sigma_to_t(self, sigma):
        """Convert sigma to time step (float) by interpolating in the log-VE-
        sigma space."""
        if sigma == 1.:
            t = torch.full_like(sigma, self.num_timesteps - 1)
        else:
            log_sigmas = vp_to_ve(self.sigmas).log().to(sigma)
            log_sigma = vp_to_ve(sigma).log()

            # interpolation
            i = torch.where((log_sigma - log_sigmas).ge(0))[0][-1].clamp(
                max=self.num_timesteps - 2).unsqueeze(0)
            j = i + 1
            w = (log_sigmas[i] - log_sigma) / (log_sigmas[i] - log_sigmas[j])
            w = w.clamp(0, 1)
            t = (1 - w) * i + w * j
        if t.ndim == 0:
            t = t.unsqueeze(0)
        return t
