"""GaussianDiffusion wraps operators for denoising diffusion models, including
the diffusion and denoising processes, as well as the loss evaluation."""

import torch

from ldm.ops.diffusion import GaussianDiffusion


def _i(tensor, t, x):
    """Index tensor using t and format the output according to x."""
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    return tensor[t.to(tensor.device)].view(shape).float().to(x.device)


class ContextGaussianDiffusion(GaussianDiffusion):

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

        if hasattr(self, 'progress'):
            self.progress(1 - float(t[0]) / 1000)

        classifier = getattr(self, 'classifier', 0)

        step_to_launch_face_guidence = model.share_cache.get(
            'step_to_launch_face_guidence', 600)
        if (t > step_to_launch_face_guidence).any():
            classifier = 0
            model.share_cache['similarity'] = model.share_cache.get(
                'lamda_feat_before_ref_guidence', 0.85)
        else:
            model.share_cache['similarity'] = model.share_cache.get(
                'ori_similarity', 0.85)
        # print('similarity', model.share_cache['similarity'])
        # print('classifier', classifier)
        if classifier > 0:

            # classifier-free guidance (arXiv:2207.12598)
            # model_kwargs[0]: conditional kwargs
            # model_kwargs[1]: non-conditional kwargs
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
            #  model.share_cache["do_mimic"] = True

            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
            cat_xt = xt.repeat(3, 1, 1, 1)
            conditional_embed = model_kwargs[0]['context']
            non_conditional_embed = model_kwargs[1]['context']
            text_embed = torch.cat(
                [conditional_embed, conditional_embed, non_conditional_embed],
                dim=0)
            # sim 1, 0, 0

            raw_out = model(cat_xt, t=t, context=text_embed)

            y_out_with_ref, y_out, u_out = torch.split(raw_out,
                                                       raw_out.size(0) // 3,
                                                       dim=0)

            out = u_out + guide_scale * (y_out - u_out) + classifier * (
                y_out_with_ref - y_out)

            if guide_rescale is not None:
                assert guide_rescale >= 0 and guide_rescale <= 1
                ratio = (
                    y_out.flatten(1).std(dim=1) /
                    (out.flatten(1).std(dim=1) + 1e-12)).view((-1, ) + (1, ) *
                                                              (y_out.ndim - 1))
                out *= guide_rescale * ratio + (1 - guide_rescale) * 1.0
        else:

            # classifier-free guidance (arXiv:2207.12598)
            # model_kwargs[0]: conditional kwargs
            # model_kwargs[1]: non-conditional kwargs
            # make it batch inference for both conditional and non-conditional
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
            cat_xt = xt.repeat(2, 1, 1, 1)
            conditional_embed = model_kwargs[0]['context']
            non_conditional_embed = model_kwargs[1]['context']
            text_embed = torch.cat([conditional_embed, non_conditional_embed],
                                   dim=0)

            #  model.share_cache["do_mimic"] = True
            y_out = model(cat_xt, t=t, context=text_embed)
            y_out_with_text, y_out_no_text = torch.split(y_out,
                                                         y_out.size(0) // 2,
                                                         dim=0)
            y_out = y_out_with_text

            out = y_out_no_text + guide_scale * (y_out_with_text -
                                                 y_out_no_text)
            # out = base_out  + guide_scale * (y_out - base_out)
            # out =   base_out + guide_rescale * (y_out - base_out)
            # rescale the output according to arXiv:2305.08891
            if guide_rescale is not None:
                assert guide_rescale >= 0 and guide_rescale <= 1
                ratio = (
                    y_out.flatten(1).std(dim=1) /
                    (out.flatten(1).std(dim=1) + 1e-12)).view((-1, ) + (1, ) *
                                                              (y_out.ndim - 1))
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

        # recompute eps using the restricted x0
        eps = (xt - alphas * x0) / sigmas

        # compute mu (mean of posterior distribution) using the restricted x0
        mu = coef1 * x0 + coef2 * xt
        return mu, var, log_var, x0, eps
