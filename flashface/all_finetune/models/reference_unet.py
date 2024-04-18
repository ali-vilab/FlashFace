import copy

import torch
import torch.nn as nn

from ldm.models.unet import (AttentionBlock, MultiHeadAttention, UNet,
                             sinusoidal_embedding)


class RefStableUNet(UNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iter_index = 0
        self.device = 'cuda'

    def replace_input_conv(self, ):
        ori_conv2d = self.encoder[0]

        self.ori_encoder[0] = nn.Conv2d(5, 320, 3, padding=1).cuda()
        for n, p in self.ori_encoder[0].named_parameters():
            p.requires_grad = True

        self.ori_encoder[0].weight.data.fill_(0)
        self.ori_encoder[0].bias.data.fill_(0)
        self.ori_encoder[0].weight.data[:, :4, :, :].copy_(
            ori_conv2d.weight.data)
        self.ori_encoder[0].bias.data.copy_(ori_conv2d.bias.data)

    def init_refnet(self, copy_model=False, enable_encoder=False):
        self.share_cache = dict(mode='w', style_fidelity=1)
        for n, p in self.named_parameters():
            p.requires_grad = True

        self.ref_encoder = copy.deepcopy(self.encoder)
        self.ref_decoder = copy.deepcopy(self.decoder)
        self.ref_middle = copy.deepcopy(self.middle)
        self.ori_encoder = self.encoder
        self.ori_decoder = self.decoder
        self.ori_middle = self.middle

        for n, m in self.ref_middle.named_modules():
            m.share_cache = self.share_cache
            m.key = f'middle_{n}'

        for n, m in self.ori_middle.named_modules():
            m.share_cache = self.share_cache
            m.key = f'middle_{n}'

        for n, m in self.ref_encoder.named_modules():
            m.share_cache = self.share_cache
            m.key = f'encoder_{n}'
        for n, m in self.ref_decoder.named_modules():
            m.share_cache = self.share_cache
            m.key = f'decoder_{n}'
        # for ref
        for n, m in self.ori_encoder.named_modules():
            m.share_cache = self.share_cache
            m.key = f'encoder_{n}'

        for n, m in self.ori_decoder.named_modules():
            m.share_cache = self.share_cache
            m.key = f'decoder_{n}'

        all_ref_related_modules = [
            self.ref_decoder, self.ref_middle, self.ori_decoder,
            self.ori_middle
        ]

        def forward_with_ref(self,
                             x,
                             context=None,
                             mask=None,
                             caching=None,
                             style_fidelity=None,
                             **kwargs):
            # self-attention
            if self.disable_self_attn:
                x = x + self.self_attn(self.norm1(x), context)
            else:
                y = self.norm1(x)
                if self.share_cache['mode'] == 'pr':

                    x = x + self.self_attn(y)

                    num_diff_condition = self.share_cache['num_diff_condition']

                    b, l, c = y.shape

                    ref = self.share_cache[self.key]
                    ref = ref[None].repeat_interleave(b, dim=0).flatten(0, 1)
                    # Batch * (L*num_ref) * C
                    ref = ref.reshape(b, -1, c)
                    ctx = torch.cat([y, ref], dim=1)
                    if num_diff_condition == 2:
                        x = x + (self.self_attn.self_attn_first(
                            y, ctx)) * self.share_cache['similarity'] + (
                                1 - self.share_cache['similarity']
                            ) * self.self_attn.self_attn_first(y)
                    elif num_diff_condition == 3:
                        similarity = self.share_cache.get('similarity')
                        similarity = similarity.new_tensor([similarity, 0, 0])
                        num_samples = b // 3
                        similarity = similarity[:, None].repeat_interleave(
                            num_samples, 1).flatten(0, 1)
                        similarity = similarity[:, None, None]

                        x = x + (self.self_attn.self_attn_first(
                            y, ctx)) * similarity + (
                                1 -
                                similarity) * self.self_attn.self_attn_first(y)

                elif self.share_cache['mode'] == 'w':
                    # TODO check this, should I detach the ref?
                    self.share_cache[self.key] = y
                    x = x + self.self_attn(y)
                else:
                    raise ValueError(
                        f"Unknown mode {self.share_cache['mode']}")

            # cross-attention & ffn

            x = x + self.cross_attn(self.norm2(x), context, mask)

            x = x + self.ffn(self.norm3(x))
            return x

        for coder in all_ref_related_modules:
            for n, m in coder.named_modules():
                if isinstance(m, AttentionBlock):
                    m.forward = forward_with_ref.__get__(m, m.__class__)

        all_ori_related_modules = [self.ori_decoder, self.ori_middle]

        for coder in all_ori_related_modules:
            for n, m in coder.named_modules():
                if isinstance(m,
                              MultiHeadAttention) and n.endswith('self_attn'):

                    m.self_attn_first = copy.deepcopy(m)
                    for p in m.self_attn_first.parameters():
                        p.requires_grad = True

    def switch_mode(self, mode='w'):
        if mode == 'w':
            self.share_cache['mode'] = 'w'
            self.encoder = self.ref_encoder
            self.decoder = self.ref_decoder
        elif mode == 'pr':
            self.share_cache['mode'] = 'pr'
            self.encoder = self.ori_encoder
            self.decoder = self.ori_decoder
        elif mode == 'nr':
            self.share_cache['mode'] = 'nr'
            self.encoder = self.ori_encoder
            self.decoder = self.ori_decoder
        else:
            raise ValueError(f'Unknown mode {mode}')

    def forward(self,
                x,
                t,
                y=None,
                context=None,
                mask=None,
                caching=None,
                style_fidelity=0.5):
        num_sample = t.shape[0]
        num_diff_condition = context.shape[0] // num_sample
        assert num_diff_condition == 2 or num_diff_condition == 3
        t = t.repeat_interleave(num_diff_condition, dim=0)
        # embeddings
        self.share_cache['num_diff_condition'] = num_diff_condition
        num_refs = self.share_cache['num_pairs']

        similarity = self.share_cache.get('similarity', 1)

        e = self.time_embedding(sinusoidal_embedding(t, self.dim))

        self.switch_mode('w')
        ref_x = self.share_cache['ref']
        ref_context = self.share_cache['ref_context']

        ref_e = e[:1].repeat_interleave(num_refs, dim=0)

        self.share_cache['similarity'] = similarity

        ref_e = ref_e
        # encoder-decoder
        args = (ref_e, ref_context, mask, caching, style_fidelity)

        ref_x, *xs = self.encode(ref_x, *args)
        self.decode(ref_x, *args, *xs)

        self.switch_mode('pr')
        args = (e, context, mask, caching, style_fidelity)
        masks = self.share_cache['masks']
        masks = masks.repeat_interleave(num_diff_condition, dim=0)
        cuda_mask = masks.cuda()[:, None]
        x = torch.cat([x, cuda_mask], dim=1)
        x, *xs = self.encode(x, *args)
        x = self.decode(x, *args, *xs)

        return x


def sd_v1_ref_unet(pretrained=False,
                   version='sd-v1-5_ema',
                   device='cpu',
                   enable_encoder=False,
                   **kwargs):
    """UNet of Stable Diffusion 1.x (1.1~1.5)."""
    # sanity check
    assert version in ('sd-v1-1_ema', 'sd-v1-1_nonema', 'sd-v1-2_ema',
                       'sd-v1-2_nonema', 'sd-v1-3_ema', 'sd-v1-3_nonema',
                       'sd-v1-4_ema', 'sd-v1-4_nonema', 'sd-v1-5_ema',
                       'sd-v1-5_nonema', 'sd-v1-5-inpainting_nonema')

    # dedue dimension
    in_dim = 4
    if 'inpainting' in version:
        in_dim = 9

    # init model
    cfg = dict(in_dim=in_dim,
               dim=320,
               y_dim=None,
               context_dim=768,
               out_dim=4,
               dim_mult=[1, 2, 4, 4],
               num_heads=8,
               head_dim=None,
               num_res_blocks=2,
               num_attn_blocks=1,
               attn_scales=[1 / 4, 1 / 2, 1],
               dropout=0.0)
    cfg.update(**kwargs)
    model = RefStableUNet(**cfg).to(device)
    model.init_refnet(copy_model=True, enable_encoder=enable_encoder)

    return model
