"""Concise reimplementation of UNet models from ``https://github.com/Stability-
AI/stablediffusion'' and ``https://github.com/CompVis/stable-diffusion''."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..ops.utils import scaled_dot_product_attention

__all__ = ['UNet']


def sinusoidal_embedding(timesteps, dim):
    # check input
    half = dim // 2
    timesteps = timesteps.float()

    # compute sinusoidal embedding
    sinusoid = torch.outer(
        timesteps, torch.pow(10000,
                             -torch.arange(half).to(timesteps).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    if dim % 2 != 0:
        x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
    return x


class GroupNorm(nn.GroupNorm):

    def __init__(self, num_groups, dim, eps=1e-6, affine=True, **kwargs):
        super().__init__(num_groups, dim, eps, affine, **kwargs)

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class Upsample(nn.Module):

    def __init__(self, mode='nearest'):
        super().__init__()
        self.mode = mode

    def forward(self, x, reference):
        """Nearest neighbor interpolation not implemented for bfloat16
        therefore we convert x to float32 first."""
        return F.interpolate(x.float(),
                             size=reference.shape[-2:],
                             mode=self.mode).type_as(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, embed_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        # layers
        self.layer1 = nn.Sequential(GroupNorm(32, in_dim), nn.SiLU(),
                                    nn.Conv2d(in_dim, out_dim, 3, padding=1))
        self.embedding = nn.Sequential(nn.SiLU(),
                                       nn.Linear(embed_dim, out_dim))
        self.layer2 = nn.Sequential(GroupNorm(32, out_dim), nn.SiLU(),
                                    nn.Dropout(dropout),
                                    nn.Conv2d(out_dim, out_dim, 3, padding=1))
        self.shortcut = nn.Identity() if in_dim == out_dim \
            else nn.Conv2d(in_dim, out_dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.layer2[-1].weight)

    def forward(self, x, e):
        identity = x
        x = self.layer1(x)
        x = x + self.embedding(e).unsqueeze(-1).unsqueeze(-1).type_as(x)
        x = self.layer2(x)
        x = x + self.shortcut(identity)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 dim,
                 context_dim=None,
                 num_heads=None,
                 head_dim=None,
                 dropout=0.0):
        # consider head_dim first, then num_heads
        num_heads = dim // head_dim if head_dim else num_heads
        head_dim = dim // num_heads
        assert num_heads * head_dim == dim
        context_dim = context_dim or dim
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = math.pow(head_dim, -0.25)

        # layers
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(context_dim, dim, bias=False)
        self.v = nn.Linear(context_dim, dim, bias=False)
        self.o = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None):
        """
        x:       [B, L1, C1].
        context: [B, L2, C2] or None.
        mask:    [B, L2] or None.
        """
        context = x if context is None else context
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).view(b, -1, n, d)
        k = self.k(context).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        p = self.dropout.p if self.training else 0.0
        if mask is not None:
            mask = mask.view(b, 1, 1, -1).expand(-1, n, q.size(1), -1)
        x = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        x = x.view(b, -1, n * d)

        # output
        x = self.o(x)
        x = self.dropout(x)
        return x


class GeGLU(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class AttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 context_dim=None,
                 num_heads=None,
                 head_dim=None,
                 disable_self_attn=False,
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.disable_self_attn = disable_self_attn

        # self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, None, num_heads, head_dim,
                                            dropout)

        # cross-attention
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, context_dim, num_heads,
                                             head_dim, dropout)

        # ffn
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(GeGLU(dim, dim * 4), nn.Dropout(dropout),
                                 nn.Linear(dim * 4, dim))

        # cache to support reference-only
        self.cache = None

    def forward(self,
                x,
                context=None,
                mask=None,
                caching=None,
                style_fidelity=0.5):
        assert caching in (None, 'write', 'read')

        # self-attention
        if self.disable_self_attn:
            x = x + self.self_attn(self.norm1(x), context, mask)
        else:
            y = self.norm1(x)
            if caching == 'read':
                assert self.cache is not None

                # read cache & self-attention
                ctx = torch.cat([
                    y,
                    self.cache.expand(len(y), -1, -1)
                    if len(self.cache) == 1 and len(y) > 1 else self.cache
                ],
                                dim=1)
                x = x + (self.self_attn(y, ctx) * (1 - style_fidelity) +
                         self.self_attn(y) * style_fidelity)
            else:
                if caching == 'write':
                    self.cache = y
                x = x + self.self_attn(y)

        # cross-attention & ffn
        x = x + self.cross_attn(self.norm2(x), context, mask)
        x = x + self.ffn(self.norm3(x))
        return x


class StackedAttentionBlocks(nn.Module):

    def __init__(self,
                 dim,
                 context_dim=None,
                 num_heads=None,
                 head_dim=None,
                 num_blocks=1,
                 disable_self_attn=False,
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.disable_self_attn = disable_self_attn

        # input
        self.norm1 = GroupNorm(32, dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        # blocks
        self.blocks = nn.ModuleList([
            AttentionBlock(dim, context_dim, num_heads, head_dim,
                           disable_self_attn, dropout)
            for _ in range(num_blocks)
        ])

        # output
        self.conv2 = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.conv2.weight)

    def forward(self,
                x,
                context=None,
                mask=None,
                caching=None,
                style_fidelity=0.5):
        assert caching in (None, 'write', 'read')
        b, c, h, w = x.size()
        identity = x

        # input
        x = self.norm1(x)
        x = self.conv1(x).view(b, c, -1).transpose(1, 2)

        # blocks
        for block in self.blocks:
            x = block(x, context, mask, caching, style_fidelity)

        # output
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.conv2(x)
        return x + identity


class UNet(nn.Module):

    def __init__(self,
                 in_dim=4,
                 dim=320,
                 y_dim=None,
                 context_dim=1024,
                 out_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_heads=None,
                 head_dim=64,
                 num_res_blocks=2,
                 num_attn_blocks=1,
                 middle_attn_blocks=None,
                 attn_scales=[1 / 4, 1 / 2, 1],
                 disable_self_attn=False,
                 use_checkpoint=False,
                 dropout=0.0):
        embed_dim = dim * 4
        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * len(dim_mult)
        if isinstance(num_attn_blocks, int):
            num_attn_blocks = [num_attn_blocks] * len(dim_mult)
        if middle_attn_blocks is None:
            middle_attn_blocks = num_attn_blocks[-1]
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.embed_dim = embed_dim
        self.y_dim = y_dim
        self.context_dim = context_dim
        self.out_dim = out_dim
        self.dim_mult = dim_mult
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_res_blocks = num_res_blocks
        self.num_attn_blocks = num_attn_blocks
        self.middle_attn_blocks = middle_attn_blocks
        self.attn_scales = attn_scales
        self.disable_self_attn = disable_self_attn
        self.use_checkpoint = use_checkpoint

        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0

        # embeddings
        self.time_embedding = nn.Sequential(nn.Linear(dim, embed_dim),
                                            nn.SiLU(),
                                            nn.Linear(embed_dim, embed_dim))
        if y_dim is not None:
            self.y_embedding = nn.Sequential(nn.Linear(y_dim, embed_dim),
                                             nn.SiLU(),
                                             nn.Linear(embed_dim, embed_dim))

        # encoder
        self.encoder = nn.ModuleList(
            [nn.Conv2d(self.in_dim, dim, 3, padding=1)])
        shortcut_dims.append(dim)
        shortcut_dims.append(dim)
        for i, (in_dim, out_dim) in enumerate(zip(enc_dims[:-1],
                                                  enc_dims[1:])):
            for j in range(num_res_blocks[i]):
                # residual (+attention) blocks
                block = nn.ModuleList(
                    [ResidualBlock(in_dim, embed_dim, out_dim, dropout)])
                if scale in attn_scales:
                    block.append(
                        StackedAttentionBlocks(out_dim, context_dim, num_heads,
                                               head_dim, num_attn_blocks[i],
                                               disable_self_attn, 0.0))
                in_dim = out_dim
                self.encoder.append(block)
                shortcut_dims.append(out_dim)

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks[i] - 1:
                    self.encoder.append(
                        nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1))
                    shortcut_dims.append(out_dim)
                    scale /= 2.0

        # middle
        self.middle = nn.ModuleList([
            ResidualBlock(out_dim, embed_dim, out_dim, dropout),
            StackedAttentionBlocks(out_dim, context_dim, num_heads, head_dim,
                                   middle_attn_blocks, disable_self_attn, 0.0),
            ResidualBlock(out_dim, embed_dim, out_dim, dropout)
        ])

        # decoder
        self.decoder = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dec_dims[:-1],
                                                  dec_dims[1:])):
            index = len(dim_mult) - i - 1
            for j in range(num_res_blocks[index] + 1):
                # residual (+attention) blocks
                block = nn.ModuleList([
                    ResidualBlock(in_dim + shortcut_dims.pop(), embed_dim,
                                  out_dim, dropout)
                ])
                if scale in attn_scales:
                    block.append(
                        StackedAttentionBlocks(out_dim, context_dim, num_heads,
                                               head_dim,
                                               num_attn_blocks[index],
                                               disable_self_attn, 0.0))
                in_dim = out_dim

                # upsample
                if i != len(dim_mult) - 1 and j == num_res_blocks[index]:
                    block.append(
                        nn.Sequential(
                            Upsample(mode='nearest'),
                            nn.Conv2d(out_dim, out_dim, 3, padding=1)))
                    scale *= 2.0
                self.decoder.append(block)

        # head
        self.head = nn.Sequential(
            GroupNorm(32, out_dim), nn.SiLU(),
            nn.Conv2d(out_dim, self.out_dim, 3, padding=1))

        # zero out the last layer params
        nn.init.zeros_(self.head[-1].weight)
        if y_dim is not None:
            nn.init.zeros_(self.y_embedding[-1].weight)

    def forward(self,
                x,
                t,
                y=None,
                context=None,
                mask=None,
                caching=None,
                style_fidelity=0.5):
        # embeddings
        e = self.time_embedding(sinusoidal_embedding(t, self.dim))
        if y is not None:
            e = e + self.y_embedding(y)

        # encoder-decoder
        args = (e, context, mask, caching, style_fidelity)
        if not (self.training and self.use_checkpoint):
            x, *xs = self.encode(x, *args)
            x = self.decode(x, *args, *xs)
        else:
            x, *xs = checkpoint(self.encode, x, *args)
            x = checkpoint(self.decode, x, *args, *xs)
        return x

    def encode(self, x, e, context, mask, caching, style_fidelity):
        args = (e, context, mask, caching, style_fidelity)

        # encoder
        xs = []
        for block in self.encoder:
            x = self._forward_single(block, x, *args)
            xs.append(x)

        # middle
        for block in self.middle:
            x = self._forward_single(block, x, *args)
        return (x, ) + tuple(xs)

    def decode(self, x, e, context, mask, caching, style_fidelity, *xs):
        args = (e, context, mask, caching, style_fidelity)
        xs = list(xs)

        # decoder
        for block in self.decoder:
            x = torch.cat([x, xs.pop()], dim=1)
            x = self._forward_single(block,
                                     x,
                                     *args,
                                     reference=xs[-1] if len(xs) > 0 else None)

        # head
        x = self.head(x)
        return x

    def _forward_single(self,
                        module,
                        x,
                        e,
                        context=None,
                        mask=None,
                        caching=None,
                        style_fidelity=None,
                        reference=None):
        if isinstance(module, ResidualBlock):
            x = module(x, e)
        elif isinstance(module, StackedAttentionBlocks):
            x = module(x, context, mask, caching, style_fidelity)
        elif isinstance(module, nn.Sequential) and isinstance(
                module[0], Upsample):
            x = module[0](x, reference)
            x = module[1:](x)
        elif isinstance(module, nn.ModuleList):
            for block in module:
                x = self._forward_single(block, x, e, context, mask, caching,
                                         style_fidelity, reference)
        else:
            x = module(x)
        return x
