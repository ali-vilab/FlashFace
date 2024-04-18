"""Concise re-implementation of autoencoders and the discriminator from
``https://github.com/Stability-AI/stablediffusion'',
``https://github.com/CompVis/latent-diffusion'', and
``https://github.com/Stability-AI/generative-models''."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops.utils import scaled_dot_product_attention

__all__ = [
    'sd_v1_vae',
]


def group_norm(dim):
    return nn.GroupNorm(32, dim, eps=1e-6, affine=True)


class Upsample(nn.Upsample):

    def forward(self, x):
        """Fix bfloat16 support for nearest neighbor interpolation."""
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ['none', 'upsample', 'downsample']
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample':
            self.resample = nn.Sequential(
                Upsample(scale_factor=2.0, mode='nearest'),
                nn.Conv2d(dim, dim, 3, padding=1))
        elif mode == 'downsample':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=2, padding=0))
        else:
            self.resample = nn.Identity()

    def forward(self, x):
        return self.resample(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            group_norm(in_dim), nn.SiLU(),
            nn.Conv2d(in_dim, out_dim, 3, padding=1), group_norm(out_dim),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, 3, padding=1))
        self.shortcut = nn.Conv2d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention with a single head."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = group_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()

        # compute query, key, value
        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(b, 1, c * 3, -1).permute(
            0, 3, 1, 2).contiguous().chunk(3, dim=-1)

        # apply attention
        x = scaled_dot_product_attention(q, k, v)
        x = x.squeeze(2).permute(0, 2, 1)
        x = x.permute(0, 2, 1).reshape(b, c, h, w)

        # output
        x = self.proj(x)
        return x + identity


class Encoder(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = nn.Conv2d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                downsamples.append(Resample(out_dim, mode='downsample'))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(ResidualBlock(out_dim, out_dim, dropout),
                                    AttentionBlock(out_dim),
                                    ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(group_norm(out_dim), nn.SiLU(),
                                  nn.Conv2d(out_dim, z_dim, 3, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.downsamples(x)
        x = self.middle(x)
        x = self.head(x)
        return x


class Decoder(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = nn.Conv2d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(ResidualBlock(dims[0], dims[0], dropout),
                                    AttentionBlock(dims[0]),
                                    ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                upsamples.append(Resample(out_dim, 'upsample'))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(group_norm(out_dim), nn.SiLU(),
                                  nn.Conv2d(out_dim, 3, 3, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.middle(x)
        x = self.upsamples(x)
        x = self.head(x)
        return x


class AutoencoderKL(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales

        # blocks
        self.encoder = Encoder(dim, z_dim * 2, dim_mult, num_res_blocks,
                               attn_scales, dropout)
        self.conv1 = nn.Conv2d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = nn.Conv2d(z_dim, z_dim, 1)
        self.decoder = Decoder(dim, z_dim, dim_mult, num_res_blocks,
                               attn_scales, dropout)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x = self.decode(z)
        return x, mu, log_var

    def encode(self, x):
        x = self.encoder(x)
        mu, log_var = self.conv1(x).chunk(2, dim=1)
        return mu, log_var

    def decode(self, z):
        x = self.conv2(z)
        x = self.decoder(x)
        return x

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        """Sample latents given the input images."""
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)


def sd_v1_vae(pretrained=False, device='cpu', **kwargs):
    """Autoencoder of Stable Diffusion 1.x (1.1~1.5 share a same
    autoencoder)."""
    cfg = dict(dim=128,
               z_dim=4,
               dim_mult=[1, 2, 4, 4],
               num_res_blocks=2,
               attn_scales=[],
               dropout=0.0)
    cfg.update(**kwargs)
    model = AutoencoderKL(**cfg).to(device)
    if pretrained:
        model.load_state_dict(
            torch.load(__file__.replace('ldm/models/vae.py',
                                        'cache/sd-v1-vae.pth'),
                       map_location=device))
    return model
