"""Concise re-implementation of ``https://github.com/openai/CLIP'' and
``https://github.com/mlfoundations/open_clip''."""
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from ..ops.utils import scaled_dot_product_attention

__all__ = ['clip_vit_l_14']


class QuickGELU(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class SelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 causal=False,
                 attn_dropout=0.0,
                 proj_dropout=0.0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout

        # layers
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x:   [B, L, C].
        """
        b, s, c, n, d = *x.size(), self.num_heads, self.head_dim

        # compute query, key, value
        q, k, v = self.to_qkv(x).view(b, s, 3, n, d).unbind(2)

        # compute attention
        p = self.attn_dropout if self.training else 0.0
        x = scaled_dot_product_attention(q,
                                         k,
                                         v,
                                         dropout_p=p,
                                         is_causal=self.causal)
        x = x.reshape(b, s, c)

        # output
        x = self.proj(x)
        x = F.dropout(x, self.proj_dropout, self.training)
        return x


class SwiGLU(nn.Module):

    def __init__(self, dim, mid_dim):
        super().__init__()
        self.dim = dim
        self.mid_dim = mid_dim

        # layers
        self.fc1 = nn.Linear(dim, mid_dim)
        self.fc2 = nn.Linear(dim, mid_dim)
        self.fc3 = nn.Linear(mid_dim, dim)

    def forward(self, x):
        x = F.silu(self.fc1(x)) * self.fc2(x)
        x = self.fc3(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio,
                 num_heads,
                 post_norm=False,
                 causal=False,
                 activation='quick_gelu',
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 norm_eps=1e-5):
        assert activation in ['quick_gelu', 'gelu', 'swi_glu']
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.post_norm = post_norm
        self.causal = causal
        self.norm_eps = norm_eps

        # layers
        self.norm1 = LayerNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(dim, num_heads, causal, attn_dropout,
                                  proj_dropout)
        self.norm2 = LayerNorm(dim, eps=norm_eps)
        if activation == 'swi_glu':
            self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio)),
                QuickGELU() if activation == 'quick_gelu' else nn.GELU(),
                nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(proj_dropout))

    def forward(self, x):
        if self.post_norm:
            x = x + self.norm1(self.attn(x))
            x = x + self.norm2(self.mlp(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class AttentionPool(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio,
                 num_heads,
                 activation='gelu',
                 proj_dropout=0.0,
                 norm_eps=1e-5):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj_dropout = proj_dropout
        self.norm_eps = norm_eps

        # layers
        gain = 1.0 / math.sqrt(dim)
        self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.norm = LayerNorm(dim, eps=norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            QuickGELU() if activation == 'quick_gelu' else nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(proj_dropout))

    def forward(self, x):
        """
        x:  [B, L, C].
        """
        b, s, c, n, d = *x.size(), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.to_q(self.cls_embedding).view(1, 1, n,
                                               d).expand(b, -1, -1, -1)
        k, v = self.to_kv(x).view(b, s, 2, n, d).unbind(2)

        # compute attention
        x = scaled_dot_product_attention(q, k, v)
        x = x.reshape(b, 1, c)

        # output
        x = self.proj(x)
        x = F.dropout(x, self.proj_dropout, self.training)

        # mlp
        x = x + self.mlp(self.norm(x))
        return x[:, 0]


class VisionTransformer(nn.Module):

    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 dim=768,
                 mlp_ratio=4,
                 out_dim=512,
                 num_heads=12,
                 num_layers=12,
                 pool_type='token',
                 pre_norm=True,
                 post_norm=False,
                 activation='quick_gelu',
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0,
                 norm_eps=1e-5):
        if image_size % patch_size != 0:
            print('[WARNING] image_size is not divisible by patch_size',
                  flush=True)
        assert pool_type in ('token', 'token_fc', 'attn_pool')
        out_dim = out_dim or dim
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size)**2
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pool_type = pool_type
        self.post_norm = post_norm
        self.norm_eps = norm_eps

        # embeddings
        gain = 1.0 / math.sqrt(dim)
        self.patch_embedding = nn.Conv2d(3,
                                         dim,
                                         kernel_size=patch_size,
                                         stride=patch_size,
                                         bias=not pre_norm)
        if pool_type in ('token', 'token_fc'):
            self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(gain * torch.randn(
            1, self.num_patches +
            (1 if pool_type in ('token', 'token_fc') else 0), dim))
        self.dropout = nn.Dropout(embedding_dropout)

        # transformer
        self.pre_norm = LayerNorm(dim, eps=norm_eps) if pre_norm else None
        self.transformer = nn.Sequential(*[
            AttentionBlock(dim, mlp_ratio, num_heads, post_norm, False,
                           activation, attn_dropout, proj_dropout, norm_eps)
            for _ in range(num_layers)
        ])
        self.post_norm = LayerNorm(dim, eps=norm_eps)

        # head
        if pool_type == 'token':
            self.head = nn.Parameter(gain * torch.randn(dim, out_dim))
        elif pool_type == 'token_fc':
            self.head = nn.Linear(dim, out_dim)
        elif pool_type == 'attn_pool':
            self.head = AttentionPool(dim, mlp_ratio, num_heads, activation,
                                      proj_dropout, norm_eps)

    def forward(self, x):
        b = x.size(0)

        # patch-embedding
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        if self.pool_type in ('token', 'token_fc'):
            x = torch.cat([self.cls_embedding.expand(b, -1, -1), x], dim=1)
        x = self.dropout(x + self.pos_embedding)
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        # transformer
        x = self.transformer(x)

        # head
        x = self.post_norm(x)
        if self.pool_type == 'token':
            x = torch.mm(x[:, 0, :], self.head)
        elif self.pool_type == 'token_fc':
            x = self.head(x[:, 0, :])
        elif self.pool_type == 'attn_pool':
            x = self.head(x)
        else:
            raise ValueError(f'Unexpected pool_type {self.pool_type}')
        return x


class TextTransformer(nn.Module):

    def __init__(self,
                 vocab_size,
                 text_len,
                 dim=512,
                 mlp_ratio=4,
                 out_dim=512,
                 num_heads=8,
                 num_layers=12,
                 causal=True,
                 pool_type='argmax',
                 head_bias=False,
                 activation='quick_gelu',
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0,
                 norm_eps=1e-5):
        assert pool_type in ('argmax', 'last')
        out_dim = out_dim or dim
        super().__init__()
        self.vocab_size = vocab_size
        self.text_len = text_len
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.causal = causal
        self.pool_type = pool_type
        self.head_bias = head_bias
        self.norm_eps = norm_eps

        # embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Parameter(0.01 * torch.randn(1, text_len, dim))
        self.dropout = nn.Dropout(embedding_dropout)

        # transformer
        self.transformer = nn.Sequential(*[
            AttentionBlock(dim, mlp_ratio, num_heads, False, causal,
                           activation, attn_dropout, proj_dropout, norm_eps)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(dim, eps=norm_eps)

        # head
        if head_bias:
            self.head = nn.Linear(dim, out_dim)
        else:
            gain = 1.0 / math.sqrt(dim)
            self.head = nn.Parameter(gain * torch.randn(dim, out_dim))

    def forward(self, x):
        index = x.argmax(dim=-1) if self.pool_type == 'argmax' else -1

        # embeddings
        x = self.dropout(self.token_embedding(x) + self.pos_embedding)

        # transformer
        x = self.transformer(x)

        # head
        x = self.norm(x)
        x = x[torch.arange(x.size(0)), index]
        if self.head_bias:
            x = self.head(x)
        else:
            x = torch.mm(x, self.head)
        return x


class CLIP(nn.Module):

    def __init__(self,
                 embed_dim=512,
                 image_size=224,
                 patch_size=16,
                 vision_dim=768,
                 vision_mlp_ratio=4,
                 vision_heads=12,
                 vision_layers=12,
                 vision_pool='token',
                 vision_pre_norm=True,
                 vision_post_norm=False,
                 vocab_size=49408,
                 text_len=77,
                 text_dim=512,
                 text_mlp_ratio=4,
                 text_heads=8,
                 text_layers=12,
                 text_causal=True,
                 text_pool='argmax',
                 text_head_bias=False,
                 logit_bias=None,
                 activation='quick_gelu',
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0,
                 norm_eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_dim = vision_dim
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_heads = vision_heads
        self.vision_layers = vision_layers
        self.vision_pool = vision_pool
        self.vision_pre_norm = vision_pre_norm
        self.vision_post_norm = vision_post_norm
        self.vocab_size = vocab_size
        self.text_len = text_len
        self.text_dim = text_dim
        self.text_mlp_ratio = text_mlp_ratio
        self.text_heads = text_heads
        self.text_layers = text_layers
        self.text_causal = text_causal
        self.text_pool = text_pool
        self.text_head_bias = text_head_bias
        self.norm_eps = norm_eps

        # models
        self.visual = VisionTransformer(image_size=image_size,
                                        patch_size=patch_size,
                                        dim=vision_dim,
                                        mlp_ratio=vision_mlp_ratio,
                                        out_dim=embed_dim,
                                        num_heads=vision_heads,
                                        num_layers=vision_layers,
                                        pool_type=vision_pool,
                                        pre_norm=vision_pre_norm,
                                        post_norm=vision_post_norm,
                                        activation=activation,
                                        attn_dropout=attn_dropout,
                                        proj_dropout=proj_dropout,
                                        embedding_dropout=embedding_dropout,
                                        norm_eps=norm_eps)
        self.textual = TextTransformer(vocab_size=vocab_size,
                                       text_len=text_len,
                                       dim=text_dim,
                                       mlp_ratio=text_mlp_ratio,
                                       out_dim=embed_dim,
                                       num_heads=text_heads,
                                       num_layers=text_layers,
                                       causal=text_causal,
                                       pool_type=text_pool,
                                       head_bias=text_head_bias,
                                       activation=activation,
                                       attn_dropout=attn_dropout,
                                       proj_dropout=proj_dropout,
                                       embedding_dropout=embedding_dropout,
                                       norm_eps=norm_eps)
        self.log_scale = nn.Parameter(math.log(1 / 0.07) * torch.ones([]))
        if logit_bias is not None:
            self.logit_bias = nn.Parameter(logit_bias * torch.ones([]))

        # initialize weights
        self.init_weights()

    def forward(self, imgs, txt_tokens):
        """
        imgs:       [B, 3, H, W] of torch.float32.
                    mean:   [0.48145466, 0.4578275, 0.40821073]
                    std:    [0.26862954, 0.26130258, 0.27577711]
        txt_tokens: [B, L] of torch.long.
                    Encoded by data.CLIPTokenizer.
        """
        xi = self.visual(imgs)
        xt = self.textual(txt_tokens)
        return xi, xt

    def init_weights(self):
        # embeddings
        nn.init.normal_(self.textual.token_embedding.weight, std=0.02)
        nn.init.normal_(self.visual.patch_embedding.weight, std=0.1)

        # attentions
        for modality in ['visual', 'textual']:
            dim = self.vision_dim if modality == 'visual' else self.text_dim
            transformer = getattr(self, modality).transformer
            proj_gain = (1.0 / math.sqrt(dim)) * (
                1.0 / math.sqrt(2 * len(transformer)))
            attn_gain = 1.0 / math.sqrt(dim)
            mlp_gain = 1.0 / math.sqrt(2.0 * dim)
            for block in transformer:
                nn.init.normal_(block.attn.to_qkv.weight, std=attn_gain)
                nn.init.normal_(block.attn.proj.weight, std=proj_gain)
                nn.init.normal_(block.mlp[0].weight, std=mlp_gain)
                nn.init.normal_(block.mlp[2].weight, std=proj_gain)

    def param_groups(self):
        groups = [{
            'params': [
                p for n, p in self.named_parameters()
                if 'norm' in n or n.endswith('bias')
            ],
            'weight_decay':
            0.0
        }, {
            'params': [
                p for n, p in self.named_parameters()
                if not ('norm' in n or n.endswith('bias'))
            ]
        }]
        return groups


def _clip(pretrained=False,
          pretrained_name=None,
          model_cls=CLIP,
          return_transforms=False,
          return_tokenizer=False,
          tokenizer_padding='eos',
          dtype=torch.float32,
          device='cpu',
          **kwargs):
    # init a meta model

    model = model_cls(**kwargs)

    # load checkpoint
    if pretrained and pretrained_name:
        path = Path(__file__).parents[4] / "models" / "clip" / "openai-clip-vit-large-14.pth"
        assert pretrained_name in str(path)
        # load
        model.load_state_dict(torch.load(path,
                                         map_location=device,
                                         weights_only=True),
                              strict=False)

    # set device

    model = model.to(dtype=dtype, device=device)
    output = (model, )

    # init transforms
    if return_transforms:
        # mean and std
        if 'siglip' in pretrained_name.lower():
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        else:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        # transforms
        transforms = T.Compose([
            T.Resize((model.image_size, model.image_size),
                     interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        output += (transforms, )

    # init tokenizer
    if return_tokenizer:
        from ldm import data
        if 'siglip' in pretrained_name.lower():
            tokenizer = data.HuggingfaceTokenizer(
                name=f'timm/{pretrained_name}',
                length=model.text_len,
                clean='canonicalize')
        elif 'xlm' in pretrained_name.lower():
            tokenizer = data.HuggingfaceTokenizer(name='xlm-roberta-large',
                                                  length=model.max_text_len -
                                                  2,
                                                  clean='whitespace')
        elif 'mba' in pretrained_name.lower():
            tokenizer = data.HuggingfaceTokenizer(
                name='facebook/xlm-roberta-xl',
                length=model.max_text_len - 2,
                clean='whitespace')
        else:
            tokenizer = data.CLIPTokenizer(length=model.text_len,
                                           padding=tokenizer_padding)
        output += (tokenizer, )
    return output[0] if len(output) == 1 else output


def clip_vit_l_14(pretrained=False,
                  pretrained_name='openai-clip-vit-large-14',
                  **kwargs):
    cfg = dict(embed_dim=768,
               image_size=224,
               patch_size=14,
               vision_dim=1024,
               vision_heads=16,
               vision_layers=24,
               vocab_size=49408,
               text_len=77,
               text_dim=768,
               text_heads=12,
               text_layers=12,
               activation='quick_gelu')
    cfg.update(**kwargs)
    return _clip(pretrained, pretrained_name, **cfg)
