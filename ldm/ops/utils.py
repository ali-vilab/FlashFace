import importlib.util
import math

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F


def scaled_dot_product_attention(q,
                                 k,
                                 v,
                                 attn_mask=None,
                                 dropout_p=0.,
                                 is_causal=False,
                                 flash_dtyp=torch.float16):
    """
    q:  [B, L1, N, C1].
    k:  [B, L2, N, C1].
    v:  [B, L2, N, C2].
    attn_mask:  [B, ..., L1, L2].
    """
    b, l1, l2 = len(q), q.size(1), k.size(1)
    if importlib.util.find_spec('flash_attn') is not None and \
            q.device.type == 'cuda' and q.size(-1) <= 256 and attn_mask is None:
        from flash_attn import flash_attn_func

        def half(x):
            return x.to(flash_dtyp) if x.dtype not in (torch.float16,
                                                       torch.bfloat16) else x

        # flash attention
        with amp.autocast():
            x = flash_attn_func(q=half(q),
                                k=half(k),
                                v=half(v),
                                dropout_p=dropout_p,
                                causal=is_causal)

        # convert the data type back
        if x.dtype != q.dtype:
            x = x.to(q.dtype)
    elif torch.__version__.startswith('2.'):
        # process mask
        if attn_mask is not None and is_causal:
            attn_mask = attn_mask.view(b, -1, l1, l2).tril()
            is_causal = False

        # compute attention
        x = F.scaled_dot_product_attention(
            query=q.transpose(1, 2).contiguous(),
            key=k.transpose(1, 2).contiguous(),
            value=v.transpose(1, 2).contiguous(),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal).transpose(1, 2)
    else:
        attn = torch.einsum('binc,bjnc->bnij', q, k) / math.sqrt(l2)

        # apply mask
        if attn_mask is not None:
            attn_mask = attn_mask.view(b, -1, l1, l2)
            if attn_mask.dtype == torch.bool:
                attn = attn.masked_fill(attn_mask == 0, float('-inf'))
            else:
                attn = attn + attn_mask

        # causal mask
        if is_causal:
            attn = attn.masked_fill(
                torch.tril(attn.new_ones(1, 1, l1,
                                         l2).float()).type_as(attn) == 0,
                float('-inf'))

        # gather context
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum('bnij,bjnc->binc', attn, v)
    return x.contiguous()
