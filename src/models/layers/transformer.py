# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Take the standard Transformer as T2T Transformer
"""

import pdb
import torch
import torch.nn as nn

from loguru import logger
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm='nn.LayerNorm'):
        super(PreNorm, self).__init__()
        # self.norm = nn.LayerNorm(dim)
        self.norm = eval(norm)(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super(Attention, self).__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # if mask is not None:
        #     dots = dots.mul(mask)

        if mask is not None:
            v = v.mul(mask)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super(Cross_Attention, self).__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, token=None, mask=None):
        if token is None:
            logger.error(f'token is None')
            exit()

        q = self.to_q(token)
        k, v = self.to_kv(x).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, dim_mlp, depth=1, dropout=0.):
        super(EncoderLayer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, dim_mlp, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x, mask=None) + x
            x = ff(x) + x
        return x

class DecoderLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, dim_mlp, depth=1, dropout=0.):
        super(DecoderLayer, self).__init__()
        self.heads = heads
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, Cross_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, dim_mlp, dropout = dropout))
            ]))
    def forward(self, x, token, mask=None):
        for attn, cross_attn, ff in self.layers:
            token = attn(token, mask=mask) + token
            x = cross_attn(x, token=token, mask=None) + x
            x = ff(x) + x

        cam_shape_feats = x[:, 0]
        point_local_feat = rearrange(cam_shape_feats, 'b (h d) -> b d h', h = self.heads).contiguous()
        return point_local_feat, cam_shape_feats

class CLS_Query_Attention(nn.Module):
    def __init__(self, dim=768, dim_head=768, dim_mlp=768, heads=24, dropout=0.):
        super(CLS_Query_Attention, self).__init__()
        self.scale = dim ** -0.5

        self.cls_token = nn.Parameter(torch.randn(1, heads, dim_head))

        self.attend = nn.Softmax(dim=-1)

        self.to_q = nn.Sequential(
            nn.Linear(dim, dim_head, bias = False),
        )
        self.to_k = nn.Sequential(
            nn.Linear(dim, dim_head, bias = False),
        )
        self.to_v = nn.Sequential(
            nn.Linear(dim, dim_head, bias = False),
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim_head, dim_mlp),
            nn.Dropout(dropout)
        ) if dim_head != dim_mlp else nn.Identity()

    def forward(self, x):
        b, n, d = x.shape

        cls_head = repeat(self.cls_token, '() h d -> b h d', b=b)

        q = self.to_q(cls_head)
        k = self.to_k(x)
        v = self.to_v(x)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        return self.to_out(out), attn

class CLSEncoderLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, dim_mlp, depth=0, dropout=0.):
        super(CLSEncoderLayer, self).__init__()
        self.attn = PreNorm(
            dim,
            CLS_Query_Attention(
                input_dim=dim, 
                hidden_dim=dim_head, 
                output_dim=dim_mlp, 
                cls_num=heads, 
                dropout=dropout
            ),
            norm='nn.Identity'
        )

    def forward(self, x):
        x, attn = self.attn(x)
        return x, attn