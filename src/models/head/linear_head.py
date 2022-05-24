import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from loguru import logger
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from ..layers import LocallyConnected2d, Conv3x3, Conv1x1, ConvInverse2x2, ConvUpsample3x3
from ..layers.transformer import CLS_Query_Attention

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class TokenHead(nn.Module):
    def __init__(
        self,
        input_res, 
        input_dim,
        filters,
        patch_size,
        num_cls,
        mask_ratio,
        align_corners: bool = True,
        visualize: bool = False,
    ):
        super(TokenHead, self).__init__()
        self.num_cls = num_cls
        self.masking_ratio = mask_ratio

        image_height, image_width = pair(input_res)
        patch_height, patch_width = pair(patch_size)
        token_height, token_width = (image_height // patch_height, image_width // patch_width)
        self.num_patches = token_height * token_width

        patch_dim = input_dim * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, filters[0]) if patch_dim != filters[0] else nn.Identity(),
        )
        self.encoder_pos_emb = nn.Parameter(torch.randn(1, self.num_patches, filters[0]))
        self.encoder = CLS_Query_Attention(
            dim=filters[0], 
            dim_head=filters[1], 
            dim_mlp=filters[2], 
            heads=num_cls, 
        )

    def forward(self, img):
        """Forward function."""
        # get patches to encoder tokens and add positions
        tokens = self.to_patch_embedding(img)
        tokens += self.encoder_pos_emb
        b, n, d = tokens.shape
        device = img.device

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        if self.training:
            masking_ratio = torch.randn(1)*0.75 if self.masking_ratio < 0 else self.masking_ratio
            num_masked = int(masking_ratio * n)
        else:
            num_masked = 0
        rand_indices = torch.rand(b, n, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(b, device=device)[:, None]
        unmasked_tokens = tokens[batch_range, unmasked_indices]
        # attend with vision transformer
        unmasked_tokens, unmasked_attn = self.encoder(unmasked_tokens)

        # un-shuffle attention map for visualize
        batch_attn = rearrange(torch.arange(b, device=device), 'b -> b 1 1')
        num_attn = rearrange(torch.arange(self.num_cls, device=device), 'n -> 1 n 1')
        unmask_attn = rearrange(unmasked_indices, 'b d -> b 1 d')
        mask_attn = rearrange(masked_indices, 'b d -> b 1 d')
        attn = torch.zeros([b, self.num_cls, n]).type_as(img)
        attn[batch_attn, num_attn, mask_attn] = -1.
        attn[batch_attn, num_attn, unmask_attn] = unmasked_attn

        return unmasked_tokens, attn

    def _get_downsample_layer(self, in_channels, filters, kernels, align_corners=False):
        assert len(filters) == len(kernels), 'ERROR: length of filters is different with kernels'

        def _get_conv_cfg(kernel):
            padding_cfg = {7:3, 5:2, 3:1, 1:0}
            return kernel, padding_cfg[kernel]

        planes_in = in_channels
        layers = []
        for i in range(len(filters)):
            kernel, padding = _get_conv_cfg(kernels[i])
            planes = filters[i]
            layers.append(
                nn.Conv2d(
                    in_channels=planes_in,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=1,
                    padding=padding,
                    bias=False
                )
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU())
            planes_in = planes

        return nn.Sequential(*layers)

    def _get_upsample_layer(self, in_channels, filters, kernels, align_corners=True):
        assert len(filters) == len(kernels), 'ERROR: length of filters is different with kernels'

        def _get_conv_cfg(kernel):
            padding_cfg = {7:3, 5:2, 3:1, 1:0}
            return kernel, padding_cfg[kernel]

        planes_in = in_channels
        layers = []
        for i in range(len(filters)):
            kernel, padding = _get_conv_cfg(kernels[i])
            planes = filters[i]
            layers.append(
                nn.Upsample(
                    scale_factor=2, 
                    mode='bilinear', 
                    align_corners=align_corners
                )
            )
            layers.append(
                nn.Conv2d(
                    in_channels=planes_in,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=1,
                    padding=padding,
                    bias=False
                )
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU())
            planes_in = planes

        return nn.Sequential(*layers)