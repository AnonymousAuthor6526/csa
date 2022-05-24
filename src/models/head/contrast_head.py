import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from loguru import logger
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from src.utils import kp_utils
from .linear_head import TokenHead
from .conv_head import OCR_Conv
from .smpl_head import SMPLHead
from .mlp_head import MLPHead


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class TargetPredictionHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        Conv network predict rgba+l iamges
    """
    def __init__(
        self,
        num_masks,
        align_corners=False,
        ):
        super(TargetPredictionHead, self).__init__()
        self.num_masks = num_masks
        # self.activation = torch.nn.Tanh()
        self.head = OCR_Conv(4+num_masks, align_corners)

    def forward(self, img):
        """Forward function."""
        rgbal = self.head(img)
        al = rgbal[:, 3:]
        al_mask = al.softmax(dim=1).round().bool()
        a_mask = al_mask[:, 0:1]

        o_mask = torch.logical_not(a_mask)
        rgb = rgbal[:, :3] * o_mask
        # rgb_norm = self.activation(rgb)
        rgb_norm = F.normalize(rgb, dim=1)
        rgba = torch.cat([rgb_norm, a_mask], dim=1)

        # self.debug_rend_out(rgbal, al, negative_a, rgb, rgba, 0)
        return {
            'target_rgba': rgba,
            'target_al': al_mask,
            'target_al_origin': al,
        }

    def debug_rend_out(
        self,
        rgbal, al, negative_a, 
        rgb, rgba, idx
    ):
        # Helper function used for visualization in the following examples
        def identify_axes(ax_dict, fontsize=8):
            """
            Helper to identify the Axes in the examples below.

            Draws the label in a large font in the center of the Axes.

            Parameters
            ----------
            ax_dict : dict[str, Axes]
                Mapping between the title / label and the Axes.
            fontsize : int, optional
                How big the label should be.
            """
            kw = dict(fontsize=fontsize, color="black")
            for k, ax in ax_dict.items():
                ax.text(0.05, 0.9, k, transform=ax.transAxes, **kw)

        vis_img_list = []

        disp_img = rearrange(rgbal[idx].detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        disp_img = rearrange(negative_a[idx].detach().clone().cpu().float(), '1 h w -> h w')
        vis_img_list.append(disp_img)

        disp_img = rearrange(al[idx].detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        disp_img = rearrange(rgba[idx,:3].detach().clone().cpu().float(), 'c h w -> h w c')
        disp_img = torch.where(
            disp_img!=0,
            disp_img/2+1/2,
            disp_img
        )
        vis_img_list.append(disp_img)

        disp_img = rearrange(rgba[idx].detach().clone().cpu().float(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        disp_img = rearrange(rgba[idx,3].detach().clone().cpu().float(), 'h w -> h w')
        vis_img_list.append(disp_img)

        fig = plt.figure(constrained_layout=True)
        ax_dict = fig.subplot_mosaic(
            [
                ["rgbal", "negative_a"],
                ["al", "rgb"],
                ["rgba", "a"],
            ],
            empty_sentinel="BLANK",
        )
        for i, k in enumerate(ax_dict.keys()):
            ax_dict[k].imshow(vis_img_list[i])
        identify_axes(ax_dict)
        plt.show()
        plt.close('all')
        pdb.set_trace()

class MPQAHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """
    def __init__(
        self, 
        cfg = None,
        use_cat_camshape=True,
        visualize: bool = True,
    ):
        super(MPQAHead, self).__init__()
        self.use_cat_camshape = use_cat_camshape
        self.visualize = visualize

        self.head = TokenHead(
            input_res = cfg.INPUT_RES, 
            input_dim = cfg.INPUT_DIM,
            filters = cfg.VIT_FILTER,
            patch_size = cfg.VIT_PATCH,
            num_cls = cfg.NUM_CLS,
            mask_ratio = cfg.MAE_MASK_RATIO,
        )

        if use_cat_camshape:
            pose_dim = cfg.VIT_FILTER[-1]
            shape_dim = cfg.VIT_FILTER[-1] * cfg.NUM_CLS
            cam_dim = cfg.VIT_FILTER[-1] * cfg.NUM_CLS
        else:
            pose_dim = cfg.VIT_FILTER[-1]
            shape_dim = cfg.VIT_FILTER[-1]
            cam_dim = cfg.VIT_FILTER[-1]

        self.mlp = MLPHead(
            pose_num = cfg.NUM_CLS,
            pose_dim = pose_dim,
            shape_dim = shape_dim,
            cam_dim = cam_dim,
            use_mean_pose = cfg.USE_MEAN_POSE,
            use_mean_camshape = cfg.USE_MEAN_CAMSHAPE,
        )
        self.smpl = SMPLHead(img_res=cfg.IMG_RES)

    def map_token_to_mlp(self, token):
        if self.use_cat_camshape:
            pose_feat = token
            shape_feat = rearrange(token, 'b n d -> b (n d)')
            cam_feat = rearrange(token, 'b n d -> b (n d)')
        else:
            pose_feat = token[:, :-2]
            shape_feat = token[:, -2]
            cam_feat = token[:, -1]
        return pose_feat, shape_feat, cam_feat

    def forward(self, img):
        """Forward function."""
        output = {}
        token, attn = self.head(img)
        pose, shape, cam = self.map_token_to_mlp(token)
        mlp_output = self.mlp(pose, shape, cam)

        smpl_output = self.smpl(
            rotmat=mlp_output['pred_rotmat'],
            betas=mlp_output['pred_shape'],
            cam=mlp_output['pred_cam'],
            normalize_joints2d=True,
        )
        output.update(mlp_output)
        output.update(smpl_output)
        if self.visualize:
            output.update({
                'online_visualize': attn,
            })
        return output

from src.losses import EnergyLossDistill
from .energy_head import EnergyHead, PseudoAttention, EnergyLoss

class EnergyLearnerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """
    def __init__(
        self, 
        probability = 'COS',
        temperature: float = 1., 
        attention: bool = False, 
        visualize: bool = True, 
        output_dim: int = 512,
        **kwarg
    ):
        super(EnergyLearnerHead, self).__init__()
        self.temperature = temperature
        self.visualize = visualize
        self.attention = attention

        self.head = EnergyHead(output_dim=output_dim, **kwarg)

    def forward(self, p: torch.Tensor, t: torch.Tensor, g: torch.Tensor, has_g: torch.Tensor):
        """Forward function."""
        p_detach = p.detach().clone().contiguous()
        t_detach = t.detach().clone().contiguous()
        g_detach = g.clone().contiguous()

        output_dict = self.head(p_detach, t_detach, g_detach, has_g)

        return output_dict