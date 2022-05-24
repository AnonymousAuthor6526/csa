import tkinter
import matplotlib
# matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
plt.rcParams["savefig.bbox"] = 'tight'

import pdb
import time
import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from matplotlib.colors import Normalize
from scipy import ndimage
from loguru import logger
from torch import linalg as LA
from einops import rearrange, reduce, repeat

class MSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(MSELoss, self).__init__()
        self.eps = eps
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, pd, gt, pd_mask, gt_mask):
        b, d, h, w = gt_mask.shape
        pd_mask = pd_mask.bool()
        gt_mask = gt_mask.bool()

        pd_n_c_hw = torch.einsum('bchw,bnhw->bnchw', pd, pd_mask)
        gt_n_c_hw = torch.einsum('bchw,bnhw->bnchw', gt, gt_mask)

        loss = self.criterion(pd_n_c_hw, gt_n_c_hw)

        # self.debug_rend_out(pd_n_c_hw, gt_n_c_hw)
        return loss

    def debug_rend_out(
        self,
        pd_n_c_hw, gt_n_c_hw, 
        idx=0
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

        disp_img = rearrange(pd_n_c_hw[idx, :, :3].detach().clone().cpu(), 'n c h w -> h (n w) c')
        disp_img = torch.where(
            disp_img!=0,
            disp_img/2+1/2,
            disp_img
        )
        vis_img_list.append(disp_img)
        disp_img = rearrange(gt_n_c_hw[idx, :, :3].detach().clone().cpu(), 'n c h w -> h (n w) c')
        disp_img = torch.where(
            disp_img!=0,
            disp_img/2+1/2,
            disp_img
        )
        vis_img_list.append(disp_img)

        disp_img = rearrange(pd_n_c_hw[idx, :, 3].detach().clone().cpu().float(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)
        disp_img = rearrange(gt_n_c_hw[idx, :, 3].detach().clone().cpu().float(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        loss_n_c_hw = F.mse_loss(pd_n_c_hw, gt_n_c_hw, reduction='none')
        disp_img = rearrange(loss_n_c_hw.detach().clone().cpu(), 'n c h w -> h (n w) c')
        vis_img_list.append(disp_img)

        fig = plt.figure(constrained_layout=True)
        ax_dict = fig.subplot_mosaic(
            [
                ["pd_rgb", "gt_rgb"],
                ["pd_o", "gt_o"],
                ["mse_rgb", "BLANK"],
            ],
            empty_sentinel="BLANK",
        )
        for i, k in enumerate(ax_dict.keys()):
            ax_dict[k].imshow(vis_img_list[i])
        identify_axes(ax_dict)
        plt.show()
        plt.close('all')
        pdb.set_trace()

class COSLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(COSLoss, self).__init__()
        self.eps = eps
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, pd, gt, pd_mask, gt_mask):
        b, d, h, w = gt_mask.shape
        pd_mask = pd_mask.bool()
        gt_mask = gt_mask.bool()

        pd_n_c_hw = torch.einsum('bchw,bnhw->bnchw', pd, pd_mask)
        gt_n_c_hw = torch.einsum('bchw,bnhw->bnchw', gt, gt_mask)

        cos_texture = torch.einsum('bnchw,bnchw->bnhw', pd_n_c_hw[:,:,:3], gt_n_c_hw[:,:,:3])
        cos_texture = 1 - torch.mean(cos_texture)

        mse_silhoue = self.criterion(pd_n_c_hw[:,:,3], gt_n_c_hw[:,:,3])
        return cos_texture + mse_silhoue

class COSLoss_dft(nn.Module):
    def __init__(self, eps=1e-6):
        super(COSLoss_dft, self).__init__()
        self.eps = eps
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, pd, gt, pd_mask, gt_mask):
        b, d, h, w = gt_mask.shape
        pd_mask = pd_mask.bool()
        gt_mask = gt_mask.bool()

        pd_n_c_hw = torch.einsum('bchw,bnhw->bnchw', pd, pd_mask)
        gt_n_c_hw = torch.einsum('bchw,bnhw->bnchw', gt, gt_mask)

        cos_texture = torch.einsum('bnchw,bnchw->bnhw', pd_n_c_hw[:,:,:3], gt_n_c_hw[:,:,:3])
        cos_texture = 1 - torch.mean(cos_texture)

        mse_silhoue = self.criterion(pd_n_c_hw[:,:,3], gt_n_c_hw[:,:,3])
        dft_n_hw = self.dft(gt_mask.float())
        dft_n = rearrange(dft_n_hw, 'b n h w -> b n (h w)')
        max_n = torch.max(dft_n, dim=2)[0]
        max_n_hw = rearrange(max_n, 'b n -> b n 1 1')
        dft_n_hw = dft_n_hw / (max_n_hw + self.eps)
        dft_silhoue = dft_silhoue * dft_n_hw
        
        return cos_texture + dft_silhoue.mean()

class IoUdftCOSLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(IoUdftCOSLoss, self).__init__()
        self.eps = eps
        self.dft = kornia.contrib.DistanceTransform(kernel_size=9, h=0.35)

    def forward(self, pd, gt, pd_mask, gt_mask):
        b, d, h, w = gt_mask.shape
        pd_mask = pd_mask.bool()
        gt_mask = gt_mask.bool()

        pd_n_c_hw = torch.einsum('bchw,bnhw->bnchw', pd, pd_mask)
        gt_n_c_hw = torch.einsum('bchw,bnhw->bnchw', gt, gt_mask)

        cos_n_hw = torch.einsum('bnchw,bnchw->bnhw', pd_n_c_hw[:,:,:3], gt_n_c_hw[:,:,:3])
        cos_texture = 1 - torch.mean(cos_n_hw)

        pd_n_hw, gt_n_hw = pd_n_c_hw[:,:,3], gt_n_c_hw[:,:,3].ceil()
        dft_n_hw = self.dft(gt_mask.float())
        dft_n_hw = torch.exp(-dft_n_hw)
        inter_n_hw = pd_n_hw * gt_n_hw
        union_n_hw = pd_n_hw * dft_n_hw + gt_n_hw - pd_n_hw * gt_n_hw
        iou_n = torch.sum(inter_n_hw, dim=[2,3]) / (torch.sum(union_n_hw, dim=[2,3]) + self.eps)
        iou_silhoue = iou_n.mean()

        # self.debug_rend_out(pd_n_c_hw, gt_n_c_hw, dft_n_hw, cos_n_hw, inter_n_hw, union_n_hw, idx=0)
        return cos_texture + iou_silhoue

    def debug_rend_out(
        self,
        pd_n_c_hw, gt_n_c_hw, 
        dft_n_hw, cos_n_hw, 
        inter_n_hw, union_n_hw,
        idx=0
    ):
        # Helper function used for visualization in the following examples
        def identify_axes(ax_dict, fontsize=8):
            kw = dict(fontsize=fontsize, color="black")
            for k, ax in ax_dict.items():
                ax.text(0.05, 0.9, k, transform=ax.transAxes, **kw)

        vis_img_list = []

        disp_img = rearrange(pd_n_c_hw[idx, :, :3].detach().clone().cpu(), 'n c h w -> h (n w) c')
        disp_img = torch.where(
            disp_img!=0,
            disp_img/2+1/2,
            disp_img
        )
        vis_img_list.append(disp_img)
        disp_img = rearrange(gt_n_c_hw[idx, :, :3].detach().clone().cpu(), 'n c h w -> h (n w) c')
        disp_img = torch.where(
            disp_img!=0,
            disp_img/2+1/2,
            disp_img
        )
        vis_img_list.append(disp_img)

        disp_img = rearrange(pd_n_c_hw[idx, :, 3].detach().clone().cpu().float(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)
        disp_img = rearrange(gt_n_c_hw[idx, :, 3].detach().clone().cpu().float(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        texture = LA.norm(pd_n_c_hw[idx,:,:3] - gt_n_c_hw[idx,:,:3], ord=2, dim=1, keepdim=False)
        disp_img = rearrange(texture.detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)
        silhoue = F.mse_loss(pd_n_c_hw[idx,:,3], gt_n_c_hw[idx,:,3], reduction='none')
        disp_img = rearrange(silhoue.detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        disp_img = rearrange(dft_n_hw[idx].detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)
        disp_img = rearrange(cos_n_hw[idx].detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        disp_img = rearrange(inter_n_hw[idx].detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)
        disp_img = rearrange(union_n_hw[idx].detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        fig = plt.figure(constrained_layout=True)
        ax_dict = fig.subplot_mosaic(
            [
                ["pd_rgb", "gt_rgb"],
                ["pd_o", "gt_o"],
                ["texture", "silhoue"],
                ["dft_n_hw", "cos_n_hw"],
                ["inter_n_hw", "union_n_hw"],
            ],
            empty_sentinel="BLANK",
        )
        for i, k in enumerate(ax_dict.keys()):
            ax_dict[k].imshow(vis_img_list[i])
        identify_axes(ax_dict)
        plt.show()
        plt.close('all')
        pdb.set_trace()

class L2Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super(L2Loss, self).__init__()
        self.eps = eps
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, pd, gt, pd_mask, gt_mask, weights=None):
        b, d, h, w = gt_mask.shape
        pd_mask = pd_mask.bool()
        gt_mask = gt_mask.bool()

        pd_n_c_hw = torch.einsum('bchw,bnhw->bnchw', pd, pd_mask)
        gt_n_c_hw = torch.einsum('bchw,bnhw->bnchw', gt, gt_mask)

        texture_n_hw = LA.norm(pd_n_c_hw[:,:,:3] - gt_n_c_hw[:,:,:3], ord=2, dim=2, keepdim=False)
        silhoue_n_hw = self.criterion(pd_n_c_hw[:,:,3], gt_n_c_hw[:,:,3].ceil())

        # self.debug_rend_out(pd_n_c_hw, gt_n_c_hw)
        if weights is not None:
            loss_n_hw = texture_n_hw + silhoue_n_hw
            loss_n = torch.mean(loss_n_hw, dim=[1,2,3])
            loss_n = torch.mul(loss_n, weights)
            loss = torch.mean(loss_n)
        else:
            loss_n_hw = texture_n_hw + silhoue_n_hw
            loss = torch.mean(loss_n_hw)
        return loss

    def debug_rend_out(
        self,
        pd_n_c_hw, gt_n_c_hw, 
        idx=0
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

        disp_img = rearrange(pd_n_c_hw[idx, :, :3].detach().clone().cpu(), 'n c h w -> h (n w) c')
        disp_img = torch.where(
            disp_img!=0,
            disp_img/2+1/2,
            disp_img
        )
        vis_img_list.append(disp_img)
        disp_img = rearrange(gt_n_c_hw[idx, :, :3].detach().clone().cpu(), 'n c h w -> h (n w) c')
        disp_img = torch.where(
            disp_img!=0,
            disp_img/2+1/2,
            disp_img
        )
        vis_img_list.append(disp_img)

        disp_img = rearrange(pd_n_c_hw[idx, :, 3].detach().clone().cpu().float(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)
        disp_img = rearrange(gt_n_c_hw[idx, :, 3].detach().clone().cpu().float(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        texture = LA.norm(pd_n_c_hw[idx,:,:3] - gt_n_c_hw[idx,:,:3], ord=2, dim=1, keepdim=False)
        disp_img = rearrange(texture.detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        silhoue = F.mse_loss(pd_n_c_hw[idx,:,3], gt_n_c_hw[idx,:,3], reduction='none')
        disp_img = rearrange(silhoue.detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        fig = plt.figure(constrained_layout=True)
        ax_dict = fig.subplot_mosaic(
            [
                ["pd_rgb", "gt_rgb"],
                ["pd_o", "gt_o"],
                ["texture", "silhoue"],
            ],
            empty_sentinel="BLANK",
        )
        for i, k in enumerate(ax_dict.keys()):
            ax_dict[k].imshow(vis_img_list[i])
        identify_axes(ax_dict)
        plt.show()
        plt.close('all')
        pdb.set_trace()

class L2Loss_dft(nn.Module):
    def __init__(self, eps=1e-6):
        super(L2Loss_dft, self).__init__()
        self.eps = eps
        self.criterion = nn.MSELoss(reduction='mean')
        self.dft = kornia.contrib.DistanceTransform(kernel_size=9, h=0.35)

    def forward(self, pd, gt, pd_mask, gt_mask):
        b, d, h, w = gt_mask.shape
        pd_mask = pd_mask.bool()
        gt_mask = gt_mask.bool()

        pd_n_c_hw = torch.einsum('bchw,bnhw->bnchw', pd, pd_mask)
        gt_n_c_hw = torch.einsum('bchw,bnhw->bnchw', gt, gt_mask)

        l2_n_hw = LA.norm(pd_n_c_hw[:,:,:3] - gt_n_c_hw[:,:,:3], ord=2, dim=2, keepdim=False)
        dft_n_hw = self.dft(gt_mask.float())
        dft_n = rearrange(dft_n_hw, 'b n h w -> b n (h w)')
        max_n = torch.max(dft_n, dim=2)[0]
        max_n_hw = rearrange(max_n, 'b n -> b n 1 1')
        dft_n_hw = torch.where(
            dft_n_hw > 0.,
            1 + dft_n_hw / (max_n_hw + self.eps),
            0.5 + dft_n_hw
        )
        l2_texture = l2_n_hw * dft_n_hw

        mse_silhoue = self.criterion(pd_n_c_hw[:,:,3], gt_n_c_hw[:,:,3])

        # self.debug_rend_out(pd_n_c_hw, gt_n_c_hw, dft_n_hw, l2_texture)
        return l2_texture.mean() + mse_silhoue

    def debug_rend_out(
        self,
        pd_n_c_hw, gt_n_c_hw, 
        dft_n_hw, l2_texture, 
        idx=0
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

        disp_img = rearrange(pd_n_c_hw[idx, :, :3].detach().clone().cpu(), 'n c h w -> h (n w) c')
        disp_img = torch.where(
            disp_img!=0,
            disp_img/2+1/2,
            disp_img
        )
        vis_img_list.append(disp_img)
        disp_img = rearrange(gt_n_c_hw[idx, :, :3].detach().clone().cpu(), 'n c h w -> h (n w) c')
        disp_img = torch.where(
            disp_img!=0,
            disp_img/2+1/2,
            disp_img
        )
        vis_img_list.append(disp_img)

        disp_img = rearrange(pd_n_c_hw[idx, :, 3].detach().clone().cpu().float(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)
        disp_img = rearrange(gt_n_c_hw[idx, :, 3].detach().clone().cpu().float(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        texture = LA.norm(pd_n_c_hw[idx,:,:3] - gt_n_c_hw[idx,:,:3], ord=2, dim=1, keepdim=False)
        disp_img = rearrange(texture.detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)
        silhoue = F.mse_loss(pd_n_c_hw[idx,:,3], gt_n_c_hw[idx,:,3], reduction='none')
        disp_img = rearrange(silhoue.detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        disp_img = rearrange(dft_n_hw[idx].detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)
        disp_img = rearrange(l2_texture[idx].detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        fig = plt.figure(constrained_layout=True)
        ax_dict = fig.subplot_mosaic(
            [
                ["pd_rgb", "gt_rgb"],
                ["pd_o", "gt_o"],
                ["texture", "silhoue"],
                ["dft_n_hw", "l2_texture"],
            ],
            empty_sentinel="BLANK",
        )
        for i, k in enumerate(ax_dict.keys()):
            ax_dict[k].imshow(vis_img_list[i])
        identify_axes(ax_dict)
        plt.show()
        plt.close('all')
        pdb.set_trace()

class DSRIoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DSRIoULoss, self).__init__()
        self.eps = eps

    def forward(self, pd, gt, pd_mask, gt_mask, weights=None):
        b, d, h, w = gt_mask.shape
        pd_mask = pd_mask.bool()
        gt_mask = gt_mask.bool()

        if weights is None:
            weights = torch.ones(b).type_as(pd)

        pd_n = rearrange(pd, 'b h w -> b 1 h w')
        gt_n = rearrange(gt, 'b h w -> b 1 h w')
        pd_n = torch.mul(pd_n, pd_mask)
        gt_n = torch.mul(gt_n, gt_mask)

        intersect = torch.sum(pd_n * gt_n, dim=[2, 3])
        union = torch.sum(pd_n + gt_n - pd_n * gt_n, dim=[2, 3])
        loss = 1 - intersect / (union + self.eps)
        loss = loss.mean(dim=1) * weights
        return loss.mean()

class DistanceLoss(nn.Module):
    def __init__(self, p=2, dim=1, keepdim=False, reduction=None):
        super(DistanceLoss, self).__init__()
        self.ord = p
        self.dim = dim
        self.keepdim = keepdim
        self.reduction = reduction

    def forward(self, pd, gt):
        distance = pd - gt
        distance = LA.norm(distance, ord=self.ord, dim=self.dim, keepdim=self.keepdim)
        if self.reduction == 'mean':
            return distance.mean()
        else:
            return distance

class IOULoss(nn.Module):
    def __init__(self, dim=1, eps=1e-6):
        super(IOULoss, self).__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, pd, gt):
        dims = tuple(range(pd.ndimension())[self.dim:])
        intersect = torch.sum(pd * gt, dim=dims)
        union = torch.sum(pd + gt - pd * gt, dim=dims) + self.eps
        iou = intersect / union
        return iou

class IoULoss_pixel(nn.Module):
    def __init__(self, dim=1, eps=1e-6):
        super(IoULoss_pixel, self).__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, pd, gt, pd_mask, gt_mask):
        b, n, h, w = pd_mask.shape
        pd_mask = pd_mask.bool()
        gt_mask = gt_mask.bool()
        and_mask = torch.logical_and(pd_mask, gt_mask)
        or_mask = torch.logical_or(pd_mask, gt_mask)
        pd_xor_mask = torch.logical_xor(or_mask, gt_mask)

        loss_list = []
        for i in range(b):
            for j in range(n):
                inter = pd[i][and_mask[i,j]].sum()
                union = gt[i][gt_mask[i,j]].sum() + pd[i][pd_xor_mask[i,j]].sum()
                loss = inter / (union + self.eps)
                loss_list.append(loss)

        return 1 - sum(loss_list) / len(loss_list)

class IoULoss(nn.Module):
    def __init__(self, dim=2, eps=1e-6, reduction='mean'):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, pd, gt, pd_mask, gt_mask):
        b, d, h, w = gt_mask.shape
        pd_mask = pd_mask.bool()
        gt_mask = gt_mask.bool()

        # apply mask
        pd_n_c_hw = torch.einsum('bchw,bnhw->bnchw', pd, pd_mask)
        gt_n_c_hw = torch.einsum('bchw,bnhw->bnchw', gt, gt_mask)
        
        pd_n_hw, gt_n_hw = pd_n_c_hw[:,:,3], gt_n_c_hw[:,:,3].ceil()

        inter_n_hw = pd_n_hw * gt_n_hw
        union_n_hw = pd_n_hw + gt_n_hw - pd_n_hw * gt_n_hw
        iou_n = torch.sum(inter_n_hw, dim=[2,3]) / (torch.sum(union_n_hw, dim=[2,3]) + self.eps)

        return 1 - iou_n.mean()

# grads = {}

# def save_grad(name):
#     def hook(grad):
#         grads[name] = grad
#     return hook

class NIoULoss(nn.Module):
    def __init__(self, dim=2, eps=1e-6, reduction='mean'):
        super(NIoULoss, self).__init__()
        self.eps = eps
        self.dft = kornia.contrib.DistanceTransform(kernel_size=9, h=0.35)

    def forward(self, pd, gt, pd_mask, gt_mask, img=None):
        b, d, h, w = gt_mask.shape
        pd_mask = pd_mask.bool()
        gt_mask = gt_mask.bool()

        # apply mask
        pd_n_c_hw = torch.einsum('bchw,bnhw->bnchw', pd, pd_mask)
        gt_n_c_hw = torch.einsum('bchw,bnhw->bnchw', gt, gt_mask)
        # pd_n_c_hw.register_hook(save_grad('pd_n_c_hw'))
        # pdb.set_trace()

        pd_n_hw, gt_n_hw = pd_n_c_hw[:,:,3], gt_n_c_hw[:,:,3].ceil()
        inter_n_hw = pd_n_hw * gt_n_hw
        union_n_hw = pd_n_hw + gt_n_hw - pd_n_hw * gt_n_hw
        iou_n = torch.sum(inter_n_hw, dim=[2,3]) / (torch.sum(union_n_hw, dim=[2,3]) + self.eps)

        pd_n_rgb_hw = pd_n_c_hw[:, :, :3]
        gt_n_rgb_hw = torch.einsum('bnchw,bnhw->bnchw', gt_n_c_hw[:, :, :3], pd_mask)
        L2_n_hw = F.mse_loss(pd_n_rgb_hw, gt_n_rgb_hw, reduction='none').sum(dim=2)

        # dist_n_hw = self._dist_map(gt_mask, pd_n_rgb.dtype)
        dist_n_hw = self.dft(gt_mask.float())

        dist_sum_n = torch.sum(pd_mask, dim=[2, 3])
        dist_max_n = rearrange(dist_n_hw, 'b n h w -> b n (h w)')
        dist_max_n = torch.max(dist_max_n, dim=2)[0]
        dist_all_n = torch.mul(dist_max_n, dist_sum_n)

        dist_n_hw = torch.where(
            dist_n_hw == 0, 
            torch.Tensor([0.5]).type_as(dist_n_hw), 
            dist_n_hw / dist_max_n[:,:,None,None]+1
        )
        punish_n_hw = torch.einsum('bnhw,bnhw->bnhw', L2_n_hw, dist_n_hw)
        punish_n = torch.sum(punish_n_hw, dim=[2, 3])

        punish_n = punish_n / (dist_all_n + self.eps)

        # punish_n_hw.sum([1,2,3]).max(0)
        # self.debug_rend_out(img, pd[:,:3], gt[:,:3], punish_n_hw, 0, alpha=0.8, cmap='afmhot')
        # pdb.set_trace()
        
        loss = 1 - (iou_n - punish_n)

        return loss.mean()

    def _dist_map(self, mask, dtype):
        b, n, _, _ = mask.shape

        gt_mask_not = torch.logical_not(mask).cpu()
        dist_maps = torch.zeros_like(mask, dtype=dtype)
        for i in range(b):
            for j in range(n):
                dist_map = ndimage.distance_transform_edt(gt_mask_not[i, j])
                dist_maps[i, j] = torch.from_numpy(dist_map + 1)
        return dist_maps

    def debug_rend_out(self,
        img, pd_rgb_hw, gt_rgb_hw, punish_n_hw, 
        idx, alpha=0.6, cmap='Reds'
    ):  
        from src.utils.vis_utils import visualize_heatmaps, reverse_norm

        img = img.detach().cpu()
        pd_rgb_hw = pd_rgb_hw.detach().cpu()
        gt_rgb_hw = gt_rgb_hw.detach().cpu()
        punish_n_hw = punish_n_hw.detach().cpu()

        fig, axs = plt.subplots(ncols=2, nrows=2, constrained_layout=True)
        img = reverse_norm(img[idx])
        img = rearrange(img, '1 c h w -> h w c')
        axs[1,1].imshow(img)

        disp_img = rearrange(pd_rgb_hw[idx].detach().clone().cpu(), 'c h w -> h w c')
        disp_img = torch.where(
            disp_img!=0,
            disp_img/2+1/2,
            disp_img
        )
        disp_img = img*(1-alpha) + alpha*disp_img
        axs[0,0].imshow(disp_img)

        disp_img = rearrange(gt_rgb_hw[idx].detach().clone().cpu(), 'c h w -> h w c')
        disp_img = torch.where(
            disp_img!=0,
            disp_img/2+1/2,
            disp_img
        )
        disp_img = img*(1-alpha) + alpha*disp_img
        axs[1,0].imshow(disp_img)

        # alphas = Normalize(0, .3, clip=True)(np.abs(weights))
        # alphas = np.clip(alphas, .4, 1)  # alpha value clipped at the bottom at .4
        # disp_img = visualize_heatmaps(img, punish_n_hw[idx].sum(0), alpha=0.8, cmap=cmap)
        # disp_img = punish_n_hw[idx].sum(0)
        axs[0,1].imshow(img)
        im = axs[0,1].imshow(punish_n_hw[idx].sum(0), alpha=0.9, cmap=cmap)
        fig.colorbar(im, ax=axs[0,1], shrink=1.)

        axs[0,0].set_axis_off()
        axs[0,1].set_axis_off()
        axs[1,0].set_axis_off()
        

        plt.show()
        plt.close('all')

    # def debug_rend_out(self,
    #     pd_n_hw, gt_n_hw, inter_n_hw, union_n_hw,
    #     pd_n_rgb_hw, gt_n_rgb_hw, dist_n_hw, L2_n_hw, punish_n_hw, 
    #     idx
    # ):
    #     # Helper function used for visualization in the following examples
    #     def identify_axes(ax_dict, fontsize=8):
    #         kw = dict(fontsize=fontsize, color="black")
    #         for k, ax in ax_dict.items():
    #             ax.text(0.05, 0.9, k, transform=ax.transAxes, **kw)

    #     vis_img_list = []

    #     disp_img = rearrange(pd_n_hw[idx].detach().clone().cpu().float(), 'n h w -> h (n w)')
    #     vis_img_list.append(disp_img)

    #     disp_img = rearrange(gt_n_hw[idx].detach().clone().cpu().float(), 'n h w -> h (n w)')
    #     vis_img_list.append(disp_img)

    #     disp_img = rearrange(inter_n_hw[idx].detach().clone().cpu().float(), 'n h w -> h (n w)')
    #     vis_img_list.append(disp_img)

    #     disp_img = rearrange(union_n_hw[idx].detach().clone().cpu().float(), 'n h w -> h (n w)')
    #     vis_img_list.append(disp_img)

    #     disp_img = rearrange(pd_n_rgb_hw[idx].detach().clone().cpu(), 'n c h w -> h (n w) c')
    #     disp_img = torch.where(
    #         disp_img!=0,
    #         disp_img/2+1/2,
    #         disp_img
    #     )
    #     vis_img_list.append(disp_img)

    #     disp_img = rearrange(gt_n_rgb_hw[idx].detach().clone().cpu(), 'n c h w -> h (n w) c')
    #     disp_img = torch.where(
    #         disp_img!=0,
    #         disp_img/2+1/2,
    #         disp_img
    #     )
    #     vis_img_list.append(disp_img)
        
    #     disp_img = rearrange(dist_n_hw[idx].detach().clone().cpu(), 'n h w -> h (n w)')
    #     vis_img_list.append(disp_img)

    #     disp_img = rearrange(L2_n_hw[idx].detach().clone().cpu(), 'n h w -> h (n w)')
    #     vis_img_list.append(disp_img)

    #     disp_img = rearrange(punish_n_hw[idx].detach().clone().cpu(), 'n h w -> h (n w)')
    #     vis_img_list.append(disp_img)

    #     fig = plt.figure(constrained_layout=True)
    #     ax_dict = fig.subplot_mosaic(
    #         [
    #             ["pd_n_hw", "gt_n_hw"],
    #             ["inter_n_hw", "union_n_hw"],
    #             ["pd_n_rgb_hw", "gt_n_rgb_hw"],
    #             ["dist_n_hw", "BLANK"],
    #             ["L2_n_hw", "punish_n_hw"],
    #         ],
    #         empty_sentinel="BLANK",
    #     )
    #     for i, k in enumerate(ax_dict.keys()):
    #         ax_dict[k].imshow(vis_img_list[i])
    #     identify_axes(ax_dict)
    #     plt.show()
    #     plt.close('all')
    #     pdb.set_trace()

class NIoULoss_mask(nn.Module):
    def __init__(self, eps=1e-6, use_dist=True):
        super(NIoULoss_mask, self).__init__()
        self.eps = eps
        self.use_dist = use_dist
        self.dft = kornia.contrib.DistanceTransform(kernel_size=9, h=0.35)

    def forward(self, pd_rgbo, gt_rgbo, pd_mask, gt_mask):
        b, d, h, w = gt_mask.shape
        pd_mask = pd_mask.contiguous().bool()
        gt_mask = gt_mask.contiguous().bool()

        # apply mask
        and_mask = torch.logical_and(pd_mask, gt_mask)
        or_mask = torch.logical_or(pd_mask, gt_mask)
        xor_mask = torch.logical_xor(or_mask, gt_mask)

        # compute iou
        pd_o = pd_rgbo[:, 3:4]
        gt_o = gt_rgbo[:, 3:4]
        inter_n_hw = pd_o * gt_o * and_mask
        union_n_hw = pd_o * xor_mask + gt_mask
        inter_n = torch.sum(inter_n_hw, dim=[2,3])
        union_n = torch.sum(union_n_hw, dim=[2,3])
        iou_n = inter_n / (union_n + self.eps)

        # compute punish
        pd_rgb = pd_rgbo[:, :3]
        gt_rgb = gt_rgbo[:, :3]
        and_hw = LA.norm(pd_rgb - gt_rgb, ord=2, dim=1, keepdim=True)
        and_n_hw = and_hw * and_mask / 2
        xor_hw = LA.norm(pd_rgb, ord=2, dim=1, keepdim=True)
        xor_n_hw = xor_hw * xor_mask

        if self.use_dist:
            dist_n_hw = self.dft(gt_mask.float())
            
            xor_n_hw = xor_n_hw * dist_n_hw
            numera_n_hw = and_n_hw + xor_n_hw
            numera_n = torch.sum(numera_n_hw, dim=[2, 3])

            num_n = torch.sum(pd_mask, dim=[2, 3])
            dist_n = rearrange(dist_n_hw, 'b n h w -> b n (h w)')
            max_n = torch.max(dist_n, dim=2)[0]
            denomina_n = torch.mul(num_n, max_n)

            punish_n = numera_n / (denomina_n + self.eps)

            # self.debug_rend_out(0, 
            #     pd_rgbo, gt_rgbo, pd_mask, gt_mask, 
            #     inter_n_hw, union_n_hw, and_n_hw, xor_n_hw,
            #     numera_n_hw, 
            #     dist_n_hw = dist_n_hw
            # )

        else:
            numera_n_hw = and_n_hw + xor_n_hw
            numera_n = torch.sum(numera_n_hw, dim=[2, 3])
            denomina_n = torch.sum(pd_mask, dim=[2, 3])
            punish_n = numera_n / (denomina_n + self.eps)

            # self.debug_rend_out(2, 
            #     pd_rgbo, gt_rgbo, pd_mask, gt_mask, 
            #     inter_n_hw, union_n_hw, and_n_hw, xor_n_hw,
            #     numera_n_hw,
            # )
        
        # compute loss
        loss_n = iou_n - punish_n

        loss = 1 - torch.mean(loss_n, dim=1)
        return loss.mean()

    def debug_rend_out(
        self, idx,
        pd_rgbo, gt_rgbo, 
        pd_mask, gt_mask, 
        inter_n_hw, union_n_hw, 
        and_n_hw, xor_n_hw,
        numera_n_hw, denomina_n_hw=None, 
        dist_n_hw=None,
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

        disp_img = rearrange(pd_rgbo[idx].detach().clone().cpu(), 'c h w -> h w c')
        disp_img = torch.where(
            disp_img!=0,
            disp_img/2+1/2,
            disp_img
        )
        vis_img_list.append(disp_img)
        disp_img = rearrange(gt_rgbo[idx].detach().clone().cpu(), 'c h w -> h w c')
        disp_img = torch.where(
            disp_img!=0,
            disp_img/2+1/2,
            disp_img
        )
        vis_img_list.append(disp_img)
        
        disp_img = rearrange(pd_mask[idx].detach().clone().cpu().float(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)
        disp_img = rearrange(gt_mask[idx].detach().clone().cpu().float(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        disp_img = rearrange(inter_n_hw[idx].detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)
        disp_img = rearrange(union_n_hw[idx].detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        disp_img = rearrange(and_n_hw[idx].detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)
        disp_img = rearrange(xor_n_hw[idx].detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        disp_img = rearrange(numera_n_hw[idx].detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)
        if denomina_n_hw is not None:
            disp_img = rearrange(denomina_n_hw[idx].detach().clone().cpu(), 'n h w -> h (n w)')
            vis_img_list.append(disp_img)
        else:
            disp_img = torch.zeros([224, 224])
            vis_img_list.append(disp_img)

        if dist_n_hw is not None:
            disp_img = rearrange(dist_n_hw[idx].detach().clone().cpu(), 'n h w -> h (n w)')
            vis_img_list.append(disp_img)
        else:
            disp_img = torch.zeros([224, 224])
            vis_img_list.append(disp_img)

        fig = plt.figure(constrained_layout=True)
        ax_dict = fig.subplot_mosaic(
            [
                ["pd_rgbo", "gt_rgbo"],
                ["pd_mask", "gt_mask"],
                ["inter_n_hw", "union_n_hw"],
                ["and_n_hw", "xor_n_hw"],
                ["numera_n_hw", "denomina_n_hw"],
                ["dist_n_hw", "BLANK"],
            ],
            empty_sentinel="BLANK",
        )
        for i, k in enumerate(ax_dict.keys()):
            ax_dict[k].imshow(vis_img_list[i])

        identify_axes(ax_dict)
        plt.show()
        plt.close('all')
        pdb.set_trace()

class NIoULoss_pixel(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean', puish_dist=False):
        super(NIoULoss_pixel, self).__init__()
        self.eps = eps
        self.puish_dist = puish_dist
        self.reduction = reduction

    def forward(self, pd_rgbo, gt_rgbo, pd_mask, gt_mask, weights=None):
        b, n, h, w = gt_mask.shape

        if weights is None:
            weights = torch.ones(b).type_as(pd_rgbo)

        pd_mask = pd_mask.bool()
        gt_mask = gt_mask.bool()
        and_mask = torch.logical_and(pd_mask, gt_mask)
        or_mask = torch.logical_or(pd_mask, gt_mask)
        pd_xor_mask = torch.logical_xor(or_mask, gt_mask)
        gt_not_mask = torch.logical_not(gt_mask).cpu()

        pd_rgb = pd_rgbo[:, :3]
        gt_rgb = gt_rgbo[:, :3]
        pd_o = pd_rgbo[:, 3].round()
        gt_o = gt_rgbo[:, 3].round()

        loss_list = []
        for i in range(b):
            weight = weights[i]
            for j in range(n):
                pd_mask_rgb = pd_rgb[i][:, pd_mask[i,j]]
                gt_mask_rgb = gt_rgb[i][:, pd_mask[i,j]]
                similarity = LA.norm(pd_mask_rgb - gt_mask_rgb, ord=2, dim=0, keepdim=False)

                if self.puish_dist:
                    dist_map = ndimage.distance_transform_edt(gt_not_mask[i, j])
                    dist_map = torch.from_numpy(dist_map + 1).type_as(pd_rgbo)
                    dist_mask_map = dist_map[pd_mask[i,j]]
                    numerator = torch.mul(similarity, dist_mask_map)
                    denominator = dist_map[or_mask[i,j]]
                    punish = numerator.sum() / (denominator.sum() + self.eps)
                else:
                    punish = similarity.sum() / (similarity.nelement() + self.eps)

                inter = pd_o[i][and_mask[i,j]].sum()
                union = gt_o[i][gt_mask[i,j]].sum() + pd_o[i][pd_xor_mask[i,j]].sum()
                iou = inter / (union + self.eps)

                loss = weight * (1 - iou + punish)
                loss_list.append(loss)

        # self.debug_rend_out(
        #     pd_rgb, gt_rgb, 
        #     pd_o, gt_o, 
        #     pd_mask, gt_mask,
        #     and_mask, or_mask,
        #     pd_xor_mask, gt_not_mask,
        #     0
        # )
        return sum(loss_list) / len(loss_list)

    def debug_rend_out(
        self,
        pd_rgb, gt_rgb, 
        pd_o, gt_o, 
        pd_mask, gt_mask,
        and_mask, or_mask,
        pd_xor_mask, gt_not_mask,
        idx
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

        disp_img = rearrange(pd_rgb[idx].detach().clone().cpu(), 'c h w -> h w c')
        disp_img = torch.where(
            disp_img!=0,
            disp_img/2+1/2,
            disp_img
        )
        vis_img_list.append(disp_img)
        disp_img = rearrange(gt_rgb[idx].detach().clone().cpu(), 'c h w -> h w c')
        disp_img = torch.where(
            disp_img!=0,
            disp_img/2+1/2,
            disp_img
        )
        vis_img_list.append(disp_img)
        
        disp_img = rearrange(pd_o[idx].detach().clone().cpu().float(), 'h w -> h w')
        vis_img_list.append(disp_img)
        disp_img = rearrange(gt_o[idx].detach().clone().cpu().float(), 'h w -> h w')
        vis_img_list.append(disp_img)

        disp_img = rearrange(pd_mask[idx].detach().clone().cpu().float(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)
        disp_img = rearrange(gt_mask[idx].detach().clone().cpu().float(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        disp_img = rearrange(and_mask[idx].detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)
        disp_img = rearrange(or_mask[idx].detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        disp_img = rearrange(pd_xor_mask[idx].detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)
        disp_img = rearrange(gt_not_mask[idx].detach().clone().cpu(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        fig = plt.figure(constrained_layout=True)
        ax_dict = fig.subplot_mosaic(
            [
                ["pd_rgb", "gt_rgb"],
                ["pd_o", "gt_o"],
                ["pd_mask", "gt_mask"],
                ["and_mask", "or_mask"],
                ["pd_xor_mask", "gt_not_mask"],
            ],
            empty_sentinel="BLANK",
        )
        for i, k in enumerate(ax_dict.keys()):
            ax_dict[k].imshow(vis_img_list[i])
        identify_axes(ax_dict)
        plt.show()
        plt.close('all')
        pdb.set_trace()