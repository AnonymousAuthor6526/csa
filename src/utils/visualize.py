import os
import pdb
import sys
import math
import torch
import random
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')

from matplotlib.colors import Normalize
from torch import linalg as LA
from einops import rearrange, reduce, repeat

############################################################
#  Visualization
############################################################


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def Plot2Image(fig: matplotlib.figure):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    image = data.reshape((int(h), int(w), -1))
    return image

def apply_mask(images: torch.Tensor, masks: torch.Tensor, colors: list, alpha: float=0.7) -> torch.Tensor:
    """Apply the given mask to the image.
    """
    b, c, h, w = images.shape
    _, n, _, _ = masks.shape
    images_mask = images.clone()
    masks = masks.clone()

    for i in range(n):
        mask = masks[:, i]
        color = colors[i]
        for j in range(c):
            images_mask[:, j, :, :] = torch.where(
                mask,
                images_mask[:, j, :, :] * (1 - alpha) + alpha * (color[j] / 255.),
                images_mask[:, j, :, :]
            )
    return images_mask

def getBatchPlotImg(image: np.array) -> np.array:
    if image.ndim == 3:
        b, h, w = image.shape
    else:
        image = np.transpose(image, [0, 2, 3, 1])
        b, h, w, c = image.shape
    
    img_list = []
    for i in range(b):
        fig, ax = plt.subplots()
        ax.imshow(image[i])
        img = Plot2Image(fig)
        img = torch.from_numpy(img).permute(2, 0, 1)[None, :, :, :]
        img_list.append(img)
        plt.close('all')
    img_list = torch.cat([img_list], dim=0)
    return img_list

def show_batch_img(imgs, img_type='normal'):
    imgs = imgs.cpu().detach().float()

    if img_type == 'normal':
        imgs = rearrange(imgs, 'b c h w -> h (b w) c')
        imgs = normal2img(imgs)
    elif img_type == 'alpha':
        imgs = rearrange(imgs, 'b h w -> h (b w)')

    fix, axs = plt.subplots(constrained_layout=True)
    axs.imshow(imgs)
    plt.show()
    plt.close('all')

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

def draw_energy(energy, cbar_kw={}, cmap='RdYlBu'):
    h, w = energy.shape

    #Ploting attention map
    fig = plt.figure(constrained_layout=True)
    ax_dict = fig.subplot_mosaic(
        [
            ["energy"]
        ],
        empty_sentinel="BLANK",
    )

    im = ax_dict["heatmap"].imshow(energy)
    cbar = ax_dict["heatmap"].figure.colorbar(im, ax=ax_dict["heatmap"], **cbar_kw)
    cbar.ax.set_ylabel('attention value', rotation=-90, va="bottom")
    # Show all ticks and label them with the respective list entries
    
    # ax_dict["heatmap"].set_xticks(np.arange(att_h))
    # ax_dict["heatmap"].set_yticks(np.arange(att_w))
    # ax_dict["heatmap"].set_xticklabels(np.arange(att_h))
    # ax_dict["heatmap"].set_xticklabels(np.arange(att_w))
    # ax_dict["heatmap"].tick_params(which="minor", bottom=False, left=False)
    # Rotate the tick labels and set their alignment.
    # plt.setp(ax_dict["heatmap"].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax_dict["heatmap"].spines[:].set_visible(False)
    ax_dict["heatmap"].grid(which="minor", color="w", linestyle='-', linewidth=3)
    # Loop over data dimensions and create text annotations.
    for i in range(h):
        for j in range(w):
            text = ax_dict["heatmap"].text(j, i, energy[i, j], ha="center", va="center", color="w")
    ax_dict["heatmap"].set_title("heatmap")

    fig.tight_layout()
    plt.show()
    plt.close('all')


# def visualize_attention_tb(img, att, idx=0):
#     _, n, d = att.shape
#     h, w = pair(int(math.sqrt(d)))

#     img = rearrange(img[idx].cpu(), 'c h w -> h w c')
#     att = rearrange(att[idx].cpu(), 'n (h w) -> n h w', h=h, w=w)
#     # pdb.set_trace()
#     att_img = TF.resize(att, size=img.shape[:2], interpolation=TF.InterpolationMode.NEAREST)

#     draw_attention(img.numpy(), att[0].numpy(), att_img[0].numpy())

# def draw_attention(img, att_map, att_img, cbar_kw={}, cmap='RdYlBu'):
#     img_h, img_w, img_c = img.shape
#     att_h, att_w = att_map.shape

#     #Ploting attention map
#     fig = plt.figure(constrained_layout=True)
#     ax_dict = fig.subplot_mosaic(
#         [
#             ["image", "attention"],
#             ["heatmap", "contour"],
#         ],
#         empty_sentinel="BLANK",
#     )

#     ax_dict["image"].imshow(img, cmap=cmap)  #plot the first image
#     ax_dict["image"].set_title('image')


#     ax_dict["attention"].imshow(img)  #plot the first image
#     ax_dict["attention"].imshow(att_map, interpolation='nearest', alpha=0.5)  #plot the first image
#     ax_dict["attention"].set_title('attention')


#     im = ax_dict["heatmap"].imshow(att_map)
#     cbar = ax_dict["heatmap"].figure.colorbar(im, ax=ax_dict["heatmap"], **cbar_kw)
#     cbar.ax.set_ylabel('attention value', rotation=-90, va="bottom")
#     # Show all ticks and label them with the respective list entries
    
#     ax_dict["heatmap"].set_xticks(np.arange(att_h))
#     ax_dict["heatmap"].set_yticks(np.arange(att_w))
#     ax_dict["heatmap"].set_xticklabels(np.arange(att_h))
#     ax_dict["heatmap"].set_xticklabels(np.arange(att_w))
#     ax_dict["heatmap"].tick_params(which="minor", bottom=False, left=False)
#     # Rotate the tick labels and set their alignment.
#     # plt.setp(ax_dict["heatmap"].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#     ax_dict["heatmap"].spines[:].set_visible(False)
#     ax_dict["heatmap"].grid(which="minor", color="w", linestyle='-', linewidth=3)
#     # Loop over data dimensions and create text annotations.
#     for i in range(att_h):
#         for j in range(att_w):
#             text = ax_dict["heatmap"].text(j, i, att_map[i, j], ha="center", va="center", color="w")
#     ax_dict["heatmap"].set_title("heatmap")

#     ax_dict["contour"].imshow(img)
#     alphas = Normalize(0, .3, clip=True)(np.abs(att_img))
#     alphas = np.clip(alphas, .4, 1)  # alpha value clipped at the bottom at .4
#     ax_dict["contour"].imshow(att_img, alpha=alphas, cmap=cmap)
#     ax_dict["contour"].contour(att_img, levels=[.5], colors='k', linestyles='-')
#     # ax_dict["heatmap"].set_axis_off()
#     ax_dict["contour"].set_title("contour")

#     fig.tight_layout()
#     plt.show()
#     plt.close('all')
#     pdb.set_trace()