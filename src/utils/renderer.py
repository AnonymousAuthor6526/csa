import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
# os.environ['DISPLAY'] = ':0.0'
# os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0] if 'GPU_DEVICE_ORDINAL' in os.environ.keys() else '0'

import cv2
import pdb
import math
import torch
import joblib
import trimesh
import pyrender
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
# import matplotlib.pyplot as plt
# plt.rcParams["savefig.bbox"] = 'tight'

from torchvision.utils import make_grid
from trimesh.visual import color
from loguru import logger
from einops import rearrange, reduce, repeat

from . import kp_utils
from .vis_utils import *

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, img_res=224, faces=None, mesh_color='blue'):
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=img_res,
            viewport_height=img_res,
            point_size=1.0
        )
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces
        self.mesh_color = get_colors()[mesh_color]
        
    def reverse_norm(self, images):
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        return images

    def visualize_tb(
        self, 
        vertices, camera_translation, images, 
        kp_2d=None, 
        online_mask=None, target_mask=None, gt_mask=None,
        online_rgbo=None, target_rgba=None, gt_rgbo=None,
        idx=6, sideview=False
    ):

        nrow = 2
        vertices = vertices[idx].cpu().numpy()
        camera_translation = camera_translation[idx].cpu().numpy()

        images = images[idx].cpu().clone()
        images = self.reverse_norm(images)
        images_np = images.permute(0,2,3,1).numpy()
        
        if kp_2d is not None:
            nrow += 1
            kp_2d = kp_2d[idx].cpu().numpy()

        if online_mask is not None:
            nrow += 1
            online_mask = online_mask[idx, 1:].cpu().bool()
            img_online_mask = visualize_segmentations(images, online_mask)
        if target_mask is not None:
            nrow += 1
            target_mask = target_mask[idx, 1:].cpu().bool()
            img_target_mask = visualize_segmentations(images, target_mask)
        if gt_mask is not None:
            nrow += 1
            gt_mask = gt_mask[idx, 1:].cpu().bool()
            img_gt_mask = visualize_segmentations(images, gt_mask)

        if online_rgbo is not None:
            nrow += 2
            img_online_rgbo = online_rgbo[idx].cpu()
            img_online_rgb = img_online_rgbo[:, :3]
            img_online_rgb = torch.where(
                img_online_rgb!=0,
                img_online_rgb/2+1/2,
                img_online_rgb
            )
            img_online_a = repeat(img_online_rgbo[:, 3], 'b h w -> b repeat h w', repeat=3)
        if target_rgba is not None:
            nrow += 2
            img_target_rgba = target_rgba[idx].cpu()
            img_target_rgb = img_target_rgba[:, :3]
            img_target_rgb = torch.where(
                img_target_rgb!=0,
                img_target_rgb/2+1/2,
                img_target_rgb
            )
            img_target_a = repeat(img_target_rgba[:, 3], 'b h w -> b repeat h w', repeat=3)
        if gt_rgbo is not None:
            nrow += 2
            img_gt_rgbo = gt_rgbo[idx].cpu()
            img_gt_rgb = img_gt_rgbo[:, :3]
            img_gt_rgb = torch.where(
                img_gt_rgb[:, :3]!=0,
                img_gt_rgb[:, :3]/2+1/2,
                img_gt_rgb[:, :3]
            )
            img_gt_a = repeat(img_gt_rgbo[:, 3].ceil(), 'b h w -> b repeat h w', repeat=3)

        if sideview: nrow += 1

        rend_imgs = []
        for i in range(images.shape[0]):
            rend_imgs.append(images[i])

            if online_rgbo is not None:
                rend_imgs.append(img_online_rgb[i])
                rend_imgs.append(img_online_a[i])
            if online_mask is not None:
                rend_imgs.append(img_online_mask[i])

            if target_rgba is not None:
                rend_imgs.append(img_target_rgb[i])
                rend_imgs.append(img_target_a[i])
            if target_mask is not None:
                rend_imgs.append(img_target_mask[i])

            if gt_rgbo is not None:
                rend_imgs.append(img_gt_rgb[i])
                rend_imgs.append(img_gt_a[i])
            if gt_mask is not None:
                rend_imgs.append(img_gt_mask[i])

            if kp_2d is not None:
                kp_img = draw_skeleton(images_np[i].copy(), kp_2d=kp_2d[i], dataset='common')
                kp_img = torch.from_numpy(np.transpose(kp_img, (2,0,1))).float()
                rend_imgs.append(kp_img)

            rend_img = torch.from_numpy(
                np.transpose(self.__call__(
                    vertices[i],
                    camera_translation[i],
                    images_np[i],
                ), (2,0,1))
            ).float()
            rend_imgs.append(rend_img)

            if sideview:
                side_img = torch.from_numpy(
                    np.transpose(
                        self.__call__(
                            vertices[i],
                            camera_translation[i],
                            np.ones_like(images_np[i]),
                            sideview=True,
                        ),
                        (2,0,1)
                    )
                ).float()
                rend_imgs.append(side_img)

        rend_imgs = make_grid(rend_imgs, nrow=nrow, normalize=False)
        return rend_imgs

    def visualize_attention_tb(self, img, att, idx=2, alpha=0.8):
        b, c, img_h, img_w = img.shape
        _, n, d = att.shape
        att_h, att_w = int(math.sqrt(d)), int(math.sqrt(d))

        att = att[idx].cpu().clone()
        img = img[idx].cpu().clone()
        img = self.reverse_norm(img)

        img = rearrange(img, 'b c h w -> b h w c')
        # visual real patch
        att_patch = repeat(
            att, 'b n (h w) -> b n (h repeat_h) (w repeat_w)', 
            h=att_h, w=att_w, repeat_h=img_h//att_h, repeat_w=img_w//att_w
        )

        # visual smooth
        att_inter = att.clamp(min=0., max=1.)
        att_inter = rearrange(att_inter, 'b n (h w) -> b n h w', h=att_h, w=att_w)
        att_inter = F.interpolate(att_inter, size=(img_h, img_w), mode='bilinear', align_corners=True)

        rend_imgs = []
        for i in range(img.shape[0]):
            masked_patch = rearrange(att_patch[i].sum(dim=0) < 0, 'h w -> h w 1').float()
            img_masked = img[i]*(1-masked_patch) + img[i]*masked_patch*(1-alpha) + 0.5*masked_patch*alpha
            rend_img = rearrange(img_masked, 'h w c -> c h w')
            rend_imgs.append(rend_img)

            for j in range(n):
                mask = (1-masked_patch).bool()
                img_heat = visualize_heatmaps(img_masked, att_inter[i, j], mask=mask)
                rend_img = rearrange(img_heat, 'h w c -> c h w')
                rend_imgs.append(rend_img)

        rend_imgs = make_grid(rend_imgs, nrow=1+n, normalize=False)
        return rend_imgs

    def __call__(
            self, vertices, camera_translation, image, vertex_colors=None,
            sideview=False, joint_labels=None, alpha=1.0, camera_rotation=None,
            sideview_angle=270, mesh_filename=None, mesh_inp=None,
            focal_length=None, cam_center=None,
    ):

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(self.mesh_color[0] / 255., self.mesh_color[1] / 255., self.mesh_color[2] / 255., alpha))

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices, self.faces, process=False)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        if sideview:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(sideview_angle), [0, 1, 0])
            mesh.apply_transform(rot)

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        if camera_rotation is not None:
            camera_pose[:3, :3] = camera_rotation
            camera_pose[:3, 3] = camera_rotation @ camera_translation
        else:
            camera_pose[:3, 3] = camera_translation

        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])

        scene.add(camera, pose=camera_pose)


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        # if joint_labels is not None:
        #     for joint, err in joint_labels.items():
        #         add_joints(scene, joints=joint, radius=err)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
        return output_img
