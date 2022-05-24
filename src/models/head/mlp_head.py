# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import pdb
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from src.utils import kp_utils
from ..layers import LocallyConnected2d, Conv3x3, Conv1x1, ConvInverse2x2
from ...core.config import SMPL_MEAN_PARAMS
from ...utils.geometry import rot6d_to_rotmat, get_coord_maps

class MLPHead(nn.Module):
    def __init__(
        self,
        pose_num,
        pose_dim,
        shape_dim,
        cam_dim,
        use_mean_pose=True, 
        use_mean_camshape=True, 
    ):
        super().__init__()
        self.pose_num = pose_num
        self.use_mean_pose = use_mean_pose
        self.use_mean_camshape = use_mean_camshape

        # Here we use 2 different MLPs to estimate shape and camera
        # They take a channelwise downsampled version of smpl features
        if pose_num == 24:
            self.pose_mlp = nn.Sequential(
                Rearrange('b n d -> b d n 1'),
                LocallyConnected2d(
                    in_channels=pose_dim,
                    out_channels=6,
                    output_size=[24, 1],
                    kernel_size=1,
                    stride=1,
                ),
                Rearrange('b d n 1 -> b n d'),
            )
            nn.init.xavier_uniform_(self.pose_mlp[1].weight, gain=0.01)
        else:
            self.smpl_joint_names = kp_utils.get_smpl_joint_names()
            self.smpl_map_dict = eval(f'kp_utils.map_smpl_to_parts{pose_num}_dict')()
            self.pose_mlp = nn.ModuleDict({
                smpl_name : nn.Linear(pose_dim, 6, bias=True) for smpl_name in self.smpl_joint_names
            })
            for v in self.pose_mlp._modules.values():
                nn.init.xavier_uniform_(v.weight, gain=0.01)
            

        self.shape_mlp = nn.Linear(shape_dim, 10, bias=True)
        self.cam_mlp = nn.Linear(cam_dim, 3, bias=True)
        nn.init.xavier_uniform_(self.cam_mlp.weight, gain=0.01)
        nn.init.xavier_uniform_(self.shape_mlp.weight, gain=0.01)

        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean_params['pose']).float()
        init_shape = torch.from_numpy(mean_params['shape']).float()
        init_cam = torch.from_numpy(mean_params['cam']).float()
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, pose_feat, shape_feat, cam_feat):
        output = {}
        b, n, d = pose_feat.shape

        rot6d_list = []
        if self.pose_num == 24:
            pred_pose = self.pose_mlp(pose_feat)
        else:
            for smpl_name, part_index in self.smpl_map_dict.items():
                smpl_rot6d = self.pose_mlp[smpl_name](pose_feat[:, part_index])
                smpl_rot6d = rearrange(smpl_rot6d, 'b d -> b 1 d')
                rot6d_list.append(smpl_rot6d)
                pred_pose = torch.cat(rot6d_list, dim=1)
        
        pred_cam = self.cam_mlp(cam_feat)
        pred_shape = self.shape_mlp(shape_feat)

        if self.use_mean_pose:
            init_pose = repeat(self.init_pose, '(d j) -> b j d', b=b, d=6)
            pred_pose = pred_pose + init_pose
        if self.use_mean_camshape:
            init_shape = repeat(self.init_shape, 'd -> b d', b=b)
            init_cam = repeat(self.init_cam, 'd -> b d', b=b)
            pred_cam = pred_cam + init_cam
            pred_shape = pred_shape + init_shape

        pred_rotmat = rot6d_to_rotmat(pred_pose).reshape(b, 24, 3, 3)
        
        return {
            'pred_pose': pred_pose,
            'pred_rotmat': pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
        }