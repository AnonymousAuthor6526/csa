import pdb
import time
import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from torch.nn import functional as F
from collections import OrderedDict

from src.core.config import CONFIG_HOME
from src.utils import kp_utils
from .backbone import *
from .head import *
from .backbone.utils import get_backbone_info
from ..utils.train_utils import add_smpl_params_to_dict, prepare_statedict


class SSPA(nn.Module):
    def __init__(
        self,
        hparams,
        renderer,
        backbone: str = 'resnet50',
        regressor: str = 'mpqa',
        num_part: int = 11,
        img_res: int = 224,
        pretrained: str = None,
    ):
        super(SSPA, self).__init__()
        self.renderer = renderer
        self.pseudo_attention = hparams.THMR.USE_PSEUDO_ATTENTION
        num_joint = len(kp_utils.get_smpl_joint_names())

        cnn, use_conv = backbone.split('-')
        # hrnet_*-conv, hrnet_*-interp
        self.backbone = eval(cnn)(
            pretrained=True,
            downsample=False,
            use_conv=(use_conv == 'conv')
        )

        self.online_head = eval(f'{regressor}_head')(
            input_dim=get_backbone_info(backbone)['n_output_channels'],
            input_res=img_res // get_backbone_info(backbone)['downsample_rate'],
            num_query=num_part+1+1,
            num_joint=num_joint,
            img_res=img_res,
            use_mean_pose=True, 
            use_mean_camshape=True,
            visualize=True,
            hparams=hparams,
        )
        self.target_head = TargetPredictionHead(
            num_masks=hparams.THMR.RENDER_MASK, 
        )
        if self.pseudo_attention:
            probability, use_sa = hparams.THMR.PSEUDO_METHOD.split('-')
            self.energy_head = EnergyLearnerHead(
                probability=probability,
                temperature=hparams.THMR.PSEUDO_TEMPRATURE,
                attention=(use_sa=='SA'),
                visualize=True,
                output_dim=256,
            )

        if hparams.TRAINING.PRETRAINED is not None:
            self.load_pretrained(hparams.TRAINING.PRETRAINED)

    def load_pretrained(self, file):
        logger.info(f'Loading pretrained weights from {file}')
        state_dict = torch.load(file)['state_dict']
        model_state_dict = OrderedDict()
        for name, param in state_dict.items():
            if 'model' in name:
                name = name.replace('model.', '')
                model_state_dict[name] = param
        self.load_state_dict(model_state_dict, strict=True)
        logger.info(f'Loading pretrained complete')

    def forward(self, input, gt_rgbo=None, gt_al=None, has_gt=None):
        output = {}

        feature = self.backbone(input)
        online = self.online_head(feature)

        if self.training:
            online_render = self.renderer(
                vertices=online['smpl_vertices'],
                camera_translation=online['pred_cam_t'],
            )
            online.update({
                'online_rgbo': online_render["rend_img"],
                'online_al': online_render["part_mask"],
                'online_al_label': online_render["part_label"],
            })
            target = self.target_head(feature)

            if self.pseudo_attention:
                online_rgbal = torch.cat([online['online_rgbo'][:,:3], online['online_al']], dim=1)
                target_rgbal = torch.cat([target['target_rgba'][:,:3], target['target_al'] ], dim=1)
                gt_rgbal = torch.cat([gt_rgbo[:,:3], gt_al], dim=1)

                energy = self.energy_head(
                    online_rgbal,
                    target_rgbal,
                    gt_rgbal,
                    has_gt,
                )
                output.update(energy)
            output.update(target)

        output.update(online)
        return output


# Masked Part Query Self-Attention
class MPQA(nn.Module):
    def __init__(
        self,
        hparams,
        img_res,
        pretrained,
    ):
        super(MPQA, self).__init__()
        cfg = self.get_default_cfg(hparams, img_res)

        cnn, use_conv = hparams.BACKBONE.split('-')
        # hrnet_*-conv, hrnet_*-interp
        self.backbone = eval(cnn)(
            pretrained=True,
            downsample=False,
            use_conv=(use_conv == 'conv')
        )

        self.regression_branch = MPQAHead(
            cfg=cfg,
            visualize=True,
        )

        if pretrained is not None:
            self.load_pretrained(pretrained)

    def get_default_cfg(self, hparams, img_res):
        cfg = CN()
        cfg.BACKBONE = hparams.BACKBONE
        cfg.IMG_RES = img_res
        cfg.NUM_JOINT = len(kp_utils.get_smpl_joint_names())
        cfg.INPUT_DIM = get_backbone_info(hparams.BACKBONE)['n_output_channels']
        cfg.INPUT_RES = img_res // get_backbone_info(hparams.BACKBONE)['downsample_rate']
        cfg.NUM_CLS = hparams.NUM_CLS
        cfg.VIT_PATCH = hparams.VIT_PATCH
        cfg.VIT_FILTER = hparams.VIT_FILTER
        cfg.USE_POS_EMBED = hparams.USE_POS_EMBED
        cfg.USE_MAE_MASK = hparams.USE_MAE_MASK
        cfg.MAE_MASK_RATIO = hparams.MAE_MASK_RATIO
        cfg.USE_MEAN_POSE = True
        cfg.USE_MEAN_CAMSHAPE = True
        return cfg

    def load_pretrained(self, file):
        logger.info(f'Loading pretrained weights from {file}')
        state_dict = torch.load(file)['state_dict']
        model_state_dict = OrderedDict()
        for name, param in state_dict.items():
            if 'online_prediction_head' in name:
                name = name.replace('online_prediction_head', 'regression_branch')
            if 'pose_mlp' in name:
                name = name.replace('pose_mlp', 'pose_mlp.1')

            if 'target_prediction_head' in name:
                continue
            if 'mask_token' in name:
                continue
            if 'init_pose' in name:
                param = param[0]
            if 'init_shape' in name:
                param = param[0]
            if 'init_cam' in name:
                param = param[0]

            if 'model' in name:
                name = name.replace('model.', '')
                model_state_dict[name] = param
        self.load_state_dict(model_state_dict, strict=True)
        logger.info(f'Loading pretrained complete')

    def forward(self, input, **kwarg):
        output = {}
        feature = self.backbone(input)
        online = self.regression_branch(feature)
        output.update(online)
        return output


# Surface Representation Supervision
class SRS(nn.Module):
    def __init__(
        self,
        hparams,
        renderer,
        backbone: str = 'resnet50',
        regressor: str = 'mpqa',
        num_part: int = 11,
        img_res: int = 224,
        pretrained: str = None,
    ):
        super(SRS, self).__init__()
        self.renderer = renderer
        num_joint = len(kp_utils.get_smpl_joint_names())

        cnn, use_conv = backbone.split('-')
        self.backbone = eval(cnn)(
            pretrained=True,
            downsample=False,
            use_conv=(use_conv == 'conv')
        )

        self.online_head = eval(f'{regressor}_head')(
            input_dim=get_backbone_info(backbone)['n_output_channels'],
            input_res=img_res // get_backbone_info(backbone)['downsample_rate'],
            num_query=num_part+1+1,
            num_joint=num_joint,
            img_res=img_res,
            use_mean_pose=False, 
            use_mean_camshape=False,
            visualize=True,
            hparams=hparams,
        )

        if hparams.TRAINING.PRETRAINED is not None:
            self.load_pretrained(hparams.TRAINING.PRETRAINED)

    def load_pretrained(self, file):
        logger.info(f'Loading pretrained weights from {file}')
        state_dict = torch.load(file)['state_dict']
        model_state_dict = OrderedDict()
        for name, param in state_dict.items():
            if 'model' in name:
                model_state_dict[name] = param
        self.load_state_dict(model_state_dict, strict=True)
        logger.info(f'Loading pretrained complete')

    def forward(self, input, **kwarg):
        output = {}
        feature = self.backbone(input)
        online = self.online_head(feature)
        online_render = self.renderer(
            vertices=online['smpl_vertices'],
            camera_translation=online['pred_cam_t'],
        )
        online.update({
            'online_rgbo': online_render["rend_img"],
            'online_al': online_render["part_mask"],
            'online_al_label': online_render["part_label"],
        })
        output.update(online)
        return output