import pdb
import time
import torch
import numpy as np
import torch.nn as nn

from loguru import logger
from .cotrain_losses import CoTrainingLoss, PseudoLoss
from ..core import constants
from ..core.config import SMPL_MODEL_DIR
from ..utils.geometry import batch_rodrigues, cal_normal_distance

class SSPALoss(nn.Module):
    def __init__(
        self,
        hparams,
        shape_loss_weight=0.,
        keypoint_loss_weight=5.,
        pose_loss_weight=1.,
        beta_loss_weight=0.001,
        openpose_train_weight=0.,
        gt_train_weight=1.,
        loss_weight=60.,
    ):
        super(THMRLoss, self).__init__()
        self.hparams = hparams
        self.criterion_shape = nn.L1Loss()
        self.criterion_keypoints = nn.MSELoss(reduction='none')
        self.criterion_regr = nn.MSELoss()
        self.criterion_surface = eval(f'{hparams.THMR.CRITERION}')()
        self.criterion_co = CoTrainingLoss(
            self.criterion_surface,
            use_pseudo=False,
            use_regress=False,
        )

        self.loss_weight = loss_weight
        self.gt_train_weight = gt_train_weight
        self.pose_loss_weight = pose_loss_weight
        self.beta_loss_weight = beta_loss_weight
        self.shape_loss_weight = shape_loss_weight
        self.keypoint_loss_weight = keypoint_loss_weight
        self.openpose_train_weight = openpose_train_weight

    def forward(self, pred, gt):
        loss_dict = {}

        pred_cam = pred['pred_cam']
        pred_betas = pred['pred_shape']
        pred_rotmat = pred['pred_rotmat']
        pred_joints = pred['smpl_joints3d']
        pred_vertices = pred['smpl_vertices']
        pred_projected_keypoints_2d = pred['smpl_joints2d']
        
        gt_pose = gt['pose']
        gt_betas = gt['betas']
        gt_joints = gt['pose_3d']
        gt_vertices = gt['vertices']
        gt_keypoints_2d = gt['keypoints']
        gt_cam_t = gt['gt_cam_t']
        has_smpl = gt['has_smpl'].bool()
        has_pose_3d = gt['has_pose_3d'].bool()
        has_pose_2d = gt['has_pose_2d'].bool()
        has_surface_2d = gt['has_surface_2d'].bool()

        online_rgbo = pred['online_rgbo']
        online_al = pred['online_al']
        online_al_label = pred['online_al_label']
        
        target_rgba = pred['target_rgba']
        target_al = pred['target_al']
        target_al_origin = pred['target_al_origin']

        gt_rgbo = gt['gt_rgbo']
        gt_al = gt['gt_al']
        gt_al_label = gt['gt_al_label']

        online_energy = pred['energy_online'] if 'energy_online' in pred.keys() else None
        target_energy = pred['energy_target'] if 'energy_target' in pred.keys() else None
        loss_online, loss_target = self.criterion_co(
            online_rgbo, target_rgba, gt_rgbo,
            online_al, target_al, gt_al,
            online_al_label, target_al_origin, gt_al_label,
            online_energy, target_energy, has_surface_2d,
        )
        loss_dict['step/regression'] = loss_online * self.hparams.THMR.SURFACE_LOSS_WEIGHT
        loss_dict['step/generation'] = loss_target * self.hparams.THMR.PSEUDO_LOSS_WEIGHT
        loss_dict['step/attention'] = pred['loss_attention'] * self.hparams.THMR.ENERGY_LOSS_WEIGHT

        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = smpl_losses(
            pred_rotmat,
            pred_betas,
            gt_pose,
            gt_betas,
            has_smpl,
            criterion=self.criterion_regr,
        )

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = projected_keypoint_loss(
            pred_projected_keypoints_2d,
            gt_keypoints_2d,
            self.openpose_train_weight,
            self.gt_train_weight,
            has_pose_2d,
            criterion=self.criterion_keypoints,
        )

        # Compute 3D keypoint loss
        loss_keypoints_3d = keypoint_3d_loss(
            pred_joints,
            gt_joints,
            has_pose_3d,
            criterion=self.criterion_keypoints,
        )

        # # Per-vertex loss for the shape
        loss_shape = shape_loss(
            pred_vertices,
            gt_vertices,
            has_smpl,
            criterion=self.criterion_shape,
        )
        
        # loss_shape *= self.shape_loss_weight
        loss_keypoints *= self.keypoint_loss_weight
        loss_keypoints_3d *= self.keypoint_loss_weight
        loss_regr_pose *= self.pose_loss_weight
        loss_regr_betas *= self.beta_loss_weight
        loss_cam = ((torch.exp(-pred_cam[:, 0] * 10)) ** 2).mean()

        loss_dict.update({
            'step/train_keypoints': loss_keypoints,
            'step/train_keypoints_3d': loss_keypoints_3d,
            'step/train_regr_pose': loss_regr_pose,
            'step/train_regr_betas': loss_regr_betas,
            'step/train_shape': loss_shape,
            'step/train_cam': loss_cam,
        })

        loss = sum(loss for loss in loss_dict.values())
        loss *= self.loss_weight
        loss_dict['step/total_loss'] = loss

        if loss.isnan():
            pdb.set_trace()
        return loss, loss_dict

class MPQALoss(nn.Module):
    def __init__(
        self,
        shape_loss_weight=0.,
        keypoint_loss_weight=5.,
        pose_loss_weight=1.,
        beta_loss_weight=0.001,
        openpose_train_weight=0.,
        gt_train_weight=1.,
        loss_weight=60.,
    ):
        super(MPQALoss, self).__init__()
        self.criterion_shape = nn.L1Loss()
        self.criterion_keypoints = nn.MSELoss(reduction='none')
        self.criterion_regr = nn.MSELoss()

        self.loss_weight = loss_weight
        self.gt_train_weight = gt_train_weight
        self.pose_loss_weight = pose_loss_weight
        self.beta_loss_weight = beta_loss_weight
        self.shape_loss_weight = shape_loss_weight
        self.keypoint_loss_weight = keypoint_loss_weight
        self.openpose_train_weight = openpose_train_weight

    def forward(self, pred, gt):
        loss_dict = {}

        pred_cam = pred['pred_cam']
        pred_betas = pred['pred_shape']
        pred_rotmat = pred['pred_rotmat']
        pred_joints = pred['smpl_joints3d']
        pred_vertices = pred['smpl_vertices']
        pred_projected_keypoints_2d = pred['smpl_joints2d']
        
        gt_pose = gt['pose']
        gt_betas = gt['betas']
        gt_joints = gt['pose_3d']
        gt_vertices = gt['vertices']
        gt_keypoints_2d = gt['keypoints']
        gt_cam_t = gt['gt_cam_t']
        has_smpl = gt['has_smpl'].bool()
        has_pose_3d = gt['has_pose_3d'].bool()
        has_pose_2d = gt['has_pose_2d'].bool()
        has_surface_2d = gt['has_surface_2d'].bool()

        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = smpl_losses(
            pred_rotmat,
            pred_betas,
            gt_pose,
            gt_betas,
            has_smpl,
            criterion=self.criterion_regr,
        )

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = projected_keypoint_loss(
            pred_projected_keypoints_2d,
            gt_keypoints_2d,
            self.openpose_train_weight,
            self.gt_train_weight,
            has_pose_2d,
            criterion=self.criterion_keypoints,
        )

        # Compute 3D keypoint loss
        loss_keypoints_3d = keypoint_3d_loss(
            pred_joints,
            gt_joints,
            has_pose_3d,
            criterion=self.criterion_keypoints,
        )

        # Per-vertex loss for the shape
        loss_shape = shape_loss(
            pred_vertices,
            gt_vertices,
            has_smpl,
            criterion=self.criterion_shape,
        )
        
        # loss_shape *= self.shape_loss_weight
        loss_keypoints *= self.keypoint_loss_weight
        loss_keypoints_3d *= self.keypoint_loss_weight
        loss_regr_pose *= self.pose_loss_weight
        loss_regr_betas *= self.beta_loss_weight
        loss_cam = ((torch.exp(-pred_cam[:, 0] * 10)) ** 2).mean()

        loss_dict.update({
            'step/train_keypoints': loss_keypoints,
            'step/train_keypoints_3d': loss_keypoints_3d,
            'step/train_regr_pose': loss_regr_pose,
            'step/train_regr_betas': loss_regr_betas,
            'step/train_shape': loss_shape,
            'step/train_cam': loss_cam,
        })

        loss = sum(loss for loss in loss_dict.values())
        loss *= self.loss_weight
        loss_dict['step/total_loss'] = loss

        if loss.isnan():
            pdb.set_trace()
        return loss, loss_dict

def projected_keypoint_loss(
        pred_keypoints_2d,
        gt_keypoints_2d,
        openpose_weight,
        gt_weight,
        has_pose_2d,
        criterion,
):
    """ Compute 2D reprojection loss on the keypoints.
    The loss is weighted by the confidence.
    The available keypoints are different for each dataset.
    """
    if torch.sum(has_pose_2d) > 0:
        pd_kp = pred_keypoints_2d[has_pose_2d]
        gt_kp = gt_keypoints_2d[has_pose_2d]

        conf = gt_kp[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * criterion(pd_kp, gt_kp[:, :, :-1])).mean()
        return loss
    else:
        return torch.tensor(0., device=pred_keypoints_2d.device, requires_grad=False)


def keypoint_loss(
        pred_keypoints_2d,
        gt_keypoints_2d,
        criterion,
):
    """ Compute 2D reprojection loss on the keypoints.
    The loss is weighted by the confidence.
    The available keypoints are different for each dataset.
    """
    loss = criterion(pred_keypoints_2d, gt_keypoints_2d).mean()
    return loss


def keypoint_3d_loss(
        pred_keypoints_3d,
        gt_keypoints_3d,
        has_pose_3d,
        criterion,
):
    """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
    The loss is weighted by the confidence.
    """
    pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    if len(gt_keypoints_3d) > 0:
        gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        return (conf * criterion(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.tensor(0., device=pred_keypoints_3d.device, requires_grad=False)


def shape_loss(
        pred_vertices,
        gt_vertices,
        has_smpl,
        criterion,
):
    """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
    pred_vertices_with_shape = pred_vertices[has_smpl == 1]
    gt_vertices_with_shape = gt_vertices[has_smpl == 1]
    if len(gt_vertices_with_shape) > 0:
        return criterion(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.tensor(0., device=pred_vertices.device, requires_grad=False)


def smpl_losses(
        pred_rotmat,
        pred_betas,
        gt_pose,
        gt_betas,
        has_smpl,
        criterion,
):
    pred_rotmat_valid = pred_rotmat[has_smpl == 1]
    gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]
    pred_betas_valid = pred_betas[has_smpl == 1]
    gt_betas_valid = gt_betas[has_smpl == 1]
    if len(pred_rotmat_valid) > 0:
        loss_regr_pose = criterion(pred_rotmat_valid, gt_rotmat_valid)
        loss_regr_betas = criterion(pred_betas_valid, gt_betas_valid)
    else:
        loss_regr_pose = torch.tensor(0., device=pred_rotmat.device, requires_grad=False)
        loss_regr_betas = torch.tensor(0., device=pred_rotmat.device, requires_grad=False)
    return loss_regr_pose, loss_regr_betas


def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims) + 1e-6
    union = (predict + target - predict * target).sum(dims) + 1e-6
    neg_iou = 1. - (intersect / union).sum() / intersect.nelement()
    return neg_iou


def smpl_condition_loss(has_smpl, criterion, *arg, **kwarg):
    """ Compute smpl condition loss on the smpl model ground truth."""
    arg = list(arg)
    for i, v in enumerate(arg):
        assert v is not None
        arg[i] = v[has_smpl == 1]

    for k, v in kwarg.items():
        assert v is not None
        kwarg[k] = v[has_smpl == 1]

    if torch.sum(has_smpl) > 0:
        return criterion(*arg, **kwarg)
    else:
        return torch.tensor(0., device=has_smpl.device, requires_grad=False)


def surface_loss(
    pd, gt,
    pd_mask, gt_mask,
    has_smpl, criterion,
):
    if torch.sum(has_smpl) > 0:
        pd_valid = pd[has_smpl]
        gt_valid = gt[has_smpl]
        pd_mask_valid = pd_mask[has_smpl]
        gt_mask_valid = gt_mask[has_smpl]
        return criterion(pd_valid, gt_valid, pd_mask_valid, gt_mask_valid)
    else:
        return torch.tensor(0., device=pd.device, requires_grad=False)