import pdb
import torch
import numpy as np
import torch.nn as nn

from torch import linalg as LA
from loguru import logger
from einops import rearrange, reduce, repeat

class CoTrainingLoss(nn.Module):
    def __init__(self, criterion, use_pseudo, use_regress):
        super(CoTrainingLoss, self).__init__()
        self.criterion_online = criterion
        self.criterion_target = PseudoLoss()
        self.use_pseudo = use_pseudo
        self.use_regress = use_regress

    def forward(
        self, 
        online_rgbo, target_rgba, gt_rgbo,
        online_al, target_al, gt_al,
        online_al_label, target_al_origin, gt_al_label,
        online_energy=None, target_energy=None, gt_energy=None,
    ):
        """ Compute co-training losses for semi-supervised learning.3
        The loss is weighted by the x_w, y_w.
        Criterion of online network focus on foreground
        Criterion of target network constrain all pixel
        Input:
            x: size([B, RGBA, res, res]) is projection surface normal from online network, 
                where A encode the probabilty of foreground
            y: size([B, RGBA, res, res]) is perdiction surface normal from target network, 
                where A encode the probabilty of background

            x_mask: size([B, AL, res, res]) is part segmentation mask of x
                where A encode the probabilty of background
            y_mask: size([B, AL, res, res]) is part segmentation mask of y
                where A encode the probabilty of background

            x_w: size([B]) is the energy of training x
            y_w: size([B]) is the energy of training y

            gt: size([B, RGBA, res, res]) is the ground truth for x and y obtained from projection
                where A encode the probabilty of foreground
            gt_mask: size([B, AL, res, res]) is corresponding segmentation of gt
                where A encode the probabilty of background

            x_labels: size([B]) is the label for CrossEntropy loss function
        Return:
            co-training loss: size([]) in [0, 1]
        """
        device = online_rgbo.device

        predict_rgbo = []
        predict_l = []
        truth_rgbo = []
        truth_l = []

        if gt_energy.sum() > 0:
            predict_rgbo.append(online_rgbo[gt_energy])
            predict_l.append(online_al[gt_energy, 1:])
            truth_rgbo.append(gt_rgbo[gt_energy])
            truth_l.append(gt_al[gt_energy, 1:])

        if self.use_pseudo:
            target_rgbo_detach = torch.cat(
                [target_rgba[:, :3], 1 - target_rgba[:, 3:4]], 
                dim=1
            ).detach()
            if online_energy is not None:
                predict_rgbo.append(online_rgbo[online_energy])
                predict_l.append(online_al[online_energy, 1:])
                truth_rgbo.append(target_rgbo_detach[online_energy])
                truth_l.append(target_al[online_energy, 1:])
            else:
                predict_rgbo.append(online_rgbo)
                predict_l.append(online_al[:, 1:])
                truth_rgbo.append(target_rgbo_detach)
                truth_l.append(target_al[:, 1:])

        # if len(predict_rgbo) > 0:
        #     assert len(predict_rgbo) == len(truth_rgbo)
        #     predict_rgbo = torch.cat(predict_rgbo, dim=0)
        #     predict_l = torch.cat(predict_l, dim=0)
        #     truth_rgbo = torch.cat(truth_rgbo, dim=0)
        #     truth_l = torch.cat(truth_l, dim=0)

        #     loss_online = self.criterion_online(
        #         predict_rgbo, truth_rgbo, 
        #         predict_l, truth_l,
        #     )
        # else:
            
        loss_online = torch.tensor(0., device=device, requires_grad=False)


        pseudo_rgb = []
        pseudo_al = []
        truth_rgb = []
        truth_al = []

        if gt_energy.sum() > 0:
            pseudo_rgb.append(target_rgba[gt_energy])
            pseudo_al.append(target_al_origin[gt_energy])
            truth_rgb.append(gt_rgbo[gt_energy, :3])
            truth_al.append(gt_al_label[gt_energy])

        if self.use_regress:
            if target_energy is not None:
                pseudo_rgb.append(target_rgba[target_energy])
                pseudo_al.append(target_al_origin[target_energy])
                truth_rgb.append(online_rgbo[target_energy,:3].detach())
                truth_al.append(online_al_label[target_energy])
            else:
                pseudo_rgb.append(target_rgba)
                pseudo_al.append(target_al_origin)
                truth_rgb.append(online_rgbo[:,:3].detach())
                truth_al.append(online_al_label)

        if len(pseudo_rgb) > 0:
            assert len(pseudo_rgb) == len(truth_rgb)
            pseudo_rgb = torch.cat(pseudo_rgb, dim=0)
            pseudo_al = torch.cat(pseudo_al, dim=0)
            truth_rgb = torch.cat(truth_rgb, dim=0)
            truth_al = torch.cat(truth_al, dim=0)

            loss_target = self.criterion_target(
                pseudo_rgb, truth_rgb,
                pseudo_al, truth_al,
            )
        else:
            loss_target = torch.tensor(0., device=device, requires_grad=False)

        return loss_online, loss_target

class PseudoLoss(nn.Module):
    def __init__(self):
        super(PseudoLoss, self).__init__()
        self.criterion_segmentation = nn.CrossEntropyLoss(reduction='none')

    def forward(
        self, 
        pd_rgba, gt_rgbo, 
        pd_al, gt_al_label,
        weights=None
    ):
        loss_segmen = self.criterion_segmentation(pd_al, gt_al_label)
        loss_normal = LA.norm(pd_rgba[:, :3]-gt_rgbo[:, :3], ord=2, dim=1, keepdim=False)

        if weights is not None:
            loss_hw = loss_segmen + loss_normal
            loss = torch.mean(loss_hw, dim=[1,2])
            loss = torch.mul(loss, weights)
        else:
            loss = loss_segmen + loss_normal
        return loss.mean()