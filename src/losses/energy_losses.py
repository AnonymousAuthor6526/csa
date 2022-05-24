import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger
from .subfunction.kl_loss import KLLoss
from .subfunction.ntx_ent_loss import NTXentLoss
from .subfunction.barlow_twins_loss import BarlowTwinsLoss
from .subfunction.negative_cosine_similarity import NegativeCosineSimilarity, PositiveCosineEmbeddingLoss

class EnergyLossDistill(nn.Module):
    def __init__(self, criterion, temperature=1.):
        super(EnergyLossDistill, self).__init__()
        self.temperature = temperature

        if criterion == 'BarlowTwins':
            self.criterion = BarlowTwinsLoss()
        elif criterion == 'SimCLR':
            self.criterion = NTXentLoss(temperature=temperature)
        elif criterion == 'MoCo':
            self.criterion = NTXentLoss(temperature=temperature, memory_bank_size=4096)
        elif criterion == 'cos':
            self.criterion = PositiveCosineEmbeddingLoss()
        elif criterion == 'KL':
            self.criterion = KLLoss(temperature=temperature)
        else:
            logger.error(f'{self.criterion} is undefined!')
            exit()

    def forward(self, p, t, f):
        """ Compute Energy losses for distillation learning """
        p_distill = self.criterion(p, f)
        t_distill = self.criterion(t, f)
        return p_distill + t_distill


class EnergyLossContrast(nn.Module):
    def __init__(self, criterion, temperature=1.):
        super(EnergyLossContrast, self).__init__()
        self.temperature = temperature

        if criterion == 'BarlowTwins':
            self.criterion = BarlowTwinsLoss()
        elif criterion == 'SimCLR':
            self.criterion = NTXentLoss(temperature=temperature)
        elif criterion == 'MoCo':
            self.criterion = NTXentLoss(temperature=temperature, memory_bank_size=4096)
        elif criterion == 'KL':
            self.criterion = KLLoss(temperature=temperature)
        else:
            logger.error(f'{self.criterion} is undefined!')
            exit()

    def forward(self, p, t):
        """ Compute Energy losses for distillation learning """
        return self.criterion(p, t)