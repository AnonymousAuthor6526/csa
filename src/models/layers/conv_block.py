import pdb
import torch
import torch.nn as nn
import numpy as np

class Conv3x3(nn.Module):
    """docstring for Conv1x1"""
    def __init__(self, channel_in, channel_out, bias=False, active='nn.ReLU'):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding=1, bias=bias)
        self.norm = nn.SyncBatchNorm(channel_out)
        self.active = eval(active)()

    def forward(self, inputs):
        return self.active(self.norm(self.conv(inputs)))

class ConvUpsample3x3(nn.Module):
    """docstring for Conv1x1"""
    def __init__(self, channel_in, channel_out, align_corners=False, bias=False, active='nn.ReLU'):
        super(ConvUpsample3x3, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=align_corners)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, stride=1, padding=1, bias=bias)
        self.norm = nn.SyncBatchNorm(channel_out)
        self.active = eval(active)()

    def forward(self, inputs):
        x = self.upsample(inputs)
        x = self.conv(x)
        x = self.norm(x)
        x = self.active(x)
        return x

class Conv1x1(nn.Module):
    """docstring for Conv1x1"""
    def __init__(self, channel_in, channel_out, bias=False, active='nn.ReLU', **kwargs):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=1, padding=0, bias=bias)
        self.norm = nn.SyncBatchNorm(channel_out)
        if active == 'nn.Softmax':
            self.active = eval(active)(dim=kwargs['dim'])
        elif active == 'nn.LeakyReLU':
            self.active = eval(active)(negative_slope=kwargs['negative_slope'])
        else:
            self.active = eval(active)()

    def forward(self, inputs):
        return self.active(self.norm(self.conv(inputs)))

class ConvInverse2x2(nn.Module):
    """docstring for Conv1x1"""
    def __init__(self, channel_in, channel_out, bias=False, active='nn.ReLU'):
        super(ConvInverse2x2, self).__init__()
        self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=2, stride=2, bias=bias)
        self.norm = nn.SyncBatchNorm(channel_out)
        self.active = eval(active)()

    def forward(self, inputs):
        return self.active(self.norm(self.conv(inputs)))