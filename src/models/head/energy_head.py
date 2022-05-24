import pdb
import time
import torch
import torchvision
import numpy as np
import matplotlib
# matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F 
from torch import nn

from einops import rearrange, reduce, repeat
from ..layers.transformer import CLSEncoderLayer
from ..layers.projection import *
from ..backbone.resnet import *


class EnergyHead(nn.Module):
	def __init__(self, pseudo_ratio: float = 0.5, img_dim: int = 7, output_dim: int = 512):
		super(EnergyHead, self).__init__()
		self.pseudo_ratio = pseudo_ratio

		resnet = torchvision.models.resnet18()
		if img_dim != 3: 
			resnet.conv1 = nn.Conv2d(img_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.backbone = nn.Sequential(*list(resnet.children())[:-1])

		self.classifier0 = PseudoAttention(embed_dim=output_dim)
		self.classifier1 = PseudoAttention(embed_dim=output_dim)

	def forward(
		self,
		p: torch.Tensor,
		t: torch.Tensor,
		g: torch.Tensor,
		has_g: torch.Tensor,
	):
		output_dict = {}

		b, _, _, _ = p.shape
		not_g = torch.logical_not(has_g)
		len_gt, len_gt_not = sum(has_g), sum(not_g)
		len_topk = min(int(len_gt*self.pseudo_ratio), len_gt_not)

		if len_gt == 0 or len_gt == b:
			return {}

		p = self.backbone(p).flatten(start_dim=1)
		t = self.backbone(t).flatten(start_dim=1)
		g = self.backbone(g[has_g]).flatten(start_dim=1)

		p_g, t_g, p_n, t_n = p[has_g], t[has_g], p[not_g], t[not_g]
		key = torch.cat([g, t_g, p_g, t_n, p_n], dim=0)
		query_gt_tn = torch.cat([g, t_n], dim=0)
		query_gt_pn = torch.cat([g, p_n], dim=0)
		

		# 0: gt+tn -> tg+pg+tn+pn
		logits0_gt_tn = self.classifier0(query_gt_tn, key)
		logits0_gt, logits0_tn = logits0_gt_tn[:len_gt], logits0_gt_tn[len_gt:]

		# ***query: gt***
		b0_gt, d0_gt = logits0_gt.shape
		# mask self
		mask0_gt = torch.arange(len_gt).type_as(p).long()
		mask0_gt_neg = F.one_hot(mask0_gt, num_classes=d0_gt).bool()
		mask0_gt_neg = torch.logical_not(mask0_gt_neg)
		logits0_gt_mask = logits0_gt[mask0_gt_neg].reshape(b0_gt, -1)
		# generate label
		label0_gt = torch.arange(2*len_gt-1, 3*len_gt-1).type_as(p).long()
		label0_gt = F.one_hot(label0_gt, num_classes=d0_gt-1).bool()

		# ***query t not has g***
		b0_tn, d0_tn = logits0_tn.shape
		# mask self
		mask0_tn = torch.arange(3*len_gt, 3*len_gt+len_gt_not).type_as(p).long()
		mask0_tn = F.one_hot(mask0_tn, num_classes=d0_tn).bool()
		mask0_tn = torch.logical_not(mask0_tn)
		logits0_tn_mask = logits0_tn[mask0_tn].reshape(b0_tn, -1)
		# generate label
		label0_tn = torch.arange(3*len_gt+len_gt_not-1, 3*len_gt+2*len_gt_not-1).type_as(p).long()
		label0_tn = F.one_hot(label0_tn, num_classes=d0_tn-1).bool()
		# select top k
		_, indice0_tn_topk = logits0_tn_mask[label0_tn].topk(len_topk)
		indice0_tn_topk, _ = torch.sort(indice0_tn_topk)
		# mask negative
		mask0_tn_neg = torch.zeros(b0_tn).type_as(p).bool()
		mask0_tn_neg[indice0_tn_topk] = True
		mask0_tn_neg = torch.logical_not(mask0_tn_neg)
		pdb.set_trace()
		logits0_tn_pn = torch.where(logits0_tn_mask[label0_tn])
		logits0_tn_pn[mask0_tn_neg] = logits0_tn_pn[mask0_tn_neg].pow(-1)
		logits0_tn_mask[label0_tn] = logits0_tn_pn

		
		# output t supervise p energy
		mask0_tn_topk = torch.zeros(b0_tn).type_as(p).bool()
		mask0_tn_topk[indice0_tn_topk] = True
		mask_p = torch.zeros(b).type_as(p).bool()
		mask_p[not_g] = mask0_tn_topk
		# loss
		logits0 = torch.cat([logits0_gt_mask, logits0_tn_mask], dim=0)
		mask0 = torch.cat([label0_gt, label0_tn], dim=0)
		loss0 = self.ntx_loss(logits0, mask0)


		# 1: gt+pn -> tg+pg+tn+pn
		logits1_gt_gp = self.classifier1(query_gt_pn, key)
		logits1_gt, logits1_gp = logits1_gt_gp[:len_gt], logits1_gt_gp[len_gt:]

		# ***query: gt***
		b1_gt, d1_gt = logits1_gt.shape
		# mask self
		mask1_gt_not = torch.arange(len_gt).type_as(p).long()
		mask1_gt_not = F.one_hot(mask1_gt_not, num_classes=d1_gt).bool()
		mask1_gt_not = torch.logical_not(mask1_gt_not)
		logits1_gt_mask = logits1_gt[mask1_gt_not].reshape(b1_gt, -1)
		# generate label
		label1_gt = torch.arange(len_gt-1, 2*len_gt-1).type_as(p).long()
		label1_gt = F.one_hot(label1_gt, num_classes=d1_gt-1).bool()

		# ***query p not has g***
		b1_gp, d1_gp = logits1_gp.shape
		# mask self
		mask1_gp = torch.arange(3*len_gt+len_gt_not, 3*len_gt+2*len_gt_not).type_as(p).long()
		mask1_gp = F.one_hot(mask1_gp, num_classes=d1_gp).bool()
		mask1_gp = torch.logical_not(mask1_gp)
		logits1_gp_mask = logits1_gp[mask1_gp].reshape(b1_gp, -1)
		# generate label
		label1_gp = torch.arange(3*len_gt, 3*len_gt+len_gt_not).type_as(p).long()
		label1_gp = F.one_hot(label1_gp, num_classes=d1_gp-1).bool()
		# select top k
		_, indice1_gp_topk = logits1_gp_mask[label1_gp].topk(len_topk)
		indice1_gp_topk, _ = torch.sort(indice1_gp_topk)
		# mask negative
		mask1_gp_neg = torch.zeros(b1_gp).type_as(p) - 1
		mask1_gp_neg = mask1_gp_neg.scatter_(dim=0, index=indice1_gp_topk, value=1.)
		logits1_gp_mask[label1_gp] *= mask1_gp_neg
		# output p supervise t energy
		mask1_gp_topk = torch.zeros(b1_gp).type_as(p).bool()
		mask1_gp_topk[indice1_gp_topk] = True
		mask_t = torch.zeros(b).type_as(p).bool()
		mask_t[not_g] = mask1_gp_topk
		# loss
		logits1 = torch.cat([logits1_gt_mask, logits1_gp_mask], dim=0)
		mask1 = torch.cat([label1_gt, label1_gp], dim=0)
		loss1 = self.ntx_loss(logits1, mask1)
		
		# final loss
		loss = loss0 + loss1

		output_dict['energy_loss'] = loss
		output_dict['energy_online'] = mask_p
		output_dict['energy_target'] = mask_t

		# viualize logits
		mask_gt = torch.zeros(len_gt).type_as(p).bool()
		mask0_tn_neg = torch.cat([mask_gt, mask0_tn_neg], dim=0)
		mask1_gp_neg = torch.cat([mask_gt, mask1_gp_neg], dim=0)
		mask_pt = torch.cat([mask0_tn_neg, mask1_gp_neg], dim=0)[:, None].clone().detach()

		visual_map = torch.cat([logits0_gt_tn, logits1_gt_gp], dim=0).clone().detach()
		visual = (-visual_map) / torch.max(-visual_map).max()
		visual = torch.cat([visual, mask_pt], dim=1)

		# self.debug_rend_out(logits0_gt_tn, logits1_gt_gp, mask_p, mask_t)
		# pdb.set_trace()
		# self.save(logits0_gt_tn, logits0[mask0], logits1_gt_gp, logits1[mask1])

		output_dict.update({'energy_visual': self.visualize(visual)})
		return output_dict

	def ntx_loss(self, logits, mask):
		ntx = -torch.log(torch.sum(logits.softmax(dim=1)*mask, dim=1))
		return ntx.mean()

	def visualize(self, logits):
		visual = repeat(logits, 'n m -> 1 (n h) (m w)', h=8, w=8)
		return visual

	def save(self, x0, y0, x1, y1):
		np.savez(
			f'/home/public/tianyiYue/Workspace/RS-ptn0.6/logs/{time.time()}',
			x0=-x0.detach().cpu(),
			y0=-y0.detach().cpu(),
			x1=-x1.detach().cpu(),
			y1=-y1.detach().cpu(),
		)

	def debug_rend_out(self, energy_TP, energy_PT, energy_P, energy_T):
		h, w = energy_TP.shape

		energy_TP = -energy_TP.detach().cpu().numpy()
		energy_PT = -energy_PT.detach().cpu().numpy()
		energy_P = energy_P.detach().cpu().numpy()
		energy_T = energy_T.detach().cpu().numpy()

		fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(5.5, 3.5), constrained_layout=True)

		im0 = axs[0].imshow(energy_TP, cmap='RdYlBu')
		# ax_fig10.set_xticks(np.arange(h))
		# ax_fig10.set_yticks(np.arange(w))
		# ax_fig10.set_xticklabels(np.arange(h))
		# ax_fig10.set_yticklabels(np.arange(w))
		axs[0].tick_params(which="minor", bottom=False, left=False)

		im1 = axs[1].imshow(energy_PT, cmap='RdYlBu')
		# ax_fig11.set_xticks(np.arange(h))
		# ax_fig11.set_yticks(np.arange(w))
		# ax_fig11.set_xticklabels(np.arange(h))
		# ax_fig11.set_yticklabels(np.arange(w))
		axs[1].tick_params(which="minor", bottom=False, left=False)

		fig.colorbar(im0, ax=axs[0], shrink=1.)
		fig.colorbar(im1, ax=axs[1], shrink=1.)

		plt.show()
		plt.close('all')
		pdb.set_trace()

class PseudoAttention(nn.Module):
	def __init__(self, input_dim=512, embed_dim=256):
		super(PseudoAttention, self).__init__()
		self.attend = nn.Softmax(dim=-1)

		self.projection = ProjectionHead((
			(input_dim, embed_dim, None, nn.ReLU()),
			(embed_dim, embed_dim, None, None),
		))

	def forward(self, x0, x1):
		b0, _ = x0.shape
		b1, _ = x1.shape

		x0 = self.projection(x0)
		x1 = self.projection(x1)

		logits_0 = F.softmax(x0, dim=-1).clamp(min=1e-7, max=1.)
		logits_1 = F.softmax(x1, dim=-1).clamp(min=1e-7, max=1.)

		entropy0 = torch.einsum('bc,bc->b', logits_0, logits_0.log())
		cross_entropy_01 = torch.einsum('nc,mc->nm', logits_0, logits_1.log())
		# [-inf -> 0]
		KL_01 = cross_entropy_01 - entropy0[:,None]
		return KL_01

class EnergyLoss(nn.Module):
	def __init__(self, probability: str = 'KL', eps: float = 1e-8):
		super(EnergyLoss, self).__init__()
		self.eps = eps
		self.probability = probability
		self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")

	def forward(self,
		out0: torch.Tensor,
		out1: torch.Tensor,
		temperature: float = 1.,
		attention: bool = False,
		visualize: bool = False,
	):

		device = out0.device
		batch_size, _ = out0.shape

		if self.probability == 'COS':
			# soft contrast
			out0 = torch.nn.functional.normalize(out0, dim=1)
			out1 = torch.nn.functional.normalize(out1, dim=1)
			logits_00 = torch.einsum('nc,mc->nm', out0, out0)
			logits_01 = torch.einsum('nc,mc->nm', out0, out1)
			logits_10 = torch.einsum('nc,mc->nm', out1, out0)
			logits_11 = torch.einsum('nc,mc->nm', out1, out1)

		elif self.probability == 'KL':
			logits_0 = F.softmax(out0, dim=-1).clamp(min=1e-7, max=1.)
			logits_1 = F.softmax(out1, dim=-1).clamp(min=1e-7, max=1.)

			entropy0 = torch.einsum('bc,bc->b', logits_0, logits_0.log())
			entropy1 = torch.einsum('bc,bc->b', logits_1, logits_1.log())
			cross_entropy_00 = torch.einsum('nc,mc->nm', logits_0, logits_0.log())
			cross_entropy_01 = torch.einsum('nc,mc->nm', logits_0, logits_1.log())
			cross_entropy_10 = torch.einsum('nc,mc->nm', logits_1, logits_0.log())
			cross_entropy_11 = torch.einsum('nc,mc->nm', logits_1, logits_1.log())
			logits_00 = cross_entropy_00 - entropy0 + 1
			logits_01 = cross_entropy_01 - entropy0 + 1
			logits_10 = cross_entropy_10 - entropy1 + 1
			logits_11 = cross_entropy_11 - entropy1 + 1
		else:
			assert False

		# initialize labels and masks
		labels = torch.arange(batch_size, device=device, dtype=torch.long)
		masks = torch.ones_like(logits_00).bool()
		masks.scatter_(dim=1, index=labels.unsqueeze(1), value=False)
			
		# remove similarities of samples to themselves
		logits_00 = logits_00[masks].view(batch_size, -1)
		logits_11 = logits_11[masks].view(batch_size, -1)

		# concatenate logits
		# the logits tensor in the end has shape (2*n, 2*m-1)
		logits_0100 = torch.cat([logits_01, logits_00], dim=1)
		logits_1011 = torch.cat([logits_10, logits_11], dim=1)
		logits = torch.cat([logits_0100, logits_1011], dim=0) / (temperature + self.eps)

		# repeat twice to match shape of logits
		labels = labels.repeat(2)

		loss = self.criterion(logits, labels) * (temperature**2)

		output_dict = {'energy_loss': loss}

		if attention:
			masks = torch.eye(batch_size).bool()
			energy0 = logits_01[masks].softmax(dim=-1)
			# energy0 = logits0 / (logits0.max().detach() + self.eps)
			energy1 = logits_10[masks].softmax(dim=-1)
			# energy1 = logits1 / (logits1.max().detach() + self.eps)
			output_dict.update({			
				'energy_online': energy0,
				'energy_target': energy1,
			})

		if visualize:
			visualize_energy = torch.cat([energy0, energy1], dim=0)[:, None].clone().detach()
			visualize_logits = logits.clone().detach()

			visualize_logits = (1 - visualize_logits)
			visualize_logits /= visualize_logits.max()
 
			visualize = torch.cat([visualize_logits, visualize_energy], dim=1)
			visualize = repeat(visualize, 'n m -> 1 (n h) (m w)', h=8, w=8)
			output_dict.update({'energy_visualize': visualize})

		return output_dict
