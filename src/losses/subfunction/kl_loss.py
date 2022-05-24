import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

class KLLoss(nn.Module):
	def __init__(self, temperature: float = 0.5):
		super(KLLoss, self).__init__()
		self.temperature = temperature
		# self.criterion_NLL = nn.NLLLoss(reduction="mean")
		self.criterion_CE = nn.CrossEntropyLoss(reduction="mean")
		self.eps = 1e-8

		if abs(self.temperature) < self.eps:
			raise ValueError('Illegal temperature: abs({}) < 1e-8'.format(self.temperature))

	def forward(
		self,
		out0: torch.Tensor,
		out1: torch.Tensor
	):
		d = out0.device
		b, d = out0.shape

		out0_logits = F.softmax(out0, dim=-1).clamp(min=1e-7, max=1.)
		out1_logits = F.softmax(out1, dim=-1).clamp(min=1e-7, max=1.)

		# calculate similiarities
		# here n = batch_size and m = batch_size * world_size
		# the resulting vectors have shape (n, m)
		CE_00 = torch.einsum('nc,mc->nm', out0_logits, out0_logits.log())
		CE_01 = torch.einsum('nc,mc->nm', out1_logits, out0_logits.log())
		CE_10 = torch.einsum('nc,mc->nm', out0_logits, out1_logits.log())
		CE_11 = torch.einsum('nc,mc->nm', out1_logits, out1_logits.log())

		# initialize labels and masks
		labels = torch.arange(b, device=d, dtype=torch.long)
		masks = torch.ones_like(CE_00, dtype=torch.bool)
		masks.scatter_(dim=1, index=labels.unsqueeze(1), value=False)
		labels = labels.repeat(2)

		# remove similarities of samples to themselves
		CE_00 = rearrange(CE_00[masks], '(n m) ->n m', n=b)
		CE_11 = rearrange(CE_01[masks], '(n m) ->n m', n=b)

		# concatenate logits
		# the logits tensor in the end has shape (2*n, 2*m-1)
		CE_0100 = torch.cat([CE_01, CE_00], dim=1)
		CE_1011 = torch.cat([CE_10, CE_11], dim=1)
		logits = torch.cat([CE_0100, CE_1011], dim=0)

		# Cross Entropy
		loss = self.criterion_CE(logits, labels)

		# pdb.set_trace()
		return loss