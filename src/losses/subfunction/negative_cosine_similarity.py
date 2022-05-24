""" Negative Cosine Similarity Loss Function """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
from torch.nn.functional import cosine_similarity


class NegativeCosineSimilarity(torch.nn.Module):
    """Implementation of the Negative Cosine Simililarity used in the SimSiam[0] paper.

    [0] SimSiam, 2020, https://arxiv.org/abs/2011.10566

    Examples:

        >>> # initialize loss function
        >>> loss_fn = NegativeCosineSimilarity()
        >>>
        >>> # generate two representation tensors
        >>> # with batch size 10 and dimension 128
        >>> x0 = torch.randn(10, 128)
        >>> x1 = torch.randn(10, 128)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(x0, x1)
    """

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        """Same parameters as in torch.nn.CosineSimilarity

        Args:
            dim (int, optional):
                Dimension where cosine similarity is computed. Default: 1
            eps (float, optional):
                Small value to avoid division by zero. Default: 1e-8
        """
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return -cosine_similarity(x0, x1, self.dim, self.eps).mean()

class PositiveCosineEmbeddingLoss(torch.nn.Module):
    def __init__(self, reduction: str = 'mean') -> None:
        """Same parameters as in torch.nn.CosineSimilarity

        Args:
            dim (int, optional):
                Dimension where cosine similarity is computed. Default: 1
            eps (float, optional):
                Small value to avoid division by zero. Default: 1e-8
        """
        super().__init__()
        self.criterion = torch.nn.CosineEmbeddingLoss(reduction=reduction)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        target = torch.ones(x0.shape[0]).type_as(x0)
        return self.criterion(x0, x1, target)
