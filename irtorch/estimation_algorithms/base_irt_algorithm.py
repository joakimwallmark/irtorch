from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import torch

if TYPE_CHECKING:
    from irtorch.models import BaseIRTModel

class BaseIRTAlgorithm(ABC):
    """
    Abstract base class for IRT algorithms. All IRT algorithms should inherit from this class.
    """
    def __init__(self):
        super().__init__()
        self.train_data = None
        self.training_theta_scores = None

    @abstractmethod
    def fit(self, model: BaseIRTModel, train_data: torch.Tensor, **kwargs):
        """
        Fit the model to the data.

        Parameters
        ----------
        model : BaseIRTModel, optional
            The model to train. Needs to inherit :class:`irtorch.models.BaseIRTModel`.
        train_data : torch.Tensor
            The training data.
        **kwargs
            Additional keyword arguments for the algorithm fit method.
        """
        # Store all training data within the algorithm instance
        if self.train_data is not None:
            self.train_data = torch.cat((self.train_data, train_data), dim=0).contiguous()
        else:
            self.train_data = train_data.contiguous()
