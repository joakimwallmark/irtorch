from abc import ABC, abstractmethod
import torch
from torch import nn
import torch.nn.functional as F

class BaseEncoder(ABC, nn.Module):
    """
    Abstract base class for Item Response Theory models.
    """

    def __init__(
        self,
        input_dim: int,
        latent_variables: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_variables = latent_variables


    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        output : torch.Tensor
            Output tensor.
        """
