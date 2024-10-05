from abc import ABC, abstractmethod
import torch

class Scale(ABC):
    """
    Abstract base class for Item Response Theory model scales.
    All scales should inherit from this class.
    """
    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)
    
    @abstractmethod
    def transform(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Transforms the input theta scores into the new scale.
        """
        raise NotImplementedError("You need to implement the forward method.")

    @abstractmethod
    def gradients(
        self,
        theta: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Computes the gradients of scale scores for each :math:`j` with respect to the input theta scores.

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing latent variable theta scores. Each column represents one latent variable.

        Returns
        -------
        torch.Tensor
            A torch tensor with the gradients for each theta score. Dimensions are (theta row, items, latent variable).
        """

    @abstractmethod
    def information(
        self,
        theta: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Computes the gradients of scale scores for each :math:`j` with respect to the input theta scores.

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing latent variable theta scores. Each column represents one latent variable.

        Returns
        -------
        torch.Tensor
            A torch tensor with the gradients for each theta score. Dimensions are (theta row, items, latent variable).
        """
