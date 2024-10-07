from abc import ABC, abstractmethod
import torch

class Scale(ABC):
    """
    Abstract base class for Item Response Theory model scale transformations.
    All scale transformations should inherit from this class.

    Not that you can make custom transformations by inheriting from this class.
    A class instance can then be supplied to :meth:`irtorch.models.BaseIRTModel.rescale` to apply the transformation to the latent variables of the model.
    The gradients method is not needed for 
    """
    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)
    
    @abstractmethod
    def transform(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Transforms the input theta scores into the new scale.

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing transformed theta scores. Each column represents one latent variable.
        """

    @abstractmethod
    def gradients(
        self,
        theta: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Computes the gradients of scale scores with respect to the input theta scores.

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing latent variable theta scores. Each column represents one latent variable.

        Returns
        -------
        torch.Tensor
            A torch tensor with the gradients for each theta score. Dimensions are (theta rows, latent variables, latent variables) where the last two are the jacobians.
        """
