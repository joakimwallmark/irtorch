from __future__ import annotations
from typing import TYPE_CHECKING
import logging
from factor_analyzer import Rotator
import torch
from irtorch.rescale import Scale

if TYPE_CHECKING:
    from irtorch.models.base_irt_model import BaseIRTModel

logger = logging.getLogger("irtorch")

class Rotate(Scale):
    """
    Rotates the latent variables to improve interpretability. Utilizes the `factor_analyzer <https://pypi.org/project/factor-analyzer/>`_ package for rotations.
    
    If the model has already been rescaled using :meth:`irtorch.rescale` the rotation is applied to the rescaled latent variables.
    If you do not want this, use :meth:`irtorch.models.BaseIRTModel.detach_rescale` before applying the rotation.
    
    Parameters
    ----------
    model : BaseIRTModel
        The IRT model which scales to rotate.
    data : torch.Tensor, optional
        The popluation data to compute the latent variable "loadings" for each item.
    theta : torch.Tensor, optional
        The original scale theta scores to compute the latent variable "loadings" for each item.
    loadings: torch.Tensor, optional
        A torch tensor with the loadings for each item. If specified, data and theta are ignored and the loadings are used for rotation. (default is None)
    rotation_method : str, optional
        The rotation method to use. For available options, see `factor_analyzer <https://pypi.org/project/factor-analyzer/>`_. (default is "promax")
    rotation_matrix : torch.Tensor, optional
        A torch tensor with the rotation matrix. If specified, data, theta and rotation_method are ignored
        and the rotation matrix is used directly. (default is None)
    **kwargs
        Additional keyword arguments used for theta estimation. Refer to :meth:`irtorch.models.BaseIRTModel.latent_scores` for additional details.
    """
    def __init__(
        self,
        model: BaseIRTModel,
        data: torch.Tensor = None,
        theta: torch.Tensor = None,
        loadings: torch.Tensor = None,
        rotation_method: str = "promax",
        rotation_matrix: torch.Tensor = None,
        **kwargs
    ):
        super().__init__(invertible=True)
        if rotation_matrix is None:
            if loadings is None:
                if theta is None:
                    if data is None:
                        raise ValueError("data, theta, loadings or a rotation matrix must be provided for rotation.")
                    theta = model.latent_scores(data, rescale=True ,**kwargs)
                loadings = model.expected_item_score_gradients(theta).mean(dim=0)
            rotator = Rotator(method = rotation_method)
            self.rot_loadings = torch.from_numpy(rotator.fit_transform(loadings.numpy())).float()
            self.rotation_matrix = torch.tensor(rotator.rotation_, dtype=torch.float)
        else:
            if rotation_matrix.shape[0] != model.latent_variables or rotation_matrix.shape[1] != model.latent_variables:
                raise ValueError("The rotation matrix must have the same dimensions as the number of latent variables.")
            self.rotation_matrix = rotation_matrix

        self.inverse_rotation_matrix = torch.inverse(self.rotation_matrix)

    def transform(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Transforms the input theta scores into the new scale.

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing latent variable theta scores. Each column represents one latent variable.
        """
        return torch.matmul(theta, self.rotation_matrix)

    def inverse(self, transformed_theta: torch.Tensor) -> torch.Tensor:
        """
        Puts the scores back to the original theta scale.

        Parameters
        ----------
        transformed_theta : torch.Tensor
            A 2D tensor containing transformed theta scores. Each column represents one latent variable.

        Returns
        -------
        torch.Tensor
            A 2D tensor containing theta scores on the the original scale.
        """
        unrotated_theta = torch.matmul(transformed_theta, self.inverse_rotation_matrix)
        return unrotated_theta


    def jacobian(
        self,
        theta: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Computes the gradients of rotated scores for each :math:`j` with respect to the original theta scores.

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing latent variable theta scores on the original theta scale. Each column represents one latent variable.

        Returns
        -------
        torch.Tensor
            A torch tensor with the gradients for each theta score. Dimensions are (theta rows, latent variables, latent variables) where the last two are the jacobians.
        """
        jacobians = self.rotation_matrix.T.unsqueeze(0).repeat(theta.shape[0], 1, 1)
        return jacobians
