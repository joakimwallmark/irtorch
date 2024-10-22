import logging
import torch
from irtorch.rescale import Scale

logger = logging.getLogger("irtorch")

class Reverse(Scale):
    """
    Reverses the chosen theta scales using.
    
    Parameters
    ----------
    reversed_latent_variables : list[bool]
        A list of booleans indicating which latent variables to reverse.
    
    Examples
    --------
    >>> import irtorch
    >>> from irtorch.models import NominalResponse, MonotoneNN
    >>> from irtorch.estimation_algorithms import AE, MML
    >>> from irtorch.rescale import Reverse
    >>> irtorch.set_seed(15)
    >>> data_sat, correct_responses = irtorch.load_dataset.swedish_sat_verbal()
    >>> model = NominalResponse(data=data_sat, mc_correct=correct_responses)
    >>> model.fit(train_data=data_sat, algorithm=MML())
    >>> model.plot.plot_item_probabilities(1).show()
    >>> # reverse the first (and only) latent variable
    >>> reverse = Reverse([True])
    >>> model.add_scale_tranformation(reverse)
    >>> model.plot.plot_item_probabilities(1).show()
    """
    def __init__(self, reversed_latent_variables: list[bool]):
        super().__init__(invertible=True)
        self._reverse = -torch.tensor(reversed_latent_variables, dtype=int).reshape(1, -1)

    def transform(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Transforms the input theta scores into the new scale.

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing latent variable theta scores. Each column represents one latent variable.
        """
        
        return theta * self._reverse

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
        return transformed_theta * self._reverse

    def jacobian(
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
            A torch tensor with the gradients for each theta score. Dimensions are (theta rows, latent variables, latent variables) where the last two are the jacobians.
        """
        theta_scores = theta.clone()
        theta_scores.requires_grad_(True)
        theta_scores = self.transform(theta_scores)
        theta_scores.sum().backward()
        jacobians = torch.diag_embed(theta_scores.grad) # Since each transformation is only dependent on one theta score
        return jacobians
