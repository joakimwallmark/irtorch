import pandas as pd
import torch
import torch.nn as nn
from irtorch.models.base_irt_model import BaseIRTModel

class ThreeParameterLogistic(BaseIRTModel):
    r"""
    Three parametric logistic (3PL) IRT model :cite:p:`Birnbaum1968`.

    Parameters
    ----------
    latent_variables : int
        Number of latent variables.
    items : int
        Number of items.
    item_theta_relationships : torch.Tensor, optional
        A boolean tensor of shape (items, latent_variables). If specified, the model will have connections between latent dimensions and items where the tensor is True. If left out, all latent variables and items are related (Default: None)
    
    Notes
    -----
    For an item :math:`j`, the model defines the probability for responding correctly as:

    .. math::

        c_j+(1-c_j)
        \frac{
            \exp(\mathbf{a}_j^\top \mathbf{\theta} + d_j)
        }{
            1+\exp(\mathbf{a}_j^\top \mathbf{\theta} + d_j)
        },

    where:

    - :math:`\mathbf{\theta}` is a vector of latent variables.
    - :math:`\mathbf{a}_j` is a vector of weights for item :math:`j`.
    - :math:`d_j` is the bias term for item :math:`j`.
    - :math:`c_j` is the probably of guessing correctly on item :math:`j`.

    Examples
    --------
    >>> from irtorch.models import ThreeParameterLogistic
    >>> from irtorch.estimation_algorithms import AE
    >>> from irtorch.load_dataset import swedish_sat_binary
    >>> # Use quantitative part of the SAT data
    >>> data = swedish_sat_binary()[:, :80]
    >>> model = ThreeParameterLogistic(1, items=80)
    >>> model.fit(train_data=data, algorithm=AE())
    """
    def __init__(
        self,
        latent_variables: int,
        items: int,
        item_theta_relationships: torch.Tensor = None
    ):
        super().__init__(latent_variables=latent_variables, item_categories = [2] * items)
        raise NotImplementedError("This model is not yet implemented.")
