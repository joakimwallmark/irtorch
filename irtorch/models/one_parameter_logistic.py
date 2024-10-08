import pandas as pd
import torch
import torch.nn as nn
from irtorch.models.base_irt_model import BaseIRTModel

class OneParameterLogistic(BaseIRTModel):
    r"""
    One parametric logistic (1PL) IRT model, also known as the Rasch model :cite:p:`Rasch1960`. This model is only available for unidimensional latent variables.

    Parameters
    ----------
    items : int
        Number of items.
    
    Notes
    -----
    For an item :math:`j`, the model defines the probability for responding correctly as:

    .. math::

        \frac{
            \exp(\theta + d_j)
        }{
            1+\exp(\theta + d_j)
        },

    where:

    - :math:`\theta` is the latent variable.
    - :math:`d_j` is the bias term for item :math:`j`.

    Examples
    --------
    >>> from irtorch.models import OneParameterLogistic
    >>> from irtorch.estimation_algorithms import AE
    >>> from irtorch.load_dataset import swedish_sat_binary
    >>> # Use quantitative part of the SAT data
    >>> data = swedish_sat_binary()[:, :80]
    >>> model = OneParameterLogistic(items=80)
    >>> model.fit(train_data=data, algorithm=AE())
    """
    def __init__(
        self,
        items: int,
    ):
        super().__init__(latent_variables=1, item_categories = [2] * items)

        self.output_size = self.items * 2
        self.bias_param = nn.Parameter(torch.zeros(self.items))

        first_category = torch.zeros(self.items, 2)
        first_category[:, 0] = 1.0
        first_category = first_category.reshape(-1)

        free_bias = 1 - first_category
        self.register_buffer("free_bias", free_bias.bool())
        self.register_buffer("first_category", first_category.bool())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.bias_param)
    
    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        theta : torch.Tensor
            2D tensor with latent variables. Rows are respondents and latent variables are columns. 

        Returns
        -------
        output : torch.Tensor
            2D tensor. Rows are respondents and columns are item category logits.
        """
        bias = torch.zeros(self.output_size, device=theta.device)
        bias[self.free_bias] = self.bias_param
        output = theta + bias
        output[:, self.first_category] = 0

        return output

    def item_parameters(self) -> pd.DataFrame:
        """
        Get the item parameters for a fitted model.

        Returns
        -------
        pd.DataFrame
            A dataframe with the item parameters.
        """
        biases = self.bias_param
        biases_df = pd.DataFrame(biases.reshape(-1, 1).detach().numpy())
        biases_df.columns = ["d"]

        return biases_df

    @torch.no_grad()
    def item_theta_relationship_directions(self, theta:torch.Tensor = None) -> torch.Tensor:
        """
        Get the relationship directions between each item and latent variable for a fitted model.

        Parameters
        ----------
        theta : torch.Tensor, optional
            Not needed for this model. (default is None)
            
        Returns
        -------
        torch.Tensor
            A 2D tensor with the relationships between the items and latent variables. Items are rows and latent variables are columns.
        """
        return torch.ones(self.items, 1)
    