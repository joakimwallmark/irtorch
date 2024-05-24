import pandas as pd
import torch
import torch.nn as nn
from irtorch.models.base_irt_model import BaseIRTModel

class TwoParameterLogistic(BaseIRTModel):
    r"""
    Two parametric logistic (2PL) IRT model.

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

        \frac{
            \exp(\mathbf{a}_j^\top \mathbf{\theta} + d_j)
        }{
            1+\exp(\mathbf{a}_j^\top \mathbf{\theta} + d_j)
        },

    where:

    - :math:`\mathbf{\theta}` is a vector of latent variables.
    - :math:`\mathbf{a}_j` is a vector of weights for item :math:`j`.
    - :math:`d_j` is the bias term for item :math:`j`.
    """
    def __init__(
        self,
        latent_variables: int,
        items: int,
        item_theta_relationships: torch.Tensor = None
    ):
        super().__init__(latent_variables=latent_variables, item_categories = [2] * items)
        if item_theta_relationships is not None:
            if item_theta_relationships.shape != (items, latent_variables):
                raise ValueError(
                    f"latent_item_connections must have shape ({items}, {latent_variables})."
                )
            assert(item_theta_relationships.dtype == torch.bool), "latent_item_connections must be boolean type."
            assert(torch.all(item_theta_relationships.sum(dim=1) > 0)), "all items must have a relationship with a least one latent variable."

        self.output_size = self.items * 2
        self.weight_param = nn.Parameter(torch.zeros(item_theta_relationships.sum().int()))
        self.bias_param = nn.Parameter(torch.zeros(self.items))

        first_category = torch.zeros(self.items, 2)
        first_category[:, 0] = 1.0
        first_category = first_category.reshape(-1)

        free_bias = 1 - first_category
        self.register_buffer("free_weights", item_theta_relationships.bool())
        self.register_buffer("free_bias", free_bias.bool())
        self.register_buffer("first_category", first_category.bool())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight_param, mean=1., std=0.01)
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
        
        weights = torch.zeros(self.items, self.latent_variables, device=theta.device)
        weights[self.free_weights] = self.weight_param
        weighted_theta = torch.matmul(theta, weights.T).repeat_interleave(self.max_item_responses, dim=1)

        output = weighted_theta + bias
        output[:, self.first_category] = 0

        return output

    def item_parameters(self, irt_format = False) -> pd.DataFrame:
        """
        Get the item parameters for a fitted model.

        Parameters
        ----------
        irt_format : bool, optional
            Only for unidimensional models. Whether to return the item parameters in traditional IRT format. Otherwise returns weights and biases. (default is False)

        Returns
        -------
        pd.DataFrame
            A dataframe with the item parameters.
        """
        if irt_format and self.latent_variables > 1:
            raise ValueError("IRT format is only available for unidimensional models.")
        
        biases = self.bias_param
        biases = biases.reshape(-1, 1)
        weights = torch.zeros(self.items, self.latent_variables)
        weights[self.free_weights] = self.weight_param

        weights_df = pd.DataFrame(weights.detach().numpy())
        if irt_format:
            weights_df.columns = [f"a{i+1}" for i in range(weights.shape[1])]
            biases_df = pd.DataFrame(-(weights*biases).detach().numpy())
        else:
            weights_df.columns = [f"w{i+1}" for i in range(weights.shape[1])]
            biases_df = pd.DataFrame(biases.detach().numpy())
            
        biases_df.columns = ["b1"]
        parameters = pd.concat([weights_df, biases_df], axis=1)

        return parameters

    @torch.inference_mode()
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
        weights = torch.zeros(self.items, self.latent_variables)
        weights[self.free_weights] = self.weight_param
        return weights.sign().int()
    