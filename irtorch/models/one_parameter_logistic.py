import pandas as pd
import torch
import torch.nn as nn
from irtorch.models.base_irt_model import BaseIRTModel

class OneParameterLogistic(BaseIRTModel):
    """
    One parametric logistic (2PL) IRT model.

    Parameters
    ----------
    latent_variables : int
        Number of latent variables.
    items : int
        Number of items.
    item_z_relationships : torch.Tensor, optional
        A boolean tensor of shape (items, latent_variables). If specified, the model will have connections between latent dimensions and items where the tensor is True. If left out, all latent variables and items are related (Default: None)
    """
    def __init__(
        self,
        latent_variables: int,
        items: int,
        item_z_relationships: torch.Tensor = None
    ):
        super().__init__(latent_variables=latent_variables, item_categories = [2] * items)
        if item_z_relationships is not None:
            if item_z_relationships.shape != (items, latent_variables):
                raise ValueError(
                    f"latent_item_connections must have shape ({items}, {latent_variables})."
                )
            assert(item_z_relationships.dtype == torch.bool), "latent_item_connections must be boolean type."
            assert(torch.all(item_z_relationships.sum(dim=1) > 0)), "all items must have a relationship with a least one latent variable."

        self.output_size = self.items * 2
        self.weight_param = nn.Parameter(torch.zeros(latent_variables))
        self.bias_param = nn.Parameter(torch.zeros(self.items))

        first_category = torch.zeros(self.items, 2)
        first_category[:, 0] = 1.0
        first_category = first_category.reshape(-1)

        free_bias = 1 - first_category
        self.register_buffer("free_weights", torch.ones(latent_variables))
        self.register_buffer("free_bias", free_bias.bool())
        self.register_buffer("first_category", first_category.bool())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight_param, mean=1., std=0.01)
        nn.init.zeros_(self.bias_param)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        z : torch.Tensor
            2D tensor with latent variables. Rows are respondents and latent variables are columns. 

        Returns
        -------
        output : torch.Tensor
            2D tensor. Rows are respondents and columns are item category logits.
        """
        bias = torch.zeros(self.output_size, device=z.device)
        bias[self.free_bias] = self.bias_param
        weighted_z = (self.weight_param * z).sum(dim=1, keepdim=True)

        output = weighted_z + bias
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
        weights = self.weight_param.repeat(self.items, 1)

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
    def item_z_relationship_directions(self, z:torch.Tensor = None) -> torch.Tensor:
        """
        Get the relationship directions between each item and latent variable for a fitted model.

        Parameters
        ----------
        z : torch.Tensor, optional
            Not needed for this model. (default is None)
            
        Returns
        -------
        torch.Tensor
            A 2D tensor with the relationships between the items and latent variables. Items are rows and latent variables are columns.
        """
        return self.weight_param.repeat(self.items, 1).sign().int()
    