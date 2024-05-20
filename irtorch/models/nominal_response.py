import pandas as pd
import torch
import torch.nn as nn
from irtorch.models.base_irt_model import BaseIRTModel

class NominalResponse(BaseIRTModel):
    """
    Nominal response IRT model.

    Parameters
    ----------
    latent_variables : int, optional
        Number of latent variables. (default is 1)
    data: torch.Tensor, optional
        A 2D torch tensor with test data. Used to automatically compute item_categories. Columns are items and rows are respondents. (default is None)
    item_categories : list[int], optional
        Number of categories for each item. One integer for each item. Missing responses exluded. (default is None)
    item_z_relationships : torch.Tensor, optional
        A boolean tensor of shape (items, latent_variables). If specified, the model will have connections between latent dimensions and items where the tensor is True. If left out, all latent variables and items are related (Default: None)
    model_missing : bool, optional
        Whether to model missing item responses as separate item response categories. (Default: False)
    mc_correct : list[int], optional
        The correct response category for each item. If specified, the logits for the correct responses are cumulative logits. (Default: None)
    reference_category : bool, optional
        Whether to use the first category as an unparameterized reference category. (Default: False and uses the original parameterization given by Bock(1972))
    """
    def __init__(
        self,
        latent_variables: int = 1,
        data: torch.Tensor = None,
        item_categories: list[int] = None,
        item_z_relationships: torch.Tensor = None,
        model_missing: bool = False,
        mc_correct: list[int] = None,
        reference_category: bool = False
    ):
        if item_categories is None:
            if data is None:
                raise ValueError("Either an instantiated model, item_categories or data must be provided to initialize the model.")
            else:
                # replace nan with -inf to get max
                item_categories = (torch.where(~data.isnan(), data, torch.tensor(float('-inf'))).max(dim=0).values + 1).int().tolist()

        super().__init__(latent_variables=latent_variables, item_categories=item_categories, model_missing=model_missing, mc_correct=mc_correct)
        if item_z_relationships is not None:
            if item_z_relationships.shape != (len(item_categories), latent_variables):
                raise ValueError(
                    f"latent_item_connections must have shape ({len(item_categories)}, {latent_variables})."
                )
            assert(item_z_relationships.dtype == torch.bool), "latent_item_connections must be boolean type."
            assert(torch.all(item_z_relationships.sum(dim=1) > 0)), "all items must have a relationship with a least one latent variable."

        self.output_size = self.items * self.max_item_responses

        free_weights = torch.zeros(self.items, self.max_item_responses, latent_variables)
        for item, item_cat in enumerate(self.modeled_item_responses):
            start_1 = 1 if reference_category else 0
            if item_z_relationships is not None:
                free_weights[item, start_1:item_cat, :] = item_z_relationships[item, :]
            else:
                free_weights[item, start_1:item_cat, :] = 1.0

        free_weights = free_weights.reshape(-1, latent_variables)
        self.weight_param = nn.Parameter(torch.zeros(free_weights.sum().int()))

        number_of_bias_parameters = sum(self.modeled_item_responses) if not reference_category else sum(self.modeled_item_responses) - self.items
        self.bias_param = nn.Parameter(torch.zeros(number_of_bias_parameters))
        first_category = torch.zeros(self.items, self.max_item_responses)
        if reference_category:
            first_category[:, 0] = 1.0
        first_category = first_category.reshape(-1)
        missing_category = torch.zeros(self.items, self.max_item_responses)
        for item, item_cat in enumerate(self.modeled_item_responses):
            missing_category[item, item_cat:self.max_item_responses] = 1.0
        missing_category = missing_category.reshape(-1)
        free_bias = (1 - first_category) * (1 - missing_category)
        self.register_buffer("free_weights", free_weights.bool())
        self.register_buffer("free_bias", free_bias.bool())
        self.register_buffer("missing_category", missing_category.bool())
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
        
        weights = torch.zeros(self.output_size, self.latent_variables, device=z.device)
        weights[self.free_weights] = self.weight_param
        weighted_z = torch.matmul(z, weights.T)

        output = weighted_z + bias
        # stop gradients from flowing through the missing categories
        output[:, self.missing_category] = -torch.inf

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
        biases = torch.zeros(self.output_size)
        biases[self.free_bias] = self.bias_param
        biases = biases.reshape(-1, self.max_item_responses)

        weights = torch.zeros(self.output_size, self.latent_variables)
        weights[self.free_weights] = self.weight_param
        weights = weights.reshape(-1, self.max_item_responses * self.latent_variables)

        weights_df = pd.DataFrame(weights.detach().numpy())
        if irt_format and self.latent_variables == 1:
            biases_df = pd.DataFrame(-(weights*biases).detach().numpy())
            weights_df.columns = [f"a{i+1}{j+1}" for i in range(self.latent_variables) for j in range(int(weights.shape[1]/self.latent_variables))]
        else:
            biases_df = pd.DataFrame(biases.detach().numpy())
            weights_df.columns = [f"w{i+1}{j+1}" for i in range(self.latent_variables) for j in range(int(weights.shape[1]/self.latent_variables))]
            
        biases_df.columns = [f"b{i+1}" for i in range(biases_df.shape[1])]
        parameters = pd.concat([weights_df, biases_df], axis=1)

        return parameters

    @torch.inference_mode()
    def item_z_relationship_directions(self, z:torch.Tensor = None) -> torch.Tensor:
        """
        Get the relationship directions between each item and latent variable for a fitted model.

        Parameters
        ----------
        z : torch.Tensor, optional
            Only needed for NR models. A 2D tensor with latent z scores from the population of interest. Each row represents one respondent, and each column represents a latent variable.

        Returns
        -------
        torch.Tensor
            A 2D tensor with the relationships between the items and latent variables. Items are rows and latent variables are columns.
        """
        if z is None:
            raise ValueError("z must be provided to get item to z relationships for NR models.")
        return super().item_z_relationship_directions(z)
