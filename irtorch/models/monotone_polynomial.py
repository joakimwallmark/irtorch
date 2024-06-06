import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from irtorch.models.base_irt_model import BaseIRTModel
from irtorch.torch_modules import MonotonePolynomialModule, NegationLayer
from irtorch.activation_functions import BoundedELU

logger = logging.getLogger("irtorch")

class MonotonePolynomial(BaseIRTModel):
    r"""
    Monotonic Polynomial IRT model.
    The model is a feedforward neural network separate monotonic functions for each item or item category.

    If mc_correct is specified, the latent variable effect for the correct item response is a cumulative sum of the effects for the other possible item responses to ensure monotonicity. This model is also referred to as the Monotone Multiple Choice (MMC) model.
    
    If mc_correct is not specified, the item response categories are treated as ordered, and the latent variable effect for an item response category is the cumulative sum of the effects for the lower categories.

    Parameters
    ----------
    latent_variables : int, optional
        Number of latent variables. (default is 1)
    data: torch.Tensor, optional
        A 2D torch tensor with test data. Used to automatically compute item_categories. Columns are items and rows are respondents. (default is None)
    item_categories : list[int], optional
        Number of categories for each item. One integer for each item. Missing responses exluded. (default is None)
    degree : int, optional
        The degree of the monotonic polynomials. (default is 3)
    model_missing : bool, optional
        Whether to model missing item responses as separate item response categories. (Default: False)
    mc_correct : list[int], optional
        For multiple choice tests. The correct response category for each item. (Default: None)
    separate : str, optional
        Whether to fit separate latent trait functions for items or items categories. Can be 'items' or 'categories'. 
        Note that 'categories' results in a more flexible model with more parameters. (Default: 'categories')
    item_theta_relationships : torch.Tensor, optional
        A boolean tensor of shape (items, latent_variables). If specified, the model will have connections between latent dimensions and items where the tensor is True. If left out, all latent variables and items are related (Default: None)
    negative_latent_variable_item_relationships : bool, optional
        Whether to allow for negative latent variable item relationships. (Default: True)

    """
    def __init__(
        self,
        latent_variables: int = 1,
        data: torch.Tensor = None,
        item_categories: list[int] = None,
        degree: int = 3,
        model_missing: bool = False,
        mc_correct: list[int] = None,
        item_theta_relationships: torch.Tensor = None,
        separate: str = "categories",
        negative_latent_variable_item_relationships: bool = True,
    ):
        if item_categories is None:
            if data is None:
                raise ValueError("Either an instantiated model, item_categories or data must be provided to initialize the model.")
            else:
                # replace nan with -inf to get max
                item_categories = (torch.where(~data.isnan(), data, torch.tensor(float('-inf'))).max(dim=0).values + 1).int().tolist()
                
        super().__init__(latent_variables, item_categories, mc_correct, model_missing)
        if item_theta_relationships is not None:
            if item_theta_relationships.shape != (len(item_categories), latent_variables):
                raise ValueError(
                    f"latent_item_connections must have shape ({len(item_categories)}, {latent_variables})."
                )
            assert(item_theta_relationships.dtype == torch.bool), "latent_item_connections must be boolean type."
            assert(torch.all(item_theta_relationships.sum(dim=1) > 0)), "all items must have a relationship with a least one latent variable."
        else:
            item_theta_relationships = torch.tensor([[True] * latent_variables] * self.items, dtype=torch.bool)

        self.separate = separate
        self.output_length = self.items * self.max_item_responses
        self.negative_latent_variable_item_relationships = negative_latent_variable_item_relationships
        if separate == "items":
            self.separations = self.items
        if separate == "categories":
            self.separations = self.output_length

        self.mc_correct_output_idx = None
        if mc_correct is not None:
            item_start_positions = torch.arange(0, self.output_length, self.max_item_responses)
            indices = (item_start_positions + torch.tensor(mc_correct) - 1 + self.model_missing).long()
            self.mc_correct_output_idx = torch.zeros(self.output_length, dtype=torch.bool)
            self.mc_correct_output_idx.index_fill_(0, indices, True)

        missing_categories = torch.zeros(self.items, self.max_item_responses, dtype=torch.int)
        for item, item_cat in enumerate(self.modeled_item_responses):
            missing_categories[item, item_cat:self.max_item_responses] = 1
        missing_categories = missing_categories.reshape(-1)
        self.register_buffer("missing_categories", missing_categories.bool())
        self.register_buffer("free_bias", (1 - missing_categories).bool())
        self.bias_param = nn.Parameter(torch.zeros(sum(self.modeled_item_responses)))

        shared_directions = self.max_item_responses if separate == "categories" else 1
        self.add_module("mono_poly", MonotonePolynomialModule(
            degree=degree,
            in_features=latent_variables,
            out_features=self.separations,
            intercept=False,
            relationship_matrix=item_theta_relationships.transpose(0, 1),
            negative_relationships=negative_latent_variable_item_relationships,
            shared_directions=shared_directions
        ))

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        out = self._modules["mono_poly"](theta)
        if self.separate == "items":
            out = out.repeat_interleave(self.max_item_responses, dim=1)

        if self.mc_correct_output_idx is None:
            out = out.view(out.shape[0], -1, self.max_item_responses).cumsum(dim=2).reshape(out.shape[0], -1)
        else:
            out[:, self.mc_correct_output_idx] = out.view(out.shape[0], -1, self.max_item_responses).sum(dim=2)

        bias = torch.zeros(self.output_length, device=theta.device)
        bias[self.free_bias] = self.bias_param
        out += bias
        out[:, self.missing_categories] = -torch.inf
        return out

    def probabilities_from_output(self, output: torch.Tensor) -> torch.Tensor:
        """
        Compute item probabilities from the output tensor.

        Parameters
        ----------
        output : torch.Tensor
            Output tensor from the forward pass.

        Returns
        -------
        torch.Tensor
            3D torch tensor with dimensions (respondents, items, item categories).
        """
        reshaped_output = output.reshape(-1, self.max_item_responses)
        return F.softmax(reshaped_output, dim=1).reshape(output.shape[0], self.items, self.max_item_responses)


    def item_theta_relationship_directions(self, *args) -> torch.Tensor:
        """
        Get the relationship directions between each item and latent variable for a fitted model.

        Returns
        -------
        torch.Tensor
            A 2D tensor with the relationships between the items and latent variables. Items are rows and latent variables are columns.
        """
        if self.negative_latent_variable_item_relationships:
            return self._modules["mono_poly"].directions.transpose(0, 1).sign().int()
        return torch.ones(self.items, self.latent_variables).int()
