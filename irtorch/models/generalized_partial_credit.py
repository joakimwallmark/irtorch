import pandas as pd
import torch
import torch.nn as nn
from irtorch.models.base_irt_model import BaseIRTModel

class GeneralizedPartialCredit(BaseIRTModel):
    r"""
    Generalized Partial Credit IRT model :cite:p:`Muraki1992`.

    Parameters
    ----------
    data: torch.Tensor, optional
        A 2D torch tensor with test data. Used to automatically compute item_categories. Columns are items and rows are respondents. (default is None)
    latent_variables : int
        Number of latent variables.
    item_categories : list[int]
        Number of categories for each item. One integer for each item. Missing responses exluded.
    item_theta_relationships : torch.Tensor, optional
        A boolean tensor of shape (items, latent_variables). If specified, the model will have connections between latent dimensions and items where the tensor is True. If left out, all latent variables and items are related (Default: None)

    Notes
    -----
    For an item :math:`j` with :math:`m=0, 1, 2, \ldots, M_j` possible item scores, the model defines the probability for responding with a score of :math:`x` as follows:

    .. math::

        P(X_j=x | \mathbf{\theta}) = \begin{cases}
            \dfrac{1}
            {1+\sum_{g=1}^{M_i}\exp \left(g\mathbf{a}_{j}^\top \mathbf{\theta} + \sum_{m=1}^gd_{jm}\right)}, & \text{if } x = 0\\
            \dfrac{\exp \left( x\mathbf{a}_{j}^\top \mathbf{\theta}+\sum_{m=1}^{x}d_{jm}\right)}
            {1+\sum_{g=1}^{M_i}\exp \left(g\mathbf{a}_{j}^\top \mathbf{\theta} + \sum_{m=1}^gd_{jm}\right)}, & \text{otherwise}
        \end{cases}

    where:

    - :math:`\mathbf{\theta}` is a vector of latent variables.
    - :math:`\mathbf{a}_{j}` is a vector of weights for item :math:`j`.
    - :math:`d_{jm}` is the bias term for item :math:`j` and score :math:`m`.

    Examples
    --------
    >>> from irtorch.models import GeneralizedPartialCredit
    >>> from irtorch.estimation_algorithms import AE
    >>> from irtorch.load_dataset import swedish_national_mathematics_1
    >>> data = swedish_national_mathematics_1()
    >>> model = GeneralizedPartialCredit(data)
    >>> model.fit(train_data=data, algorithm=AE())
    """
    def __init__(
        self,
        data: torch.Tensor = None,
        latent_variables: int = 1,
        item_categories: list[int] = None,
        item_theta_relationships: torch.Tensor = None,
    ):
        if item_categories is None:
            if data is None:
                raise ValueError("Either item_categories or data must be provided to initialize the model.")
            else:
                # replace nan with -inf to get max
                item_categories = (torch.where(~data.isnan(), data, torch.tensor(float('-inf'))).max(dim=0).values + 1).int().tolist()

        super().__init__(latent_variables=latent_variables, item_categories=item_categories)
        if item_theta_relationships is not None:
            if item_theta_relationships.shape != (len(item_categories), latent_variables):
                raise ValueError(
                    f"latent_item_connections must have shape ({len(item_categories)}, {latent_variables})."
                )
            assert(item_theta_relationships.dtype == torch.bool), "latent_item_connections must be boolean type."
            assert(torch.all(item_theta_relationships.sum(dim=1) > 0)), "all items must have a relationship with a least one latent variable."

        self.output_size = self.items * self.max_item_responses

        free_weights = torch.ones(self.items, latent_variables)
        self.register_buffer("gpc_weight_multiplier", torch.arange(0, self.max_item_responses).repeat(self.items))
        if item_theta_relationships is not None:
            for item, item_cat in enumerate(self.item_categories):
                free_weights[item, :] = item_theta_relationships[item, :]

        self.weight_param = nn.Parameter(torch.zeros(free_weights.sum().int()))

        number_of_bias_parameters = sum(self.item_categories) - self.items
        self.bias_param = nn.Parameter(torch.zeros(number_of_bias_parameters))
        first_category = torch.zeros(self.items, self.max_item_responses)
        first_category[:, 0] = 1.0
        first_category = first_category.reshape(-1)
        missing_category = torch.zeros(self.items, self.max_item_responses)
        for item, item_cat in enumerate(self.item_categories):
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
            2D tensor. Rows are respondents and columns are item score logits.
        """
        bias = torch.zeros(self.output_size, device=theta.device)
        bias[self.free_bias] = self.bias_param
        
        weights = torch.zeros(self.items, self.latent_variables, device=theta.device)
        weights[self.free_weights] = self.weight_param
        weighted_theta = torch.matmul(theta, weights.T).repeat_interleave(self.max_item_responses, dim=1)
        weighted_theta *= self.gpc_weight_multiplier

        output = weighted_theta + bias
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

        weights = torch.zeros(self.items, self.latent_variables)
        weights[self.free_weights] = self.weight_param

        weights_df = pd.DataFrame(weights.detach().numpy())
        weights_df.columns = [f"a{i+1}" for i in range(weights.shape[1])]
        if irt_format and self.latent_variables == 1:
            biases_df = pd.DataFrame(-(biases/weights).detach()[:, 1:].numpy())
            biases_df.columns = [f"b{i+1}" for i in range(biases_df.shape[1])]
        else:
            biases_df = pd.DataFrame(biases.detach().numpy())
            biases_df.columns = [f"d{i+1}" for i in range(biases_df.shape[1])]
            
        parameters = pd.concat([weights_df, biases_df], axis=1)

        return parameters

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
        weights = torch.zeros(self.items, self.latent_variables)
        weights[self.free_weights] = self.weight_param
        return weights.sign().int()
    