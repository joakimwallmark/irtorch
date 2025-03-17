import pandas as pd
import torch
import torch.nn as nn
from irtorch.models.base_irt_model import BaseIRTModel

class GradedResponse(BaseIRTModel):
    r"""
    Graded response IRT model :cite:p:`Samejima1968`.

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
    For an item :math:`j` with ordered item scores :math:`x=0, 1, 2, ...` the model defines the probability for responding with a score of :math:`x` or higher for all :math:`x>0` as follows:

    .. math::

        P(X_j \geq x | \mathbf{\theta}) = \dfrac{\exp \left(\mathbf{a}_{j}^\top \mathbf{\theta} + d_{jx}\right)}{1+\exp \left(\mathbf{a}_{j}^\top \mathbf{\theta} + d_{jx}\right)}

            
    where:

    - :math:`\mathbf{\theta}` is a vector of latent variables.
    - :math:`\mathbf{a}_{j}` is a vector of weights for item :math:`j`.
    - :math:`d_{jx}` is the bias term for item :math:`j` and score :math:`x`.

    From here, the probability of responding with a score of :math:`x` is calculated as:

    .. math::
        P(X_j = x | \mathbf{\theta}) = \begin{cases}
            1-P(X_j \geq x +1 | \mathbf{\theta}), & \text{if } x = 0\\
            P(X_j \geq x | \mathbf{\theta})-P(X_j \geq x+1 | \mathbf{\theta}), & \text{otherwise}
        \end{cases}

    Examples
    --------
    >>> from irtorch.models import GradedResponse
    >>> from irtorch.estimation_algorithms import JML
    >>> from irtorch.load_dataset import swedish_national_mathematics_1
    >>> data = swedish_national_mathematics_1()
    >>> model = GradedResponse(data)
    >>> model.fit(train_data=data, algorithm=JML())
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
        if item_theta_relationships is not None:
            for item, item_cat in enumerate(self.item_categories):
                free_weights[item, :] = item_theta_relationships[item, :]

        self.weight_param = nn.Parameter(torch.zeros(free_weights.sum().int()))
        
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
        # Xavier uniform initialization for weights https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
        a = torch.sqrt(torch.tensor(6 / (self.latent_variables + self.items)))
        nn.init.uniform_(self.weight_param, a=-a, b=a)
        # initialize bias parameters
        bias = torch.zeros_like(self.free_bias, dtype=torch.float)
        for item in range(self.items):
            item_bias = torch.zeros(self.max_item_responses, dtype=torch.float)
            item_ind = range(item * self.max_item_responses, item * self.max_item_responses + self.max_item_responses)
            probabilities = torch.linspace(
                1/(1+torch.exp(torch.tensor(4))),
                1/(1+torch.exp(torch.tensor(-4))),
                self.free_bias[item_ind].sum() + 2
            )[1:-1]
            item_bias[self.free_bias[item_ind]] = probabilities.pow(-1).add(-1).log()
            bias[item_ind] = item_bias

        self.bias_param = nn.Parameter(bias[self.free_bias])

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass of the model.

        Parameters
        ----------
        theta : torch.Tensor
            2D tensor with latent variables. Rows are respondents and latent variables are columns. 

        Returns
        -------
        output : torch.Tensor
            2D tensor. Rows are respondents and columns are :math:`\mathbf{a}_{j}^\top \mathbf{\theta} + d_{jx}`.
        """
        bias = torch.zeros(self.output_size, device=theta.device)
        bias[self.free_bias] = self.bias_param
        
        weights = torch.zeros(self.items, self.latent_variables, device=theta.device)
        weights[self.free_weights] = self.weight_param
        weighted_theta = torch.matmul(theta, weights.T).repeat_interleave(self.max_item_responses, dim=1)

        output = weighted_theta + bias
        output[:, self.missing_category] = -torch.inf
        output[:, self.first_category] = torch.inf

        return output

    def probabilities_from_output(self, output: torch.Tensor) -> torch.Tensor:
        """
        Compute probabilities from the output tensor from the forward method.

        Parameters
        ----------
        output : torch.Tensor
            Output tensor from the forward pass.

        Returns
        -------
        torch.Tensor
            3D tensor with dimensions (respondents, items, item categories)
        """
        # great than or equal to probabilities as per graded response model
        geq_probs = output.sigmoid().reshape(output.shape[0], self.items, self.max_item_responses)
        # convert to probabilities for each score
        probs = geq_probs.clone()  # Create a copy of `probs` to avoid modifying the original in place
        probs[:, :, :-1] -= geq_probs[:, :, 1:]
        return probs

    def log_likelihood(
        self,
        data: torch.Tensor,
        output: torch.Tensor,
        missing_mask: torch.Tensor = None,
        loss_reduction: str = "sum",
    ) -> torch.Tensor:
        """
        Compute the log likelihood.

        Parameters
        ----------
        data: torch.Tensor
            A 2D tensor with test data. Columns are items and rows are respondents
        output: torch.Tensor
            A 2D tensor with output. Columns are item response categories and rows are respondents
        missing_mask: torch.Tensor, optional
            A 2D tensor with missing data mask. (default is None)
        loss_reduction: str, optional 
            The reduction argument. (default is 'sum')
        
        Returns
        -------
        torch.Tensor
            The log likelihood.
        """
        probabilities = self.probabilities_from_output(output)
        data = data.long().view(-1)
        reshaped_probabilities = probabilities.reshape(-1, self.max_item_responses)

        if missing_mask is not None:
            missing_mask = missing_mask.view(-1)
            reshaped_probabilities = reshaped_probabilities[~missing_mask]
            respondents = data.size(0)
            data = data[~missing_mask]

        ll = reshaped_probabilities[torch.arange(data.size(0)), data].log()
        if loss_reduction == "sum":
            return ll.sum()
        elif loss_reduction == "none" and missing_mask is not None:
            ll_masked = torch.full((respondents, ), torch.nan, device= ll.device)
            ll_masked[~missing_mask] = ll
            return ll_masked
        else:
            raise ValueError("loss_reduction must be 'sum' or 'none'")

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
            biases_df = pd.DataFrame(biases.detach()[:, 1:].numpy())
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
    