import pandas as pd
import torch
import torch.nn as nn
from irtorch.models.base_irt_model import BaseIRTModel

class ThreeParameterLogistic(BaseIRTModel):
    r"""
    Three parametric logistic (3PL) IRT model :cite:p:`Birnbaum1968`. *Note that parameter estimation for this model is currently very unstable.*

    Parameters
    ----------
    data: torch.Tensor, optional
        A 2D torch tensor with test data. Used to automatically compute the number of items. Columns are items and rows are respondents. (default is None)
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
    >>> model = ThreeParameterLogistic(items=80)
    >>> model.fit(train_data=data, algorithm=AE())
    """
    def __init__(
        self,
        data: torch.Tensor = None,
        latent_variables: int = 1,
        items: int = None,
        item_theta_relationships: torch.Tensor = None
    ):
        if items is None and data is None:
            raise ValueError("Either items or data must be provided to initialize the model.")
        if data is not None:
            items = data.size(1)

        super().__init__(latent_variables=latent_variables, item_categories = [2] * items)
        if item_theta_relationships is not None:
            if item_theta_relationships.shape != (items, latent_variables):
                raise ValueError(
                    f"latent_item_connections must have shape ({items}, {latent_variables})."
                )
            assert(item_theta_relationships.dtype == torch.bool), "latent_item_connections must be boolean type."
            assert(torch.all(item_theta_relationships.sum(dim=1) > 0)), "all items must have a relationship with a least one latent variable."
        else:
            item_theta_relationships = torch.ones(items, latent_variables, dtype=torch.bool)

        self.output_size = self.items * 2
        self.guessing_param = nn.Parameter(torch.full((self.items,), -1.1))
        self.weight_param = nn.Parameter(torch.zeros(item_theta_relationships.sum().int()))
        self.bias_param = nn.Parameter(torch.zeros(self.items))

        self.register_buffer("free_weights", item_theta_relationships.bool())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight_param)
        nn.init.zeros_(self.bias_param)
        nn.init.constant_(self.guessing_param, -1.1)

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
            2D tensor. Rows are respondents and columns are probabilities.
        """
        bias = self.bias_param

        weights = torch.zeros(self.items, self.latent_variables, device=theta.device)
        weights[self.free_weights] = self.weight_param
        weighted_theta = torch.matmul(theta, weights.T)
        probs = (weighted_theta + bias).sigmoid()
        # Apply sigmoid to guessing_param to keep it in range (0, 1)
        guessing_param = self.guessing_param.sigmoid()
        probs = probs * (1 - guessing_param) + guessing_param

        return probs

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
        # output is the probability of getting the item correct
        probs = torch.stack([1 - output, output], dim=2)
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
        reshaped_probabilities = probabilities.reshape(-1, 2)

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
        if irt_format and self.latent_variables > 1:
            raise ValueError("IRT format is only available for unidimensional models.")

        biases = self.bias_param
        biases = biases.reshape(-1, 1)
        weights = torch.zeros(self.items, self.latent_variables)
        weights[self.free_weights] = self.weight_param

        weights_df = pd.DataFrame(weights.detach().numpy())
        weights_df.columns = [f"a{i+1}" for i in range(weights.shape[1])]
        if irt_format:
            biases_df = pd.DataFrame(-(weights*biases).detach().numpy())
            biases_df.columns = ["b"]
        else:
            biases_df = pd.DataFrame(biases.detach().numpy())
            biases_df.columns = ["d"]

        guessing_df = pd.DataFrame(self.guessing_param.sigmoid().detach().numpy(), columns=["c"])
        parameters = pd.concat([weights_df, biases_df, guessing_df], axis=1)

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
