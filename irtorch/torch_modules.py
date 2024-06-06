import logging
import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger("irtorch")

class MonotonePolynomialModule(nn.Module):
    """
    A polynomial with monotonicity constraints.

    Parameters
    ----------
    degree: int
        Degree of the polynomial. Needs to be an uneven number.
    in_features: int
        Number of input features.
    out_features: int
        Number of output features.
    intercept: bool
        Whether to include an intercept term. (Default: False)
    relationship_matrix : torch.Tensor, optional
        A boolean tensor of shape (in_features, out_features,) that determines which inputs are related to which outputs. Typically used for IRT models to remove relationships between some items or item cateogires and latent traits. (Default: None)
    negative_relationships : bool, optional
        Whether to allow for negative relationships. (Default: False)
    shared_directions : int, optional
        Only when negative_relationships is true. Number of out_features with shared relationship directions. out_features needs to be divisible with this. (Default: 1)
    """
    def __init__(
        self,
        degree: int,
        in_features: int = 1,
        out_features: int = 1,
        intercept: int = False,
        relationship_matrix: torch.Tensor = None,
        negative_relationships: bool = False,
        shared_directions: int = 1
    ) -> None:
        super().__init__()
        if degree % 2 == 0:
            raise ValueError("Degree must be an uneven number.")
        self.k = (degree - 1) // 2
        self.input_dim = in_features
        self.output_dim = out_features
        self.relationship_matrix = relationship_matrix
        self.negative_relationships = negative_relationships
        self.shared_directions = shared_directions

        self.omega = nn.Parameter(torch.zeros(1, in_features, out_features, requires_grad=True))
        self.alpha = nn.Parameter(torch.zeros(self.k, in_features, out_features, requires_grad=True))
        self.tau = nn.Parameter(torch.full((self.k, in_features, out_features), -5.0, requires_grad=True))
        if intercept:
            self.intercept = nn.Parameter(torch.zeros(out_features, requires_grad=True))
        else:
            self.register_buffer('intercept', None)

        if negative_relationships:
            if shared_directions == 0:
                raise ValueError("shared_directions must be greater than 0.")
            if out_features % shared_directions != 0:
                raise ValueError("out_features must be divisible by shared_directions.")
            self.directions = nn.Parameter(torch.zeros(in_features, int(out_features / shared_directions), requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sp_tau = F.softplus(self.tau)
        
        b = F.softplus(self.omega)
        for i in range(self.k):
            matrix = torch.zeros((2*(i+1)+1, 2*(i+1)-1, self.input_dim, self.output_dim), device=x.device)
            range_indices = torch.arange(2*(i+1)-1, device=x.device)
            matrix[range_indices, range_indices, :, :] = 1
            matrix[range_indices + 1, range_indices, :, :] = -2 * self.alpha[i]
            matrix[range_indices + 2, range_indices, :, :] = self.alpha[i] ** 2 + sp_tau[i]
            b = torch.einsum('abio,bio->aio', matrix, b) / (i + 1)
        
        if self.negative_relationships:
            b.multiply_(self.directions.repeat_interleave(self.shared_directions, dim=1))

        # remove relationship between some items and latent variables
        if self.relationship_matrix is not None:
            b[:, ~self.relationship_matrix] = 0.0
        x_powers = x.unsqueeze(2) ** torch.arange(1, 2*self.k+2, device=x.device)
        # x_powers dimensions: (batch, input_dim, degree)
        # b dimensions: (degree, input_dim, output_dim)
        result = torch.einsum('abc,cbd->ad', x_powers, b)
        if self.intercept is not None:
            result += self.intercept
        return result
    
    @torch.inference_mode()
    def get_polynomial_coefficients(self) -> torch.Tensor:
        """
        Returns the polynomial coefficients.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of tensors containing the coefficients of the polynomial with dimensions (degree, input_dim, output_dim), the second tensor contains the intercept if it exists and None otherwise.
        """
        sp_tau = F.softplus(self.tau)
        
        b = F.softplus(self.omega)
        for i in range(self.k):
            matrix = torch.zeros((2*(i+1)+1, 2*(i+1)-1, self.input_dim, self.output_dim))
            range_indices = torch.arange(2*(i+1)-1)
            matrix[range_indices, range_indices, :, :] = 1
            matrix[range_indices + 1, range_indices, :, :] = -2 * self.alpha[i]
            matrix[range_indices + 2, range_indices, :, :] = self.alpha[i] ** 2 + sp_tau[i]
            b = torch.einsum('abio,bio->aio', matrix, b) / (i + 1)

        if self.negative_relationships:
            b.multiply_(self.directions.repeat_interleave(self.shared_directions, dim=1))

        # remove relationship between some items and latent variables
        if self.relationship_matrix is not None:
            b[:, ~self.relationship_matrix] = 0.0

        return b, self.intercept

class SoftplusLinear(nn.Module):
    """
    A linear layer with positive weights ensured using the softplus function. Should be more stable than squaring and maintains gradients in contrast to clipping/abs.
    """
    def __init__(
            self,
            in_features,
            out_features,
            separate_inputs: torch.Tensor=None,
            separate_outputs: torch.Tensor=None,
            zero_inputs: torch.Tensor=None,
            zero_outputs: torch.Tensor=None,
            softplus: bool=True
        ):
        """
        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        separate_inputs : torch.Tensor, optional
            A tensor with integers summing up to in_features that determines if the inputs should be separated into multiple groups. 
            If specified, the layer will have multiple groups of inputs/outputs with separate weights. (Default: None)
        separate_outputs : torch.Tensor, optional
            Only when separate inputs is specified. A tensor with integers summing up to out_features.
            Has to have the same number of groups as separate_inputs.
            If specified, the layer will have multiple groups of inputs/outputs with separate weights. 
            (Default: None)
        zero_inputs : torch.Tensor, optional
            A boolean tensor of shape (in_features,) that determines for which inputs weights should be trained. Mostly relevant when training polytomous IRT models for which each response category has a separate network. (Default: None)
        zero_outputs : torch.Tensor, optional
            A boolean tensor of shape (out_features,) that determines which outputs should be zero. Typically used for IRT models to remove relationships between some items or item cateogires and latent traits. (Default: None)
        softplus : bool, optional
            Whether to use the softplus function or not. If False, the raw weights are used like regular linear layer. (Default: True)
        """
        super().__init__()
        if zero_outputs is None:
            zero_outputs = torch.zeros(out_features).int()
        if zero_inputs is None:
            zero_inputs = torch.zeros(in_features).int()
        if separate_inputs is not None:
            if separate_inputs.sum() != in_features:
                raise ValueError("separate_inputs must sum to in_features")
            if separate_outputs is None:
                # Split out_features into the input groups as evenly as possible
                quotient, remainder = divmod(out_features, separate_inputs.shape[0])
                groups = [quotient + 1] * remainder + [quotient] * (separate_inputs.shape[0] - remainder)
                separate_outputs = torch.tensor(groups)
            if separate_outputs.sum() != out_features:
                raise ValueError("separate_outputs must sum to out_features")
        elif separate_outputs is not None:
            raise ValueError("separate_outputs must be None if separate_inputs is None")
        
        self.in_features = in_features
        self.out_features = out_features
        self.softplus = softplus
        self.zero_inputs = zero_inputs
        self.zero_outputs = zero_outputs
        
        if separate_inputs is not None:
            free_weight = self.separate_weights(in_features, out_features, separate_inputs, separate_outputs)
        else:
            free_weight = torch.full((out_features, in_features), True)

        free_weight[zero_outputs.bool()] = False
        free_weight[:, zero_inputs.bool()] = False
        self.register_buffer("free_bias", (1 - zero_outputs).bool())
        self.register_buffer("free_weight", free_weight)

        # These are learnable parameters representing the raw weights of the layer.
        # The actual weights used in the layer's operations are the softplus of these raw weights.
        self.raw_weight_param = nn.Parameter(torch.empty(self.free_weight.sum()))
        self.bias_param = nn.Parameter(torch.empty(self.free_bias.sum()))

        self.reset_parameters()

    @staticmethod
    def separate_weights(in_features, out_features, separate_inputs, separate_outputs):
        free_weights = torch.full((out_features, in_features), False)

        input_indices = torch.cumsum(separate_inputs, dim=0) - separate_inputs
        output_indices = torch.cumsum(separate_outputs, dim=0) - separate_outputs

        for input_group_size, output_group_size, input_idx, output_idx in zip(separate_inputs, separate_outputs, input_indices, output_indices):
            free_weights[output_idx:output_idx+output_group_size, input_idx:input_idx+input_group_size] = True

        return free_weights
        
    def reset_parameters(self):
        """
        Reset the parameters of the layer.
        """
        nn.init.zeros_(self.raw_weight_param)
        nn.init.zeros_(self.bias_param)

    def forward(self, x):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        output : torch.Tensor
            Output tensor.
        """
        weight = torch.zeros((self.out_features, self.in_features), device=x.device)
        bias = torch.zeros(self.out_features, device=x.device)
        if self.softplus:
            weight[self.free_weight] = F.softplus(self.raw_weight_param)
        else:
            weight[self.free_weight] = self.raw_weight_param
        bias[self.free_bias] = self.bias_param
        output = F.linear(x, weight, bias)
        return output

class NegationLayer(nn.Module):
    def __init__(self, item_theta_relationships: torch.Tensor, inputs_per_item: int, zero_outputs: torch.Tensor):
        """
        Parameters
        ----------
        item_theta_relationships : torch.Tensor
            An integer tensor of shape (items,) that determines for which weights should be trained. Mostly used for training multidimensional IRT models for which each item or item category has a separate network and some of the item/latent variable relationships are set to be 0.
        inputs_per_item : int
            Number of inputs per item. One weight for all inputs per item.
        zero_outputs : torch.Tensor
            A boolean tensor of shape (out_features,) that determines which outputs should be 0. Typically used for the final layer for IRT models with polytomously scored items where different items have differing number of response categories. Items with fewer categories should set the outputs for nonexisting categories to 0.
        """
        super().__init__()

        self.inputs_per_item = inputs_per_item
        self.item_theta_relationships = item_theta_relationships
        self.zero_outputs = zero_outputs
        self.zero_weights = (1-item_theta_relationships).repeat_interleave(inputs_per_item).bool()
        self.register_buffer("weight", torch.zeros(item_theta_relationships.numel() * inputs_per_item))
        # One weight for each item related to the latent variable
        self.weight_param = nn.Parameter(torch.Tensor(sum(item_theta_relationships).item()))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.weight_param)

    def forward(self, x: torch.Tensor):
        # break the gradients for weights for non-used categories
        x[:, self.zero_outputs] = 0.0
        weight = self.weight.clone().detach()
        weight[~self.zero_weights] = self.weight_param.repeat_interleave(self.inputs_per_item)
        x.multiply_(weight)
        return x

    def all_item_weights(self):
        """
        Returns
        -------
        weight : torch.Tensor
            All layer weights, including the unused ones.
        """
        weights = torch.zeros(self.item_theta_relationships.numel())
        mask = (1 - self.item_theta_relationships).bool()
        weights[~mask] = self.weight_param
        return weights
