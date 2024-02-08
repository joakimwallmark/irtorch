import torch
from torch import nn
import torch.nn.functional as F


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
    def __init__(self, item_z_relationships: torch.Tensor, inputs_per_items: int, zero_outputs: torch.Tensor):
        """
        Parameters
        ----------
        item_z_relationships : torch.Tensor
            An integer tensor of shape (items,) that determines for which weights should be trained. Mostly used for training multidimensional IRT models for which each item or item category has a separate network and some of the item/latent variable relationships are set to be 0.
        inputs_per_items : int
            Number of inputs per item. One weight for all inputs per item.
        zero_outputs : torch.Tensor
            A boolean tensor of shape (out_features,) that determines which outputs should be 0. Typically used for the final layer for IRT models with polytomously scored items where different items have differing number of response categories. Items with fewer categories should set the outputs for nonexisting categories to 0.
        """
        super().__init__()

        self.inputs_per_items = inputs_per_items
        self.item_z_relationships = item_z_relationships
        self.zero_outputs = zero_outputs
        self.zero_weights = (1-item_z_relationships).repeat_interleave(inputs_per_items).bool()
        self.register_buffer("weight", torch.zeros(item_z_relationships.numel() * inputs_per_items))
        # One weight for each item related to the latent variable
        self.weight_param = nn.Parameter(torch.Tensor(sum(item_z_relationships).item()))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.weight_param)

    def forward(self, x: torch.Tensor):
        # break the gradients for weights for non-used categories
        x[:, self.zero_outputs] = 0.0
        weight = self.weight.clone().detach()
        weight[~self.zero_weights] = self.weight_param.repeat_interleave(self.inputs_per_items)
        x.multiply_(weight)
        return x

    def all_item_weights(self):
        """
        Returns
        -------
        weight : torch.Tensor
            All layer weights, including the unused ones.
        """
        weights = torch.zeros(self.item_z_relationships.numel())
        mask = (1 - self.item_z_relationships).bool()
        weights[~mask] = self.weight_param
        return weights
