import torch
from torch import nn

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
