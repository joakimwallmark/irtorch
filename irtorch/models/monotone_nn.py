import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from irtorch.models.base_irt_model import BaseIRTModel
from irtorch.layers import SoftplusLinear, NegationLayer
from irtorch.activation_functions import BoundedELU

logger = logging.getLogger('irtorch')

class MonotoneNN(BaseIRTModel):
    r"""
    Nonparametric Monotone Neural Network IRT model.
    The model is a feedforward neural network with a separate monotone nonparametric latent variable weight function for each item or item category.

    If mc_correct is specified, the latent variable effect for the correct item response is a cumulative sum of the effects for the other possible item responses to ensure monotonicity. This model is also referred to as the Monotone Multiple Choice Neural Network (MMCNN) model.
    
    If mc_correct is not specified, the item response categories are treated as ordered, and the latent variable effect for an item response category is the cumulative sum of the effects for the lower categories.

    Parameters
    ----------
    latent_variables : int
        Number of latent variables.
    item_categories : list[int]
        Number of categories for each item. One integer for each item. Missing responses exluded.
    hidden_dim : list[int]
        Number of neurons in each hidden layer. For separate='items' or separate='categories', each element is the number of neurons for each separate item or category. For separate='none', each element is the number of neurons for each layer. Needs to be a multiple of 3 is when use_bounded_activation=True and a multiple of 2 when use_bounded_activation=False.
    model_missing : bool, optional
        Whether to model missing item responses as separate item response categories. (Default: False)
    mc_correct : list[int], optional
        The correct response category for each item. (Default: None)
    separate : str, optional
        Whether to fit separate latent trait weight functions for items or items categories. Can be 'items' or 'categories'. 
        Note that 'categories' results in a more flexible model with more parameters. (Default: 'categories')
    item_z_relationships : torch.Tensor, optional
        A boolean tensor of shape (items, latent_variables). If specified, the model will have connections between latent dimensions and items where the tensor is True. If left out, all latent variables and items are related (Default: None)
    negative_latent_variable_item_relationships : bool, optional
        Whether to allow for negative latent variable item relationships. (Default: True)
    use_bounded_activation : bool, optional
        Whether to use bounded activation functions. (Default: True)

    Notes
    -----
    For an item :math:`j` with :math:`m=0, 1, 2, \ldots, M_j` possible item responses/scores, the model defines the probability for responding with a score of :math:`x` as follows (selecting response option :math:`x` for multiple choice items):

    .. math::

        P(X_j=x | \mathbf{z}) = \frac{
            \exp(z_{jx}(\mathbf{z}))
        }{
            \sum_{m=0}^{M_j}
                \exp(z_{jm}(\mathbf{z}))
        },

    where:

    - :math:`\mathbf{z}` is a vector of latent variables.
    - When mc_correct is not specified: 
        - :math:`z_{jm}(\mathbf{z}) = \sum_{c=0}^{m}\text{monotone}_{jc}(\mathbf{z}) + b_{jm}`.
    - When mc_correct is specified:
        - :math:`z_{jm}(\mathbf{z}) = \text{monotone}_{jm}(\mathbf{z}) + b_{jm}` for all incorrect response options, and :math:`z_{jm}(\mathbf{z}) = \sum_{c=0}^{M_j}\text{monotone}_{jc}(\mathbf{z}) + b_{jm}` for the correct response option.
    - :math:`\text{monotone}_{jm}(\mathbf{z})` is a monotonic neural network with ELU based activation functions as per :cite:t:`Runje2023`.
    """
    def __init__(
        self,
        latent_variables: int,
        item_categories: list[int],
        hidden_dim: list[int],
        model_missing: bool = False,
        mc_correct: list[int] = None,
        item_z_relationships: torch.Tensor = None,
        separate: str = "categories",
        negative_latent_variable_item_relationships: bool = True,
        use_bounded_activation: bool = True
    ):
        super().__init__(latent_variables, item_categories, mc_correct, model_missing)
        if item_z_relationships is not None:
            if item_z_relationships.shape != (len(item_categories), latent_variables):
                raise ValueError(
                    f"latent_item_connections must have shape ({len(item_categories)}, {latent_variables})."
                )
            assert(item_z_relationships.dtype == torch.bool), "latent_item_connections must be boolean type."
            assert(torch.all(item_z_relationships.sum(dim=1) > 0)), "all items must have a relationship with a least one latent variable."
        else:
            item_z_relationships = torch.tensor([[True] * latent_variables] * self.items, dtype=torch.bool)
        if hidden_dim is None:
            hidden_dim = [3]
        else:
            if use_bounded_activation and not all(x % 3 == 0 for x in hidden_dim):
                raise ValueError("hidden_dim must be a multiple of 3 when use_bounded_activation=True")
            if not use_bounded_activation and not all(x % 2 == 0 for x in hidden_dim):
                raise ValueError("hidden_dim must be a multiple of 2 when use_bounded_activation=False")

        self.item_z_relationships = item_z_relationships
        self.separate = separate
        self.hidden_layers = len(hidden_dim)
        self.output_length = self.items * self.max_item_responses
        self.negative_latent_variable_item_relationships = negative_latent_variable_item_relationships
        self.use_bounded_activation = use_bounded_activation
        self.hidden_out_dim = hidden_dim[-1]
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

        for z_dim in range(latent_variables):
            zero_outputs = 1 - item_z_relationships[:, z_dim].int()

            # Input layer
            input_dim = 1
            if separate == "items":
                output_dim = self.items * hidden_dim[0]
                layer_zero_out = zero_outputs.repeat_interleave(hidden_dim[0])
            elif separate == "categories":
                output_dim = self.output_length * hidden_dim[0]
                # missing categories output 0
                missing_category_out = missing_categories.repeat_interleave(hidden_dim[0]).int()
                layer_zero_out = zero_outputs.repeat_interleave(hidden_dim[0] * self.max_item_responses)
                layer_zero_out = torch.bitwise_or(missing_category_out, layer_zero_out)

            self.add_module(f"linear0_dim{z_dim}", SoftplusLinear(input_dim, output_dim, zero_outputs=layer_zero_out))

            # Hidden layers
            for i in range(1, len(hidden_dim)):
                if separate == "items":
                    output_dim = self.items * hidden_dim[i]
                    layer_zero_out = zero_outputs.repeat_interleave(hidden_dim[i])
                    separate_inputs = torch.tensor([hidden_dim[i - 1]] * self.items)
                    separate_outputs = torch.tensor([hidden_dim[i]] * self.items)
                    input_dim = hidden_dim[i - 1] * self.items
                if separate == "categories":
                    output_dim = self.output_length * hidden_dim[i]
                    missing_category_out = missing_categories.repeat_interleave(hidden_dim[i]).int()
                    layer_zero_out = zero_outputs.repeat_interleave(hidden_dim[i] * self.max_item_responses)
                    # missing categories output 0
                    layer_zero_out = torch.bitwise_or(missing_category_out, layer_zero_out)
                    separate_inputs = torch.tensor([hidden_dim[i - 1]] * self.output_length)
                    separate_outputs = torch.tensor([hidden_dim[i]] * self.output_length)
                    input_dim = hidden_dim[i - 1] * self.output_length

                self.add_module(f"linear{i}_dim{z_dim}", SoftplusLinear(
                    input_dim,
                    output_dim,
                    separate_inputs=separate_inputs,
                    separate_outputs=separate_outputs,
                    zero_inputs=getattr(self, f"linear{i-1}_dim{z_dim}").zero_outputs,
                    zero_outputs=layer_zero_out,
                ))

            if negative_latent_variable_item_relationships:
                inputs_per_items = 1 if separate == "items" else self.max_item_responses
                missing_cats = self.missing_categories if separate == "categories" else torch.zeros(self.items, dtype=torch.bool)
                item_relationships = 1 - zero_outputs
                self.add_module(
                    f"negation_dim{z_dim}",
                    NegationLayer(
                        item_z_relationships=item_relationships,
                        inputs_per_items=inputs_per_items,
                        zero_outputs=missing_cats
                    )
                )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        latent_variable_outputs = []
        for latent_variable in range(self.latent_variables):
            layer_out = self._modules[f'linear0_dim{latent_variable}'](z[:, latent_variable].unsqueeze(1))

            layer_out = self.split_activation(layer_out)
            for i in range(1, self.hidden_layers):
                layer_out = self._modules[f'linear{i}_dim{latent_variable}'](layer_out)
                layer_out = self.split_activation(layer_out)

            layer_out = layer_out.reshape(-1, self.separations, self.hidden_out_dim).sum(dim=2)

            if self.negative_latent_variable_item_relationships:
                layer_out = self._modules[f'negation_dim{latent_variable}'](layer_out)

            latent_variable_outputs.append(layer_out)
        
        out = torch.stack(latent_variable_outputs, dim=-1).sum(dim=-1)

        if self.separate == "items":
            out = out.repeat_interleave(self.max_item_responses, dim=1)

        if self.mc_correct_output_idx is None:
            out = out.view(out.shape[0], -1, self.max_item_responses).cumsum(dim=2).reshape(out.shape[0], -1)
        else:
            out[:, self.mc_correct_output_idx] = out.view(out.shape[0], -1, self.max_item_responses).sum(dim=2)

        bias = torch.zeros(self.output_length, device=z.device)
        bias[self.free_bias] = self.bias_param
        out += bias
        out[:, self.missing_categories] = -torch.inf
        return out

    def split_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs various activation functions on every second/third item in the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape as the input tensor.
        """
        if self.use_bounded_activation:
            x1 = F.elu(x[:, ::3])
            x2 = -F.elu(-x[:, 1::3])
            x3 = BoundedELU.apply(x[:, 2::3], 1.0)
            y = torch.stack((x1, x2, x3), dim=2).view(x.shape)
        else:
            x1 = F.elu(x[:, ::2])
            x2 = -F.elu(-x[:, 1::2])
            y = torch.stack((x1, x2), dim=2).view(x.shape)
        return y

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


    def item_z_relationship_directions(self, *args) -> torch.Tensor:
        """
        Get the relationship directions between each item and latent variable for a fitted model.

        Returns
        -------
        torch.Tensor
            A 2D tensor with the relationships between the items and latent variables. Items are rows and latent variables are columns.
        """
        return torch.stack([getattr(self, f"negation_dim{i}").all_item_weights().sign() for i in range(self.latent_variables)], dim=1).int()
