import torch
import torch.nn as nn
import torch.nn.functional as F
from irtorch.models.base_irt_model import BaseIRTModel
from irtorch.torch_modules import SoftplusLinear, NegationLayer
from irtorch.activation_functions import BoundedELU

class SurprisalNN(BaseIRTModel):
    r"""
    Surprisal IRT model.
    """
    def __init__(
        self,
        data: torch.Tensor = None,
        latent_variables: int = 1,
        items: int = None,
        item_theta_relationships: torch.Tensor = None,
        hidden_dim: list[int] = None,
        negative_latent_variable_item_relationships: bool = True,
        use_bounded_activation: bool = True,

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
        if hidden_dim is None:
            hidden_dim = [3] if use_bounded_activation else [2]
        else:
            if use_bounded_activation and not all(x % 3 == 0 for x in hidden_dim):
                raise ValueError("hidden_dim must be a multiple of 3 when use_bounded_activation=True")
            if not use_bounded_activation and not all(x % 2 == 0 for x in hidden_dim):
                raise ValueError("hidden_dim must be a multiple of 2 when use_bounded_activation=False")
        
        self.hidden_layers = len(hidden_dim)
        self.use_bounded_activation = use_bounded_activation
        self.negative_latent_variable_item_relationships = negative_latent_variable_item_relationships
        self.hidden_out_dim = hidden_dim[-1]
        self.output_size = self.items * 2
        # 1/exp(ln(1+exp(-1.4)) = 0.2
        self.bias_param = nn.Parameter(torch.full((self.items,), -1.4))
        
        for theta_dim in range(latent_variables):
            zero_outputs = 1 - item_theta_relationships[:, theta_dim].int()
            layer_theta_zero_out = zero_outputs.repeat_interleave(hidden_dim[0])
            self.add_module(f"linear0_dim{theta_dim}", SoftplusLinear(1, self.items * hidden_dim[0], zero_outputs=layer_theta_zero_out))

            # Hidden layers
            for i in range(1, len(hidden_dim)):
                input_dim = hidden_dim[i - 1] * self.items
                output_dim = hidden_dim[i] * self.items
                layer_theta_zero_out = zero_outputs.repeat_interleave(hidden_dim[i])
                separate_inputs = torch.tensor([hidden_dim[i - 1]] * self.items)
                separate_outputs = torch.tensor([hidden_dim[i]] * self.items)

                self.add_module(f"linear{i}_dim{theta_dim}", SoftplusLinear(
                    input_dim,
                    output_dim,
                    separate_inputs=separate_inputs,
                    separate_outputs=separate_outputs,
                    zero_inputs=getattr(self, f"linear{i-1}_dim{theta_dim}").zero_outputs,
                    zero_outputs=layer_theta_zero_out,
                ))

            if negative_latent_variable_item_relationships:
                inputs_per_items = 1
                missing_cats = torch.zeros(self.items, dtype=torch.bool)
                item_relationships = 1 - zero_outputs
                self.add_module(
                    f"negation_dim{theta_dim}",
                    NegationLayer(
                        item_theta_relationships=item_relationships,
                        inputs_per_item=inputs_per_items,
                        zero_outputs=missing_cats
                    )
                )

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
            2D tensor. Rows are respondents and columns are surprisals.
        """
        latent_variable_outputs = []
        for latent_variable in range(self.latent_variables):
            layer_out = self._modules[f"linear0_dim{latent_variable}"](theta[:, latent_variable].unsqueeze(1))
            layer_out = self.split_activation(layer_out)
            for i in range(1, self.hidden_layers):
                layer_out = self._modules[f"linear{i}_dim{latent_variable}"](layer_out)
                layer_out = self.split_activation(layer_out)

            layer_out = layer_out.reshape(-1, self.items, self.hidden_out_dim).sum(dim=2)

            if self.negative_latent_variable_item_relationships:
                layer_out = self._modules[f"negation_dim{latent_variable}"](layer_out)

            latent_variable_outputs.append(layer_out)

        out = torch.stack(latent_variable_outputs, dim=-1).sum(dim=-1)
        out = F.softplus(out)
        out += F.softplus(self.bias_param)

        return out

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
        probs = 1/output.exp()
        probs = torch.stack([probs, 1-probs], dim=2)
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
        return torch.ones(self.items, self.latent_variables)

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