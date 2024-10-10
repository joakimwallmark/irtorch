import torch
import torch.nn as nn
import torch.nn.functional as F
from irtorch.models.base_irt_model import BaseIRTModel
from irtorch.torch_modules import RationalQuadraticSpline, NegationLayer

class SurprisalSpline(BaseIRTModel):
    r"""
    Surprisal IRT model.
    """
    def __init__(
        self,
        data: torch.Tensor = None,
        latent_variables: int = 1,
        items: int = None,
        spline_bins: int = 5,
        spline_input_bounds: tuple[float, float] = (-1.0, 4.0),
        item_theta_relationships: torch.Tensor = None,
        negative_latent_variable_item_relationships: bool = False,
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
        
        self.negative_latent_variable_item_relationships = negative_latent_variable_item_relationships
        self.output_size = self.items * 2
        self.bias_param = nn.Parameter(torch.full((self.items,), -1.4))
        
        self.add_module("spline", RationalQuadraticSpline(
            variables = self.latent_variables * (self.items * self.max_item_responses - self.items),
            num_bins=spline_bins,
            lower_input_bound=spline_input_bounds[0],
            upper_input_bound=spline_input_bounds[1],
            lower_output_bound=0.0,
            upper_output_bound=5.0,
            derivative_outside_lower_input_bound=0.
        ))
        
        for theta_dim in range(latent_variables):
            zero_outputs = 1 - item_theta_relationships[:, theta_dim].int()

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
            2D tensor. Rows are respondents and columns are probabilities.
        """
        spline_out, _ = self._modules["spline"](
            theta.repeat(1, self.items * self.max_item_responses - self.items)
        )
        # Shape below is respondents x item categories x latent variables
        spline_out = spline_out.reshape(theta.shape[0], -1, theta.shape[1])
        spline_out = spline_out.sum(dim=-1) # sum over latent variables
        return spline_out + F.softplus(self.bias_param)

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
