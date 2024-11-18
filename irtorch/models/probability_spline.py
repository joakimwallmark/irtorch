import torch
import torch.nn as nn
import torch.nn.functional as F
from irtorch.models.base_irt_model import BaseIRTModel
from irtorch.torch_modules import RationalQuadraticSpline, NegationLayer

class ProbabilitySpline(BaseIRTModel):
    r"""
    Probabilty spline IRT model.
    """
    def __init__(
        self,
        data: torch.Tensor = None,
        latent_variables: int = 1,
        item_categories: list[int] = None,
        spline_bins: int = 5,
        item_theta_relationships: torch.Tensor = None,
        negative_latent_variable_item_relationships: bool = False,
        mc_correct: list[int] = None,
    ):
        if item_categories is None:
            if data is None:
                raise ValueError("Either item_categories or data must be provided to initialize the model.")
            else:
                # replace nan with -inf to get max
                item_categories = (torch.where(~data.isnan(), data, torch.tensor(float('-inf'))).max(dim=0).values + 1).int().tolist()

        super().__init__(latent_variables=latent_variables, item_categories=item_categories, mc_correct=mc_correct)
        self.output_size = self.items * self.max_item_responses

        correct_idx = torch.zeros(self.items, self.max_item_responses, dtype=torch.int)
        guessing = torch.full((self.items, self.max_item_responses), -torch.inf)
        guessing_idx = torch.zeros(self.items, self.max_item_responses, dtype=torch.int)
        incorrect_categories = torch.ones(self.items, self.max_item_responses, dtype=torch.int)
        for item, item_cat in enumerate(self.item_categories):
            correct_idx[item, mc_correct[item]] = 1
            guessing[item, mc_correct[item]] = 1.0
            guessing_idx[item, item_cat:self.max_item_responses] = 1
            incorrect_categories[item, mc_correct[item]] = 0
            incorrect_categories[item, item_cat:self.max_item_responses] = 0
            incorrect_categories[item, mc_correct[item]] = 0

        self.register_buffer("correct_idx", correct_idx.bool())
        self.register_buffer("guessing", guessing)
        self.register_buffer("guessing_idx", guessing_idx.bool())
        self.register_buffer("incorrect_categories", incorrect_categories.bool())
        self.guessing_param = nn.Parameter(torch.ones(incorrect_categories.sum()))
        self.spline_out = self.latent_variables * incorrect_categories.sum()

        self.add_module("spline", RationalQuadraticSpline(
            variables = self.spline_out,
            num_bins=spline_bins,
            lower_input_bound=0.0,
            upper_input_bound=1.0,
            lower_output_bound=0.0,
            upper_output_bound=1.0,
            derivative_outside_lower_input_bound=0.,
            derivative_outside_upper_input_bound=0.
        ))
        
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
        guessing = self.guessing.clone()
        guessing[self.incorrect_categories] = self.guessing_param
        probs = guessing.softmax(dim=1).unsqueeze(0).repeat(theta.shape[0], 1, 1)
        # probs = guessing.softmax(dim=1).unsqueeze(0).expand(theta.shape[0], -1, -1)
        spline_out = 1 - self._modules["spline"](
            # theta.sigmoid().expand(-1, self.spline_out)
            theta.sigmoid().repeat(1, self.spline_out)
        )[0]
        # Shape below is respondents , (items x incorrect item categories), latent variables
        spline_out = spline_out.reshape(theta.shape[0], -1, theta.shape[1])
        spline_out = spline_out.sum(dim=-1) # sum over latent variables
        probs[:, self.incorrect_categories] *= spline_out
        probs[:, self.correct_idx] = 0
        probs[:, self.correct_idx] = 1 - probs.sum(dim=2)
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
        return output

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
