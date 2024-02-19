import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from irtorch.models.base_irt_model import BaseIRTModel

logger = logging.getLogger('irtorch')

class Parametric(BaseIRTModel):
    """
    Parametric IRT model.
    """
    def __init__(
        self,
        latent_variables: int,
        item_categories: list[int],
        model: str,
        model_missing: bool = False,
        mc_correct: list[int] = None,
        item_z_relationships: torch.Tensor = None,
        reference_category: bool = False
    ):
        """
        Parameters
        ----------
        latent_variables : int
            Number of latent variables.
        item_categories : list[int]
            Number of categories for each item. One integer for each item. Missing responses exluded.
        model : str
            Type of parametric model. Can be "1PL", "2PL", "3PL", "GPC" or "nominal".
        model_missing : bool, optional
            Whether to model missing item responses as separate item response categories. (Default: False)
        mc_correct : list[int], optional
            The correct response category for each item. If specified, the logits for the correct responses are cumulative logits. (Default: None)
        item_z_relationships : torch.Tensor, optional
            A boolean tensor of shape (items, latent_variables). If specified, the model will have connections between latent dimensions and items where the tensor is True. If left out, all latent variables and items are related (Default: None)
        reference_category : bool, optional
            Only for the nomimal model. Whether to use the first category as an unparameterized reference category. (Default: False and uses the original parameterization given by Bock(1972))
        """
        super().__init__(latent_variables=latent_variables, item_categories=item_categories, model_missing=model_missing, mc_correct=mc_correct)
        if item_z_relationships is not None:
            if item_z_relationships.shape != (len(item_categories), latent_variables):
                raise ValueError(
                    f"latent_item_connections must have shape ({len(item_categories)}, {latent_variables})."
                )
            assert(item_z_relationships.dtype == torch.bool), "latent_item_connections must be boolean type."
            assert(torch.all(item_z_relationships.sum(dim=1) > 0)), "all items must have a relationship with a least one latent variable."
        if model not in ["1PL", "2PL", "3PL", "GPC", "nominal"]:
            raise ValueError("model_type must be one of '1PL', '2PL', '3PL', 'GPC', 'nominal'.")
        if model in ["1PL", "2PL", "3PL"]:
            item_categories = [2] * len(item_categories)

        self.model = model
        self.output_size = self.items * self.max_item_responses

        if model == "1PL":
            free_weights = torch.ones(latent_variables)
        elif model in ["2PL", "3PL"]:
            free_weights = torch.zeros(self.items, self.max_item_responses - 1, latent_variables)
            for item, item_cat in enumerate(self.modeled_item_responses):
                if item_z_relationships is not None:
                    free_weights[item, 0:item_cat, :] = item_z_relationships[item, :]
                else:
                    free_weights[item, 0:item_cat, :] = 1.0
            free_weights = free_weights.reshape(-1, latent_variables)
        elif model == "nominal":
            free_weights = torch.zeros(self.items, self.max_item_responses, latent_variables)
            for item, item_cat in enumerate(self.modeled_item_responses):
                start_1 = 1 if reference_category else 0
                if item_z_relationships is not None:
                    free_weights[item, start_1:item_cat, :] = item_z_relationships[item, :]
                else:
                    free_weights[item, start_1:item_cat, :] = 1.0
            free_weights = free_weights.reshape(-1, latent_variables)
        elif model == "GPC":
            free_weights = torch.ones(self.items, latent_variables)
            self.register_buffer("gpc_weight_multiplier", torch.arange(0, self.max_item_responses).repeat(self.items))
            if item_z_relationships is not None:
                for item, item_cat in enumerate(self.modeled_item_responses):
                    free_weights[item, :] = item_z_relationships[item, :]

        if model == "1PL":
            self.weight_param = nn.Parameter(torch.zeros(latent_variables))
        elif model in ["2PL", "3PL", "nominal", "GPC"]:
            self.weight_param = nn.Parameter(torch.zeros(free_weights.sum().int()))
            if model == "3PL":
                self.guessing_param = nn.Parameter(torch.zeros(self.items))

        number_of_bias_parameters = sum(self.modeled_item_responses) if model == "nominal" and not reference_category else sum(self.modeled_item_responses) - self.items
        self.bias_param = nn.Parameter(torch.zeros(number_of_bias_parameters))
        first_category = torch.zeros(self.items, self.max_item_responses)
        if model != "nominal" or reference_category:
            first_category[:, 0] = 1.0
        first_category = first_category.reshape(-1)
        missing_category = torch.zeros(self.items, self.max_item_responses)
        for item, item_cat in enumerate(self.modeled_item_responses):
            missing_category[item, item_cat:self.max_item_responses] = 1.0
        missing_category = missing_category.reshape(-1)
        free_bias = (1 - first_category) * (1 - missing_category)
        self.register_buffer("free_weights", free_weights.bool())
        self.register_buffer("free_bias", free_bias.bool())
        self.register_buffer("missing_category", missing_category.bool())
        self.register_buffer("first_category", first_category.bool())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight_param, mean=1., std=0.01)
        nn.init.zeros_(self.bias_param)
        if self.model == "3PL":
            self.guessing_param.data.fill_(0.25)
    
    def forward(self, z: torch.Tensor):
        """
        Forward pass of the model.

        Parameters
        ----------
        z : torch.Tensor
            2D tensor with latent variables. Rows are respondents and latent variables are columns. 

        Returns
        -------
        output : torch.Tensor
            2D tensor. Rows are respondents and columns are item category logits.
        """
        bias = torch.zeros(self.output_size, device=z.device)
        bias[self.free_bias] = self.bias_param
        
        if self.model == "1PL":
            weighted_z = (self.weight_param * z).sum(dim=1, keepdim=True)
        elif self.model in ["2PL", "3PL", "GPC"]:
            weights = torch.zeros(self.items, self.latent_variables, device=z.device)
            weights[self.free_weights] = self.weight_param
            weighted_z = torch.matmul(z, weights.T).repeat_interleave(self.max_item_responses, dim=1)
            if self.model == "GPC":
                weighted_z *= self.gpc_weight_multiplier
        elif self.model == "nominal":
            weights = torch.zeros(self.output_size, self.latent_variables, device=z.device)
            weights[self.free_weights] = self.weight_param
            weighted_z = torch.matmul(z, weights.T)

        output = weighted_z + bias
        if self.model in ["GPC", "nominal"]:
            # stop gradients from flowing through the missing categories
            output[:, self.missing_category] = -torch.inf

        output[:, self.first_category] = 0

        return output

    def log_likelihood(
        self,
        data: torch.Tensor,
        output: torch.Tensor,
        loss_reduction: str = "sum",
    ):
        """
        Compute the log likelihood of the data given the model. This is equivalent to the negative cross entropy loss.

        Parameters
        ----------
        data: torch.Tensor
            A 2D tensor with test data. Columns are items and rows are respondents
        output: torch.Tensor
            A 2D tensor with output. Columns are item response categories and rows are respondents
        loss_reduction: str, optional 
            The reduction argument for torch.nn.CrossEntropyLoss. (default is 'sum')
        
        Returns
        -------
        torch.Tensor
            The log likelihood.
        """
        data = data.long()
        data = data.view(-1)
        reshaped_output = output.reshape(-1, self.max_item_responses)
        if self.model == "3PL":
            raise NotImplementedError("3PL not implemented yet")
            # probabilities = torch.softmax(reshaped_output, dim=1)
            # guessing = self.guessing_param.unsqueeze(1).repeat(int(data.shape[0]/self.items), reshaped_output.shape[1])
            # guessing[:, 0] *= -1
            # probabilities += guessing
            # return -F.nll_loss(torch.log(probabilities), data, reduction=loss_reduction)
        else:
            return -F.cross_entropy(reshaped_output, data, reduction=loss_reduction)

    def probabilities_from_output(self, output: torch.Tensor) -> list:
        """
        Compute item probabilities from the output tensor.

        Parameters
        ----------
        output : torch.Tensor
            Output tensor from the forward pass.

        Returns
        -------
        probabilities : torch.Tensor
            3D torch tensor with dimensions (respondents, items, item categories).
        """
        reshaped_output = output.reshape(-1, self.max_item_responses)
        return F.softmax(reshaped_output, dim=1).reshape(output.shape[0], self.items, self.max_item_responses)

    def item_parameters(self) -> torch.Tensor:
        """
        Get the item parameters for a fitted model.

        Returns
        -------
        torch.Tensor, torch.Tensor
            A tuple of 2D tensor with the item parameters. Items are rows and parameters are columns. Weights are in the first tensor, ordered by latent variable. The second tensor holds the biases, ordered by item category.
        """
        biases = torch.zeros(self.output_size)
        biases[self.free_bias] = self.bias_param
        biases = biases.reshape(-1, self.max_item_responses)
        if self.model == "1PL":
            weights = self.weight_param.repeat(self.items, 1)
        elif self.model == "nominal":
            weights = torch.zeros(self.output_size, self.latent_variables)
            weights[self.free_weights] = self.weight_param
            weights = weights.reshape(-1, self.max_item_responses * self.latent_variables)
        else:
            weights = torch.zeros(self.items, self.latent_variables)
            weights[self.free_weights] = self.weight_param

        return weights, biases

    # TODO remove if not used
    @torch.inference_mode()
    def item_z_relationship_directions(self, z: torch.tensor = None) -> torch.Tensor:
        """
        Get the relationship directions between each item and latent variable for a fitted model.

        Parameters
        ----------
        z : torch.Tensor, optional
            A 2D tensor with population z scores. Only required for the nominal model.

        Returns
        -------
        torch.Tensor
            A 2D tensor with the relationships between the items and latent variables. Items are rows and latent variables are columns.
        """
        if self.model == "1PL":
            weights = self.weight_param.repeat(self.items, 1)
        if self.model == "nominal":
            weights = self.expected_item_score_slopes(z)
        else:
            weights = torch.zeros(self.items, self.latent_variables)
            weights[self.free_weights] = self.weight_param
        return weights.sign().int()
    