from abc import ABC, abstractmethod
import logging
import torch
from torch import nn
import torch.nn.functional as F
from irtorch._internal_utils import linear_regression

logger = logging.getLogger('irtorch')

class BaseIRTModel(ABC, nn.Module):
    """
    Abstract base class for Item Response Theory models. All IRT models should inherit from this class.
    
    Parameters
    ----------
    latent_variables : int
        The number of latent variables.
    item_categories : list[int]
        A list of the number of categories for each item.
    mc_correct : list[int], optional
        A list of the correct response category for each multiple choice item. (default is None)
    model_missing : bool, optional
        Whether to model missing data. (default is False)
    """
    def __init__(
        self,
        latent_variables: int,
        item_categories: list[int],
        mc_correct: list[int] = None,
        model_missing: bool = False
    ):
        super().__init__()
        if mc_correct is not None:
            if not all(item >= mc for item, mc in zip(item_categories, mc_correct)):
                raise ValueError("mc_correct cannot be greater than the number of categories for each item.")
        self.latent_variables = latent_variables
        self.item_categories = item_categories
        self.modeled_item_responses = [x + 1 for x in item_categories] if model_missing else item_categories
        self.max_item_responses = max(self.modeled_item_responses)
        self.items = len(item_categories)
        self.mc_correct = mc_correct
        self.model_missing = model_missing


    @abstractmethod
    def forward(self, z) -> torch.Tensor:
        """
        Forward pass of the model. PyTorch requires this method to be implemented in subclasses of nn.Module.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor.

        Returns
        -------
        output : torch.Tensor
            Output tensor.
        """

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
        reshaped_output = output.reshape(-1, self.max_item_responses)
        return F.softmax(reshaped_output, dim=1).reshape(output.shape[0], self.items, self.max_item_responses)

    def item_probabilities(self, z: torch.Tensor) -> torch.Tensor:
        """
        Returns the probabilities for each possible response for all items.

        Parameters
        ----------
        z : torch.Tensor
            A 2D tensor. Rows are respondents and latent variables are columns.

        Returns
        -------
        torch.Tensor
            3D tensor with dimensions (respondents, items, item categories)
        """
        output = self(z)
        return self.probabilities_from_output(output)


    def log_likelihood(
        self,
        data: torch.Tensor,
        output: torch.Tensor,
        loss_reduction: str = "sum",
    ) -> torch.Tensor:
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
        return -F.cross_entropy(reshaped_output, data, reduction=loss_reduction)

    def expected_scores(self, z: torch.Tensor, return_item_scores: bool = True) -> torch.Tensor:
        """
        Computes the model expected item scores/test scores for each respondent.

        Parameters
        ----------
        z : torch.Tensor
            A 2D tensor with z scores. Each row represents one respondent, and each column represents a latent variable.
        return_item_scores : bool, optional
            Whether to return the expected item scores. If False, returns the expected sum scores (default is True)

        Returns
        -------
        torch.Tensor
            A tensor with the expected scores for each respondent.
        """
        item_probabilities = self.item_probabilities(z)
        if self.mc_correct:
            item_scores = torch.zeros(item_probabilities.shape[1], item_probabilities.shape[2])
            item_scores.scatter_(1, (torch.tensor(self.mc_correct) - 1 + self.model_missing).unsqueeze(1), 1)
        else:
            item_scores = (torch.arange(item_probabilities.shape[2])).repeat(item_probabilities.shape[1], 1)
            if self.model_missing:
                item_scores[:, 1:] = item_scores[:, 1:] - 1
        expected_item_scores = (item_probabilities * item_scores).sum(dim=2)

        if return_item_scores:
            return expected_item_scores
        else:
            return expected_item_scores.sum(dim=1)
        
    def item_z_relationship_directions(self, z: torch.Tensor) -> torch.Tensor:
        """
        Get the relationships between each item and latent variable for a fitted model.

        Parameters
        ----------
        z : torch.Tensor
            A 2D tensor with latent z scores from the population of interest. Each row represents one respondent, and each column represents a latent variable.

        Returns
        -------
        torch.Tensor
            A 2D tensor with the relationships between the items and latent variables. Items are rows and latent variables are columns.
        """
        item_sum_scores = self.expected_scores(z)
        item_z_mask = torch.zeros(self.items, self.latent_variables)
        for item, _ in enumerate(self.modeled_item_responses):
            weights = linear_regression(z, item_sum_scores[:,item].reshape(-1, 1))[1:].reshape(-1)
            item_z_mask[item, :] = weights.sign().int()
        
        return item_z_mask.int()

    @torch.inference_mode()
    def sample_test_data(self, z: torch.Tensor) -> torch.Tensor:
        """
        Sample test data for the provided z scores.

        Parameters
        ----------
        z : torch.Tensor
            The latent scores.

        Returns
        -------
        torch.Tensor
            The sampled test data.
        """
        probs = self.item_probabilities(z)
        dist = torch.distributions.Categorical(probs)
        return dist.sample().float()
    
    @torch.inference_mode(False)
    def probability_gradients(self, z: torch.Tensor) -> torch.Tensor:
        """
        Calculate the gradients of the item response probabilities with respect to the z scores.

        Parameters
        ----------
        z : torch.Tensor
            A 2D tensor containing latent variable z scores. Each column represents one latent variable.

        Returns
        -------
        torch.Tensor
            A torch tensor with the gradients for each z score. Dimensions are (z rows, items, item categories, latent variables).
        """
        z = z.clone().requires_grad_(True)

        def compute_jacobian(z_single):
            jacobian = torch.func.jacrev(self.item_probabilities)(z_single.view(1, -1))
            return jacobian.squeeze((0, 3))

        logger.info("Computing Jacobian for all items and item categories...")
        # vectorized version of jacobian
        gradients = torch.vmap(compute_jacobian)(z)
        return gradients
