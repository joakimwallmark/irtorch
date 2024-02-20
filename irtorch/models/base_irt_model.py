from abc import ABC, abstractmethod
import logging
import torch
from torch import nn
import torch.nn.functional as F
from irtorch.helper_functions import linear_regression

logger = logging.getLogger('irtorch')

class BaseIRTModel(ABC, nn.Module):
    """
    Abstract base class for Item Response Theory models.
    
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
    def forward(self, z):
        """
        Forward pass of the model.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor.

        Returns
        -------
        output : torch.Tensor
            Output tensor.
        """

    @abstractmethod
    def probabilities_from_output(self, output: torch.Tensor):
        """
        Compute probabilities from the output tensor.

        Parameters
        ----------
        output : torch.Tensor
            Output tensor from the forward pass.

        Returns
        -------
        torch.Tensor
            3D tensor with dimensions (respondents, items, item categories)
        """

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
        return -F.cross_entropy(reshaped_output, data, reduction=loss_reduction)

    def expected_item_sum_score(self, z: torch.Tensor, return_item_scores: bool = True):
        """
        Computes the model expected item scores/sum scores for each respondent.

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
        
    @torch.inference_mode(False)
    def expected_item_score_slopes(
        self,
        z: torch.Tensor,
        bit_scores: torch.Tensor = None,
        rescale_by_item_score: bool = True,
    ):
        """
        Computes the slope of the expected item scores, averaged over the sample in z. Similar to loadings in traditional factor analysis. For each separate latent variable, the slope is computed as the average of the slopes of the expected item scores for each item, using the median z scores for the other latent variables.

        Parameters
        ----------
        z : torch.Tensor
            A 2D tensor with latent z scores from the population of interest. Each row represents one respondent, and each column represents a latent variable.
        bit_scores : torch.Tensor, optional
            A 2D tensor with bit scores corresponding to each z score in z. If provided, slopes will be computed on the bit scales. (default is None)
        rescale_by_item_score : bool, optional
            Whether to rescale the expected items scores to have a max of one by dividing by the max item score. (default is True)

        Returns
        -------
        torch.Tensor
            A tensor with the expected item score slopes.
        """
        if z.shape[0] < 2:
            raise ValueError("z must have at least 2 rows.")
        if bit_scores is not None and z.shape != bit_scores.shape:
            raise ValueError("z and bit_scores must have the same shape.")
        if z.requires_grad:
            z.requires_grad_(False)

        median, _ = torch.median(z, dim=0)
        mean_slopes = torch.zeros(len(self.modeled_item_responses), z.shape[1])
        if bit_scores is not None:
            raise NotImplementedError("bit score slopes not implemented yet.")
        #     item_z_directions = self.item_z_relationship_directions()
        for latent_variable in range(z.shape[1]):
            z_scores = median.repeat(z.shape[0], 1)
            z_scores[:, latent_variable], sort_indices = z[:, latent_variable].sort()
            z_scores.requires_grad_(True)
            expected_item_sum_scores = self.expected_item_sum_score(z_scores, return_item_scores=True)
            if not self.mc_correct and rescale_by_item_score:
                expected_item_sum_scores = expected_item_sum_scores / (torch.tensor(self.modeled_item_responses) - 1)

            if bit_scores is None:
                # sum z_scores gradients per item
                for i in range(expected_item_sum_scores.shape[1]):
                    if z_scores.grad is not None:
                        z_scores.grad.zero_()
                    expected_item_sum_scores[:, i].sum().backward(retain_graph=True)
                    mean_slopes[i, latent_variable] = z_scores.grad[:, latent_variable].mean()
            # else: TODO for bit scores
                # item_z_directions[:, latent_variable]
                # unique_bit = bit_scores[:, latent_variable].unique(sorted=True)
                # dy_dx = (expected_item_sum_scores[1:, :] - expected_item_sum_scores[:-1, :]) / (unique_bit[1:] - unique_bit[:-1]).view(-1, 1)
            # else:
            # unique_z = z[:, latent_variable].unique(sorted=True)
            # z_scores = median.repeat(unique_z.shape[0], 1)
            # z_scores[:, latent_variable] = unique_z
            # expected_item_sum_scores = self.expected_item_sum_score(z_scores, return_item_scores=True)
            # dy_dx = (expected_item_sum_scores[1:, :] - expected_item_sum_scores[:-1, :]) / (unique_z[1:] - unique_z[:-1]).view(-1, 1)
            # mean_slopes[:, latent_variable] = torch.mean(dy_dx, dim=0)
            

        return mean_slopes

    def item_z_relationship_directions(self, z: torch.Tensor):
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
        item_sum_scores = self.expected_item_sum_score(z)
        item_z_mask = torch.zeros((len(self.items), self.latent_variables)).bool()
        for item, _ in enumerate(self.modeled_item_responses):
            weights = linear_regression(z, item_sum_scores[:,item].reshape(-1, 1))[1:].reshape(-1)
            item_z_mask[item, :] = weights >= 0
        
        return item_z_mask.int()

    @torch.inference_mode()
    def sample_test_data(self, z: torch.Tensor):
        """
        Sample test data given latent z scores.

        Parameters
        ----------
        z : torch.Tensor
            The latent scores.

        Returns
        -------
        torch.Tensor
            The sampled test data.
        """
        # TODO change to work with new item_probabilities method
        probs = self.item_probabilities(z)
        # Initialize an empty tensor to hold the samples
        test_data = torch.empty(z.shape[0], len(probs))
        for item_index, item_probs in enumerate(probs):
            dist = torch.distributions.Categorical(item_probs)
            test_data[:, item_index] = dist.sample()
        return test_data
    
    @torch.inference_mode(False)
    def probability_gradients(self, z: torch.Tensor):
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
        # Jacobian for each row in z
        gradients = [torch.autograd.functional.jacobian(self.item_probabilities, z[i].view(1, -1)).squeeze((0, 3)) for i in range(z.shape[0])]
        gradients = torch.stack(gradients)

        return gradients
    
    def information(self, z: torch.Tensor, item: bool = True):
        """
        Calculate the Fisher information for the z scores (matrix for multidimensional z). If 'item' is True, the item information is computed. Otherwise, the test information is computed.

        Parameters
        ----------
        z : torch.Tensor
            A 2D tensor containing latent variable z scores for which to compute the information. Each column represents one latent variable.
        item : bool, optional
            Whether to compute the information. If False, the test information is computed. (default is True)

        Returns
        -------
        torch.Tensor
            A torch tensor with the information for each z score. Dimensions are (z, 1) if item is False and (z, items) if item is True.
        """
        probabilities = self.item_probabilities(z.clone())
        gradients = self.probability_gradients(z)

        squared_grad_matrices = torch.zeros(gradients.shape[0], gradients.shape[1], gradients.shape[2], gradients.shape[3], gradients.shape[3])
        for i in range(gradients.shape[0]):
            for j in range(gradients.shape[1]):
                for k in range(gradients.shape[2]):
                    squared_grad_matrices[i, j, k] = torch.outer(gradients[i, j, k], gradients[i, j, k])

        information_matrices = squared_grad_matrices / probabilities.unsqueeze(-1).unsqueeze(-1).expand_as(squared_grad_matrices)

        if item:
            return information_matrices.nansum(dim=2)
        else:
            return information_matrices.nansum(dim=(1, 2))
