from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import logging
import torch
from torch import nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
from irtorch._internal_utils import linear_regression, impute_missing_internal, dynamic_print, one_hot_encode_test_data, get_missing_mask

if TYPE_CHECKING:
    from irtorch.rescale.scale import Scale
    from irtorch.estimation_algorithms import BaseIRTAlgorithm

logger = logging.getLogger("irtorch")

class BaseIRTModel(ABC, nn.Module):
    """
    Abstract base class for Item Response Theory models. All IRT models should inherit from this class.
    
    Parameters
    ----------
    latent_variables : int
        The number of latent variables.
    item_categories : list[int]
        A list of the number of response categories for each item.
    mc_correct : list[int], optional
        A list of the correct response category for each multiple choice item. (default is None)
    """
    def __init__(
        self,
        latent_variables: int,
        item_categories: list[int],
        mc_correct: list[int] = None,
    ):
        super().__init__()
        if mc_correct is not None:
            if not all(item >= mc for item, mc in zip(item_categories, mc_correct)):
                raise ValueError("mc_correct cannot be greater than the number of categories for each item.")
        self.latent_variables = latent_variables
        self.item_categories = item_categories
        self.max_item_responses = max(self.item_categories)
        self.items = len(item_categories)
        self.mc_correct = mc_correct
        self.algorithm = None
        self.scale : list[Scale] = []

        self._evaluate = None
        self._plot = None

    @property
    def evaluate(self):
        """
        Various methods for IRT model evaluation as described in :class:`irtorch.Evaluator`.

        Returns
        -------
        Evaluation
            An instance of the :class:`irtorch.Evaluation` class.
        """
        if self._evaluate is None:
            from ..evaluator import Evaluator
            self._evaluate = Evaluator(self)
        return self._evaluate

    @property
    def plot(self):
        """
        Methods for IRT model plotting. An instance of :class:`irtorch.Plotter`.

        Returns
        -------
        Plotting
            An instance of the :class:`irtorch.Plotting` class.
        """
        if self._plot is None:
            from ..plotter import Plotter
            self._plot = Plotter(self)
        return self._plot
    
    @abstractmethod
    def forward(self, theta) -> torch.Tensor:
        """
        Forward pass of the model. PyTorch requires this method to be implemented in subclasses of nn.Module.

        Parameters
        ----------
        theta : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

    def fit(
        self,
        train_data: torch.Tensor,
        algorithm: BaseIRTAlgorithm,
        **kwargs
    ) -> None:
        """
        Train the model.

        Parameters
        ----------
        train_data : torch.Tensor
            The training data. Item responses should be coded 0, 1, ... and missing responses coded as nan or -1.
        algorithm : BaseIRTAlgorithm
            The algorithm to use for training the model. Needs to inherit :class:`irtorch.estimation_algorithms.BaseIRTAlgorithm`
        **kwargs
            Additional keyword arguments to pass to the algorithm's fit method.
        """
        algorithm.fit(self, train_data=train_data, **kwargs)
        self.algorithm = algorithm

    def add_scale_transformation(self, scale: Scale) -> None:
        """
        Add a scale transformation to the model. The rescaling instance should inherit from :class:`irtorch.rescale.Scale`.
        """
        self.scale.append(scale)

    def detach_scale_transformations(self) -> None:
        """
        Remove a scale transformation from the model.
        """
        self.scale = []

    def transform_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Transform the latent variables using the scale transformation(s).

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing latent variable theta scores. Each column represents one latent variable.

        Returns
        -------
        torch.Tensor
            A 2D tensor containing transformed theta scores. Each column represents one latent variable.
        """
        for scale in self.scale:
            theta = scale(theta)
        return theta

    def inverse_transform_theta(self, transformed_theta: torch.Tensor) -> torch.Tensor:
        """
        Reverse transformed latent variables to their original theta scale.

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing transformed theta scores. Each column represents one latent variable.

        Returns
        -------
        torch.Tensor
            A 2D tensor containing original theta scores. Each column represents one latent variable.
        """
        for scale in self.scale:
            transformed_theta = scale.inverse(transformed_theta)
        return transformed_theta
    
    def theta_transform_jacobian(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute the Jacobian matrix of the scale transformations for each row in the input theta scores.

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing latent variable theta scores. Each column represents one latent variable.

        Returns
        -------
        torch.Tensor
            A torch tensor with the gradients for each theta score. Dimensions are (theta rows, latent variables, latent variables) where the last two are the jacobians.
        """
        jacobian = torch.eye(self.latent_variables).repeat(theta.shape[0], 1, 1)
        for scale in self.scale:
            jacobian = torch.matmul(scale.jacobian(theta), jacobian)
            theta = scale(theta)
        return jacobian

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

    def item_probabilities(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Returns the probabilities for each possible response for all items.

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor. Rows are respondents and latent variables are columns.

        Returns
        -------
        torch.Tensor
            3D tensor with dimensions (respondents, items, item categories)
        """
        output = self(theta)
        return self.probabilities_from_output(output)


    def log_likelihood(
        self,
        data: torch.Tensor,
        output: torch.Tensor,
        missing_mask: torch.Tensor = None,
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
        missing_mask: torch.Tensor, optional
            A 2D tensor with missing data mask. (default is None)
        loss_reduction: str, optional 
            The reduction argument for torch.nn.functional.cross_entropy(). (default is 'sum')
        
        Returns
        -------
        torch.Tensor
            The log likelihood.
        """
        data = data.long()
        data = data.view(-1)
        reshaped_output = output.reshape(-1, self.max_item_responses)
        # Remove missing values from log likelihood calculation
        if missing_mask is not None:
            missing_mask = missing_mask.view(-1)
            reshaped_output = reshaped_output[~missing_mask]
            respondents = data.size(0)
            data = data[~missing_mask]
        
        ll = -F.cross_entropy(reshaped_output, data, reduction=loss_reduction)
        # For MML, we need the output to include missing values for missing item responses
        if loss_reduction == "none" and missing_mask is not None:
            ll_masked = torch.full((respondents, ), torch.nan, device= ll.device)
            ll_masked[~missing_mask] = ll
            return ll_masked

        return ll

    def expected_scores(self, theta: torch.Tensor, return_item_scores: bool = True) -> torch.Tensor:
        r"""
        Computes the model expected item scores/test scores for each respondent (row in theta).

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor with theta scores. Each row represents one respondent, and each column represents a latent variable.
        return_item_scores : bool, optional
            Whether to return the expected item scores. If False, returns the expected sum scores (default is True)

        Returns
        -------
        torch.Tensor
            A tensor with the expected scores for each respondent.
        """
        item_probabilities = self.item_probabilities(theta)
        if self.mc_correct:
            item_scores = torch.zeros(item_probabilities.shape[1], item_probabilities.shape[2])
            item_scores.scatter_(1, torch.tensor(self.mc_correct).unsqueeze(1), 1)
        else:
            item_scores = (torch.arange(item_probabilities.shape[2])).repeat(item_probabilities.shape[1], 1)
        expected_item_scores = (item_probabilities * item_scores).sum(dim=2)

        if return_item_scores:
            return expected_item_scores
        else:
            return expected_item_scores.sum(dim=1)

    @torch.inference_mode(False)
    def expected_item_score_gradients(
        self,
        theta: torch.Tensor,
        rescale_by_item_score: bool = True,
        rescale: bool = True,
    ) -> torch.Tensor:
        r"""
        Computes expected item score gradients for each item :math:`j` with respect to the latent variable(s) :math:`\mathbf{\theta}`.

        Parameters
        ----------
        theta : torch.Tensor, optional
            A 2D tensor with theta scores from the population of interest. Each row represents one respondent, and each column represents a latent variable.
        rescale_by_item_score : bool, optional
            Whether to rescale the expected items scores to have a max of one by dividing by the max item score. (default is True)
        rescale : bool, optional
            Whether to compute the gradients on the rescaled scale if it exists. Only possible for scale transformations for which gradients are available. (default is True)

        Returns
        -------
        torch.Tensor
            A 3D tensor with the expected item score gradients. Dimensions are (theta rows, items, latent_variables).

        Notes
        -----
        When rescale is True, the gradients are computed on the transformed scale using the Jacobian of the transformation.
        Let :math:`\mathbf{\theta} = f(\mathbf{\theta}^*)` be the transformed theta scores (original scores being :math:`\mathbf{\theta}^*`) with Jacobian :math:`\mathbf{J}_f`
        and Jacobian inverse :math:`\mathbf{J}^{-1}_f`.
        We know that :math:`\nabla_{\boldsymbol{\theta}} \mathbb{E}(X_j|\boldsymbol{\theta})=\nabla_{\boldsymbol{\theta}^*} \mathbb{E}(X_j|\boldsymbol{\theta}^*) \mathbf{J}^{-1}_f` by the chain rule.
        Thus we can rewrite as follows:

        .. math::

            \begin{align}
                \nabla_{\boldsymbol{\theta}} \mathbb{E}(X_j|\boldsymbol{\theta})\mathbf{J}_f &=\nabla_{\boldsymbol{\theta}^*} \mathbb{E}(X_j|\boldsymbol{\theta}^*) \mathbf{J}^{-1}_f\mathbf{J}_f \\
                \left(\nabla_{\boldsymbol{\theta}} \mathbb{E}(X_j|\boldsymbol{\theta})\mathbf{J}_f\right)^{T} &=\left(\nabla_{\boldsymbol{\theta}^*} \mathbb{E}(X_j|\boldsymbol{\theta}^*) \mathbf{I}\right)^{T}\\
                \mathbf{J}_f^{T}\nabla_{\boldsymbol{\theta}} \mathbb{E}(X_j|\boldsymbol{\theta})^T &=\nabla_{\boldsymbol{\theta}^*} \mathbb{E}(X_j|\boldsymbol{\theta}^*)^{T}.
            \end{align}

        The last line gives us a linear equation that is solved using `torch.linalg.solve` to get :math:`\nabla_{\boldsymbol{\theta}} \mathbb{E}(X_j|\boldsymbol{\theta})`.
        This is more efficient as computing :math:`\mathbf{J}^{-1}_f` is not needed.
        """
        if theta.requires_grad:
            theta.requires_grad_(False)

        gradients = torch.zeros(theta.shape[0], len(self.item_categories), theta.shape[1])
        theta_scores = theta.clone()
        theta_scores.requires_grad_(True)
        expected_item_sum_scores = self.expected_scores(theta_scores, return_item_scores=True)
        if not self.mc_correct and rescale_by_item_score:
            expected_item_sum_scores = expected_item_sum_scores / (torch.tensor(self.item_categories) - 1)

        # item score slopes for each item
        for item in range(expected_item_sum_scores.shape[1]):
            if theta_scores.grad is not None:
                theta_scores.grad.zero_()
            dynamic_print(f"Computing gradients for item {item+1}...")
            expected_item_sum_scores[:, item].sum().backward(retain_graph=True)
            gradients[:, item, :] = theta_scores.grad[:, :]

        if rescale and self.scale:
            gradients = self._rescale_gradients(gradients, theta)

        return gradients

    def _rescale_gradients(self, gradients: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Rescale the gradients to the original theta scale using the Jacobian of the transformation.

        Parameters
        ----------
        gradients : torch.Tensor
            A tensor with gradients with respect to the original thetas. The last dimension holds the partial derivatives.
        theta : torch.Tensor
            A 2D tensor with theta scores for which the gradients tensor were computed.

        Returns
        -------
        torch.Tensor
            A tensor with gradients with respect to the transformed thetas.
        """
        original_shape = gradients.shape
        non_transform_gradients = gradients.unsqueeze(-1)
        non_transform_gradients = non_transform_gradients.view(-1, self.latent_variables, 1)
        transform_jacobian = self.theta_transform_jacobian(theta)
        transform_jacobian_T = transform_jacobian.transpose(1, 2)
        transform_jacobian_T = transform_jacobian_T.repeat_interleave(
            int(non_transform_gradients.shape[0]/transform_jacobian_T.shape[0]),
            dim=0
        )
        gradients = torch.linalg.solve(transform_jacobian_T, non_transform_gradients)
        gradients[gradients.isnan()] = 0.
        gradients[gradients == torch.inf] = 0.
        # transpose back to original shape
        gradients = gradients.view(original_shape)
        return gradients

    def information(
        self,
        theta: torch.Tensor,
        item: bool = True,
        degrees: list[int] = None,
        rescale: bool = True,
    ) -> torch.Tensor:
        r"""
        Calculate the Fisher information matrix (FIM) for the theta scores (or the information in the direction supplied by degrees).

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing latent variable theta scores for which to compute the information. Each column represents one latent variable.
        item : bool, optional
            Whether to compute the information for each item (True) or for the test as a whole (False). (default is True)
        degrees : list[int], optional
            For multidimensional models. A list of angles in degrees between 0 and 90, one for each latent variable. Specifies the direction in which to compute the information. (default is None and returns the full FIM)
        rescale : bool, optional
            Whether to compute information on the rescaled scale if it exists. Only possible for scale transformations for which gradients are available. (default is True)

        Returns
        -------
        torch.Tensor
            A tensor with the information for each theta score. Dimensions are:
            
            - By default: (theta rows, items, FIM rows, FIM columns).
            - If degrees are specified: (theta rows, items).
            - If item is False: (theta rows, FIM rows, FIM columns).
            - If degrees are specified and item is False: (theta rows).

        Notes
        -----
        In the context of IRT, the Fisher information matrix measures the amount of information
        that a test taker's responses :math:`X` carries about the latent variable(s)
        :math:`\mathbf{\theta}`. It is defined as:

        .. math::

            I(\mathbf{\theta}) = \mathbb{E}_{X|\mathbf{\theta}}\left[ \left(\nabla_{\mathbf{\theta}} \ell(\mathbf{\theta}|X) \right) \left(\nabla_{\mathbf{\theta}} \ell(\mathbf{\theta}|X) \right)^T \right] =
            -\mathbb{E}_{X|\mathbf{\theta}}\left[\left(\nabla_{\mathbf{\theta}}^2 \ell(\mathbf{\theta}|X) \right)\right]

        Where:

        - :math:`I(\mathbf{\theta})` is the Fisher Information Matrix.
        - :math:`\ell(\mathbf{\theta}|X)` is the log-likelihood of :math:`\mathbf{\theta}`, given the latent variable vector :math:`X`.
        - :math:`\nabla_{\mathbf{\theta}} \ell(\mathbf{\theta}|X)` is the gradient vector of the log-likelihood with respect to :math:`\mathbf{\theta}`.
        - :math:`\nabla_{\mathbf{\theta}}^2 \ell(\mathbf{\theta}|X)` is the Hessian matrix (the second derivatives of the log-likelihood with respect to :math:`\mathbf{\theta}`).
        
        For additional details, see :cite:t:`Chang2017`.
        """
        if degrees is not None and len(degrees) != self.latent_variables:
            raise ValueError("There must be one degree for each latent variable.")

        probabilities = self.item_probabilities(theta.clone())
        gradients = self.probability_gradients(theta, rescale=rescale).detach()

        # squared gradient matrices for each latent variable
        # Uses einstein summation with batch permutation ...
        squared_grad_matrices = torch.einsum("...i,...j->...ij", gradients, gradients)
        information_matrices = squared_grad_matrices / probabilities.unsqueeze(-1).unsqueeze(-1).expand_as(squared_grad_matrices)
        information_matrices = information_matrices.nansum(dim=2) # sum over item categories

        if degrees is not None and theta.shape[1] > 1:
            cos_degrees = torch.tensor(degrees).float().deg2rad_().cos_()
            # For each theta and item: Matrix multiplication cos_degrees^T @ information_matrix @ cos_degrees
            information = torch.einsum("i,...ij,j->...", cos_degrees, information_matrices, cos_degrees)
        else:
            information = information_matrices

        if item:
            return information.detach()
        else:
            return information.detach().nansum(dim=1) # sum over items

    def item_theta_relationship_directions(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Get the directions of the relationships between each item and latent variable for a fitted model.

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor with latent theta scores from the population of interest. Each row represents one respondent, and each column represents a latent variable.

        Returns
        -------
        torch.Tensor
            A 2D tensor with the relationships between the items and latent variables. Items are rows and latent variables are columns.
        """
        item_sum_scores = self.expected_scores(theta)
        item_theta_mask = torch.zeros(self.items, self.latent_variables)
        for item, _ in enumerate(self.item_categories):
            weights = linear_regression(theta, item_sum_scores[:,item].reshape(-1, 1))[1:].reshape(-1)
            item_theta_mask[item, :] = weights.sign().int()
        
        return item_theta_mask.int()

    @torch.no_grad()
    def sample_test_data(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Sample test data for the provided theta scores.

        Parameters
        ----------
        theta : torch.Tensor
            The latent scores.

        Returns
        -------
        torch.Tensor
            The sampled test data.
        """
        probs = self.item_probabilities(theta)
        dist = torch.distributions.Categorical(probs)
        return dist.sample().float()
    
    @torch.inference_mode(False)
    def probability_gradients(self, theta: torch.Tensor, rescale: bool = True) -> torch.Tensor:
        r"""
        Calculate the gradients of the item response probabilities with respect to the theta scores.

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing latent variable theta scores. Each column represents one latent variable.
        rescale : bool, optional
            Whether to compute the gradients on the rescaled scale if it exists. Only possible for scale transformations for which gradients are available. (default is True)

        Returns
        -------
        torch.Tensor
            A torch tensor with the gradients for each theta score. Dimensions are (theta rows, items, item categories, latent variables).
        
        
        Notes
        -----
        When rescale is True, the gradients are computed on the transformed scale using the Jacobian of the transformation.
        See notes for :meth:`expected_item_score_gradients` for more details as the method is the same.
        """
        theta = theta.clone().requires_grad_(True)

        def compute_jacobian(theta_single):
            jacobian = torch.func.jacrev(self.item_probabilities)(theta_single.view(1, -1))
            return jacobian.squeeze((0, 3))

        logger.info("Computing Jacobian for all items and item categories...")
        # vectorized version of jacobian
        gradients = torch.vmap(compute_jacobian)(theta)

        if rescale and self.scale:
            gradients = self._rescale_gradients(gradients, theta)

        return gradients

    @torch.no_grad()
    def latent_scores(
        self,
        data: torch.Tensor,
        standard_errors: bool = False,
        theta_estimation: str = "ML",
        lbfgs_learning_rate: float = 0.2,
        eap_theta_integration_points: int = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        rescale: bool = True,
    ):
        r"""
        Returns the latent scores :math:`\mathbf{\theta}` for the provided test data using encoder the neural network (NN), maximum likelihood (ML), expected a posteriori (EAP) or maximum a posteriori (MAP). 
        ML and MAP uses the LBFGS algorithm. EAP and MAP are not recommended for non-variational autoencoder models as there is nothing pulling the latent distribution towards a normal.        
        EAP for models with more than three factors is not recommended since the integration grid becomes huge.

        Parameters
        ----------
        data : torch.Tensor
            A 2D tensor with test data. Each row represents one respondent, each column an item.
        standard_errors : bool, optional
            Whether to return standard errors for the latent scores. (default is False)
        theta_estimation : str, optional
            Method used to obtain the theta scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'ML')
        device: str, optional
            The device to use for computation. Can be 'cpu' or 'cuda'. (default is "cuda" if available else "cpu")
        lbfgs_learning_rate: float, optional
            For ML and MAP. The learning rate to use for the LBFGS optimizer. Try lowering this if your loss runs rampant. (default is 0.2)
        eap_theta_integration_points: int, optional
            For EAP. The number of integration points for each latent variable. (default is 'None' and uses a function of the number of latent variables)
        rescale : bool, optional
            Whether to compute the latent scores on the rescaled scale if it exists. (default is True)

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            A 2D tensor of latent scores, with latent variables as columns. If standard_errors is True, returns a tuple with the latent scores and the standard errors.
        """
        if theta_estimation not in ["NN", "ML", "EAP", "MAP"]:
            raise ValueError("Invalid theta_estimation. Choose either 'NN', 'ML', 'EAP' or 'MAP'.")
        if standard_errors and theta_estimation not in ["ML", "NN"]:
            raise ValueError("Standard errors are only available for theta scores with ML or NN estimation.")
        if not hasattr(self.algorithm, "encoder") and theta_estimation == "NN":
            raise ValueError("NN estimation is only available for autoencoder models.")

        data = data.contiguous()
        if data.dim() == 1:  # if we have only one observations
            data = data.view(1, -1)

        if theta_estimation == "EAP":
            theta = self._eap_theta_scores(data, eap_theta_integration_points)
        else:
            if hasattr(self.algorithm, "one_hot_encoded"):
                if data.isnan().any() or data.eq(-1).any():
                    if self.algorithm.imputation_method is not None:
                        encoder_data = impute_missing_internal(
                            data = data,
                            method=self.algorithm.imputation_method,
                        )
                    elif not self.algorithm.one_hot_encoded:
                        raise ValueError("The algorithm encoder does not use one-hot encoded data, and autoencoder does not have a pre-specified imputation method. Please impute beforehand.")
                    else:
                        encoder_data = data
                else:
                    encoder_data = data

                if self.algorithm.one_hot_encoded:
                    encoder_data = one_hot_encode_test_data(encoder_data, self.item_categories)
                    theta = self.algorithm.theta_scores(encoder_data).clone()
                else:
                    theta = self.algorithm.theta_scores(encoder_data).clone()
            else:
                if hasattr(self.algorithm, "training_theta_scores") and self.algorithm.training_theta_scores is not None:
                    logger.info("Finding training data theta scores to use as initial theta estimates...")
                    theta = self._initial_theta_from_training_data(data, device=device)
                else:
                    theta = torch.zeros(data.shape[0], self.latent_variables).float()

            if theta_estimation in ["ML", "MAP"]:
                theta = self._ml_map_theta_scores(data, theta, theta_estimation, learning_rate=lbfgs_learning_rate, device=device)
        
        if rescale and self.scale:
            return_theta = self.transform_theta(theta)
        else:
            return_theta = theta

        if standard_errors:
            if theta_estimation == "ML" or theta_estimation == "NN":
                if hasattr(self.algorithm, 'latent_mean_se') and theta_estimation == "NN":
                    theta_orig, se = self.algorithm.latent_mean_se(encoder_data)
                    if rescale and self.scale:
                        var = torch.diag_embed(se)**2
                        # delta method for transformed standard errors by assuming uncorrelated original thetas
                        jacobian = self.theta_transform_jacobian(theta_orig)
                        se = torch.sqrt(torch.einsum("...ij,...jk,...ik->...i", jacobian, var, jacobian))
                else:
                    fisher_info = self.information(theta, item=False, degrees=None, rescale=rescale)
                    se = 1/torch.einsum("...ii->...i", fisher_info).sqrt()
                return return_theta, se
            else:
                logger.warning("Standard errors are only implemented for theta scores with ML or NN estimation.")
        
        return return_theta

    def _initial_theta_from_training_data(self, data, device):
        try:
            self.to(device)
            data = data.to(device)
            training_thetas = self.algorithm.training_theta_scores.to(device)
            max_size = 1e6
            if data.shape[0] * training_thetas.shape[0] > max_size:
                if data.shape[0] > 5e5:
                    logger.info("Large dataset of more than half a million respondents. Setting initial theta estimates to the median training data theta scores.")
                    return training_thetas.median(dim=0).values.repeat(data.shape[0], 1)
                logger.info("Sampling training data theta scores to avoid running out of memory...")
                sample_size = int(max_size/data.shape[0])
                # sample 1000 random indices
                sampled_indices = torch.randperm(training_thetas.shape[0])[:sample_size]
                training_thetas = training_thetas[sampled_indices]

            logits = self(training_thetas)
            missing_mask = get_missing_mask(data)
            theta = training_thetas.repeat(data.shape[0], 1)
            init_data = data.repeat_interleave(training_thetas.shape[0], dim=0)
            missing_mask = missing_mask.repeat_interleave(training_thetas.shape[0], dim=0)
            
            logits = logits.repeat(data.shape[0], 1)
            lls = self.log_likelihood(init_data, logits, missing_mask, loss_reduction="none")
            lls = lls.view(logits.shape[0], -1).nansum(dim=1)
            lls = lls.view(data.shape[0], -1)
            best_theta_ind = lls.argmax(dim=1)
            best_theta_ind = torch.arange(0, data.shape[0], device=device) * training_thetas.shape[0] + best_theta_ind
            theta = theta[best_theta_ind, :]
        except Exception as e:
            logger.error("Error in initial theta estimation. Using all zeros. Error message: %s", e)
            self.to("cpu")
            return torch.zeros(data.shape[0], self.latent_variables).to("cpu")
        finally:
            self.to("cpu")
        return theta.to("cpu")
    
    def population_difficulty(self, theta: torch.Tensor) -> torch.Tensor:
        r"""
        The averge population difficulty for each item. Ranges from 0 to 1.
        
        Calculated as the average proportion of item points missing for each item as:
        
        .. math::

            \int_{\mathbf{\theta}}\left[ 1 - \frac{\mathbb{E}(x_j|\mathbf{\theta})}{\text{max}_j}\right]d\mathbf{\theta} \approx
            \frac{1}{N} \sum_{i=1}^{N} \left[1 - \frac{\mathbb{E}(x_j|\mathbf{\hat{\theta}}_i)}{\text{max}_j}\right]

        where:

        - :math:`\mathbf{\theta}` is a vector of latent variables.
        - :math:`\mathbb{E}(x_j|\mathbf{\theta})` is the expected score on item :math:`j` given :math:`\mathbf{\theta}`.
        - :math:`\text{max}_j` is the maximum score on item :math:`j`.
        - :math:`N` is the sample size and :math:`\mathbf{\hat{\theta}}_i` is the estimated :math:`\mathbf{\theta}` for respondent :math:`i`.
            
        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor with latent variable theta scores from the population of interest. Each row represents one respondent, and each column represents a latent variable.
        rescale_by_item_score : bool, optional
            Whether to rescale the expected items scores to have a max of one by dividing by the max item score. 
            This makes different item difficulties comparable. (default is True)

        Returns
        -------
        torch.Tensor
            A 1D tensor with the difficulty for each item.
        """
        item_scores = self.expected_scores(theta, return_item_scores=True)
        if self.mc_correct:
            item_scores = 1-item_scores
        else:
            item_scores = item_scores / (torch.tensor(self.item_categories) - 1)
            item_scores = 1-item_scores
        return item_scores.mean(dim=0)

    def population_discimination(self, theta: torch.Tensor, rescale: bool = True, **kwargs) -> torch.Tensor:
        r"""
        The averge population discrimination for each item. 
        Relatively large values means that an item is good at distinguishing between higher and lower ability respondents for the population supplied by the theta argument.
        
        Calculated as the average gradients of the expected item scores with respect to the latent variables scaled by the maximum item scores.
        
        .. math::

            \int_{\mathbf{\theta}}\left[ \frac{\nabla_{\mathbf{\theta}}\mathbb{E}(x_j|\mathbf{\theta})}{\text{max}_j}\right]d\mathbf{\theta} \approx
            \frac{1}{N} \sum_{i=1}^{N} \left[\frac{\nabla_{\mathbf{\theta}}\mathbb{E}(x_j|\mathbf{\hat{\theta}}_i)}{\text{max}_j}\right]

        where:

        - :math:`\mathbf{\theta}` is a vector of latent variables.
        - :math:`\mathbb{E}(x_j|\mathbf{\theta})` is the expected score on item :math:`j` given :math:`\mathbf{\theta}`.
        - :math:`\text{max}_j` is the maximum score on item :math:`j`.
        - :math:`N` is the sample size and :math:`\mathbf{\hat{\theta}}_i` is the estimated :math:`\mathbf{\theta}` for respondent :math:`i`.
            
        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor with latent variable theta scores from the population of interest. Each row represents one respondent, and each column represents a latent variable.
        rescale_by_item_score : bool, optional
            Whether to rescale the expected items scores to have a max of one by dividing by the max item score. 
            This makes different item difficulties comparable. (default is True)

        Returns
        -------
        torch.Tensor
            A 2D tensor with the average expected item score gradients. Dimensions are (items, latent_variables).
        """
        item_gradients = self.expected_item_score_gradients(theta, rescale_by_item_score=True, rescale=rescale, **kwargs)
        return item_gradients.mean(dim=0)

    @torch.no_grad()
    def _ml_map_theta_scores(
        self,
        data: torch.Tensor,
        initial_theta_scores:torch.Tensor = None,
        theta_estimation: str = "ML",
        learning_rate: float = 0.3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> torch.Tensor:
        """
        Get the latent scores from test data using an already fitted model.

        Parameters
        ----------
        data: torch.Tensor
            A 2D tensor with test data. Columns are items and rows are respondents
        initial_theta_scores: torch.Tensor
            A 2D tensor with the theta scores of the training data. Columns are latent variables and rows are respondents.
        theta_estimation: str, optional
            Method used to obtain the theta scores. Can be 'ML', 'MAP' for maximum likelihood or maximum a posteriori respectively. (default is 'ML')
        learning_rate: float, optional
            The learning rate to use for the LBFGS optimizer. (default is 0.3)
        device: str, optional
            The device to use for computation. Can be 'cpu' or 'cuda'. (default is "cuda" if available else "cpu")
        
        Returns
        -------
        torch.Tensor
            A torch.Tensor with the theta scores. The columns are latent variables and rows are respondents.
        """
        try:
            if theta_estimation == "MAP": # Approximate prior
                if hasattr(self.algorithm, "training_theta_scores") and self.algorithm.training_theta_scores is not None:
                    train_theta_scores = self.algorithm.training_theta_scores
                    # Center the data and compute the covariance matrix.
                    mean_centered_theta_scores = train_theta_scores - train_theta_scores.mean(dim=0)
                    cov_matrix = mean_centered_theta_scores.T @ mean_centered_theta_scores / (train_theta_scores.shape[0] - 1)
                    # Create prior (multivariate normal distribution).
                    prior_density = MultivariateNormal(torch.zeros(train_theta_scores.shape[1]).to(device), cov_matrix.to(device))
                elif hasattr(self.algorithm, "covariance_matrix"):
                    prior_density = MultivariateNormal(torch.zeros(self.latent_variables).to(device), self.algorithm.covariance_matrix.to(device))

            # Ensure model parameters gradients are not updated
            self.requires_grad_(False)

            if initial_theta_scores is None:
                initial_theta_scores = torch.zeros(data.shape[0], self.latent_variables).float()

            if device == "cuda":
                self.to(device)
                initial_theta_scores = initial_theta_scores.to(device)
                data = data.to(device)
                max_iter = 50
            else:
                max_iter = 40

            # Initial guess for the theta_scores are the outputs from the encoder
            optimized_theta_scores = initial_theta_scores.clone().detach().requires_grad_(True)

            optimizer = torch.optim.LBFGS([optimized_theta_scores], lr = learning_rate)
            loss_history = []
            tolerance = 1e-8

            logger.info("Optimizing theta scores...")
            missing_mask = get_missing_mask(data)
            def closure():
                optimizer.zero_grad()
                forward_output = self(optimized_theta_scores)
                if theta_estimation == "MAP": # maximize -log likelihood - log prior
                    loss = -self.log_likelihood(data, forward_output, missing_mask, loss_reduction = "sum") - prior_density.log_prob(optimized_theta_scores).sum()
                else: # maximize -log likelihood for ML
                    loss = -self.log_likelihood(data, forward_output, missing_mask, loss_reduction = "sum")
                loss.backward()
                return loss

            for i in range(max_iter):
                optimizer.step(closure)
                with torch.no_grad():
                    forward_output = self(optimized_theta_scores)
                    if theta_estimation == "MAP": # maximize -log likelihood - log prior
                        loss = -self.log_likelihood(data, forward_output, missing_mask, loss_reduction = "sum") - prior_density.log_prob(optimized_theta_scores).sum()
                    else: # maximize -log likelihood for ML
                        loss = -self.log_likelihood(data, forward_output, missing_mask, loss_reduction = "sum")
                    loss = loss.item()

                denominator = data.numel()
                dynamic_print(f"Iteration {i+1}: Current Loss = {loss}")
                if len(loss_history) > 0 and abs(loss - loss_history[-1]) / denominator < tolerance:
                    logger.info("Converged at iteration %s.", i+1)
                    break

                loss_history.append(loss)
        except Exception as e:
            logger.error("Error in %s iteration %s: %s", theta_estimation, i+1, e)
            raise e
        finally:
            # Reset requires_grad train model later
            self.to("cpu")
            optimized_theta_scores = optimized_theta_scores.detach().to("cpu")
            self.requires_grad_(True)
        return optimized_theta_scores
    
    @torch.no_grad()
    def _eap_theta_scores(self, data: torch.Tensor, grid_points: int = None) -> torch.Tensor:
        """
        Get the latent theta scores from test data using an already fitted model.

        Parameters
        ----------
        data: torch.Tensor
            A 2D tensor with test data. Columns are items and rows are respondents
        grid_points: int, optional
            The number of grid points for each latent variable. (default is 'None' and uses a function of the number of latent variables)
        
        Returns
        -------
        torch.Tensor
            A torch.Tensor with the theta scores. The columns are latent variables and rows are respondents.
        """
        if self.latent_variables > 4:
            raise ValueError("EAP is not implemented for more than 4 latent variables because of large integration grid.")
        # Get grid for integration.
        if grid_points is None:
            grid_points = {
                1: 200,
                2: 15,
                3: 7,
                4: 5
            }.get(self.latent_variables, 15)

        if hasattr(self.algorithm, "training_theta_scores"):
            if self.algorithm.training_theta_scores is None:
                raise ValueError("Please fit the model before computing latent scores.")
            train_theta_scores = self.algorithm.training_theta_scores.to("cpu")
            theta_grid = self._theta_grid(train_theta_scores, grid_size=grid_points)
            # Center the data and compute the covariance matrix.
            mean_centered_theta_scores = train_theta_scores - train_theta_scores.mean(dim=0)
            cov_matrix = mean_centered_theta_scores.T @ mean_centered_theta_scores / (train_theta_scores.shape[0] - 1)
        else:
            grid_values = torch.linspace(-3, 3, grid_points).view(-1, 1)
            grid_values = grid_values.expand(-1, self.latent_variables).contiguous()
            if self.latent_variables == 1:
                theta_grid = grid_values
            else:
                columns = [grid_values[:, i] for i in range(grid_values.size(1))]
                theta_grid = torch.cartesian_prod(*columns)

            if hasattr(self.algorithm, "covariance_matrix"):
                cov_matrix = self.algorithm.covariance_matrix
            else:
                cov_matrix = torch.eye(self.latent_variables)

        # Create prior (multivariate normal distribution).
        means = torch.zeros(self.latent_variables)
        prior_density = MultivariateNormal(means, cov_matrix)
        # Compute log of the prior.
        log_prior = prior_density.log_prob(theta_grid)

        # Compute the log likelihood.
        logits = self(theta_grid)
        replicated_data = data.repeat_interleave(theta_grid.shape[0], dim=0)
        replicated_logits = torch.cat([logits] * data.shape[0], dim=0)
        replicated_theta_grid = torch.cat([theta_grid] * data.shape[0], dim=0)
        log_prior = torch.cat([log_prior] * data.shape[0], dim=0)
        missing_mask = get_missing_mask(replicated_data)
        grid_log_likelihoods = self.log_likelihood(replicated_data, replicated_logits, missing_mask, loss_reduction = "none")
        grid_log_likelihoods = grid_log_likelihoods.view(-1, data.shape[1]).nansum(dim=1) # sum likelihood over items

        # Approximate integration integral(p(x|theta)*p(theta)dtheta)
        # p(x|theta)p(theta) / sum(p(x|theta)p(theta)) needs to sum to 1 for each respondent response pattern.
        log_posterior = (log_prior + grid_log_likelihoods).view(-1, theta_grid.shape[0]) # each row is one respondent
        # convert to float 64 to prevent 0 probabilities
        exp_log_posterior = log_posterior.to(dtype=torch.float64).exp()
        posterior = (exp_log_posterior.T / exp_log_posterior.sum(dim=1)).T.view(-1, 1) # transform to one column

        # Get expected theta
        posterior_times_theta = replicated_theta_grid * posterior
        expected_theta = posterior_times_theta.reshape(-1, theta_grid.shape[0], posterior_times_theta.shape[1])
        return expected_theta.sum(dim=1).to(dtype=torch.float32)

    @torch.no_grad()
    def _theta_grid(self, theta_scores: torch.Tensor, grid_size: int = None):
        """
        Returns a new theta score tensor covering a large range of latent variable values in a grid.
        
        Parameters
        ----------
        theta_scores: torch.Tensor
            The input test scores. Typically obtained from the training data.
        grid_size: int
            The number of grid points for each latent variable.
        
        Returns
        -------
        torch.Tensor
            A tensor with all combinations of latent variable values. Latent variables as columns.
        """
        if grid_size is None:
            grid_size = int(1e7 / (100 ** theta_scores.shape[1]))
        min_vals, _ = theta_scores.min(dim=0)
        max_vals, _ = theta_scores.max(dim=0)
        # add / remove 0.25 times the diff minus max and min in the training data
        plus_minus = (max_vals-min_vals) * 0.25
        min_vals = min_vals - plus_minus
        max_vals = max_vals + plus_minus

        # Using linspace to generate a range between min and max for each latent variable
        grids = [torch.linspace(min_val, max_val, grid_size) for min_val, max_val in zip(min_vals, max_vals)]

        # Use torch.cartesian_prod to generate all combinations of the tensors in grid
        result = torch.cartesian_prod(*grids)
        # Ensure result is always a 2D tensor even with 1 latent variable
        return result.view(-1, theta_scores.shape[1])

    def save_model(self, path: str) -> None:
        """
        Save the fitted model.

        Parameters
        -------
        path : str
            Where to save fitted model.
        """
        to_save = {
            "model_state_dict": self.state_dict(),
            "algorithm": self.algorithm,
            "scales": self.scale,
        }
        torch.save(to_save, path)

    def load_model(self, path: str) -> None:
        """
        Loads the model from a file. The initialized model should have the same structure and hyperparameter settings as the fitted model that is being loaded (e.g., the same number of latent variables).

        Parameters
        -------
        path : str
            Where to load fitted model from.
        """
        checkpoint = torch.load(path, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
        if "algorithm" in checkpoint:
            self.algorithm = checkpoint["algorithm"]
        if "scales" in checkpoint:
            self.scale = checkpoint["scales"]
