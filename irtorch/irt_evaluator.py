import logging
import torch
from irtorch.models import BaseIRTModel
from irtorch.estimation_algorithms import BaseIRTAlgorithm, VAEIRT
from irtorch.irt_scorer import QuantileMVNormal, GaussianMixtureTorch
from irtorch.irt_scorer import IRTScorer
from irtorch._internal_utils import (
    impute_missing,
    conditional_score_distribution,
    sum_incorrect_probabilities,
)

logger = logging.getLogger('irtorch')

class IRTEvaluator:
    def __init__(self, model: BaseIRTModel, algorithm: BaseIRTAlgorithm, scorer: IRTScorer):
        """
        Initializes the IRTEvaluator class.

        Parameters
        ----------
        model : BaseIRTModel
            BaseIRTModel object.
        algorithm : BaseIRTAlgorithm
            BaseIRTAlgorithm object.
        scorer : IRTScorer
            IRTScorer object used to obtain latent variable scores.
        """
        self.model = model
        self.algorithm = algorithm
        self.scorer = scorer

    def _evaluate_data_z_input(
            self,
            data: torch.Tensor = None,
            z: torch.Tensor = None,
            z_estimation_method: str = "ML",
            ml_map_device: str = "cuda" if torch.cuda.is_available() else "cpu",
            lbfgs_learning_rate: float = 0.3,
        ):
        """"
        Helper function for evaluating the data and z inputs for various performance measure methods.

        Parameters
        ----------
        data : torch.Tensor
            The input data.
        z: torch.Tensor, optional
            The latent variable z scores for the provided data. If not provided, they will be computed using z_estimation_method.
        z_estimation_method : str, optional
            Method used to obtain the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively.
        ml_map_device: str, optional
            For ML and MAP. The device to use for computation. Can be 'cpu' or 'cuda'. (default is "cuda" if available else "cpu")
        lbfgs_learning_rate: float, optional
            For ML and MAP. The learning rate to use for the LBFGS optimizer. (default is 0.3)
        """
        if data is None:
            data = self.algorithm.train_data
        else:
            data = data.contiguous()

        if not self.model.model_missing:
            data = impute_missing(data, self.model.mc_correct, self.model.item_categories)

        if z is None:
            z = self.scorer.latent_scores(data=data, scale="z", z_estimation_method=z_estimation_method, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate)

        data = self.algorithm.fix_missing_values(data)
        
        return data, z

    @torch.inference_mode()
    def residuals(
        self,
        data: torch.Tensor = None,
        z: torch.Tensor = None,
        z_estimation_method: str = "ML",
        average_over: str = "none",
    ) -> torch.Tensor:
        """
        Calculate the residuals of the model for the supplied data. 
        
        For multiple choice models, the residuals are computed as 1 - the probability of the selected response option.
        For other models, the residuals are computed as the difference between the observed and model expected item scores.

        Parameters
        ----------
        data : torch.Tensor
            The input data.
        z: torch.Tensor, optional
            The latent variable z scores for the provided data. If not provided, they will be computed using z_estimation_method.
        z_estimation_method : str, optional
            Method used to obtain the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively.
        average_over: str = "none", optional
            Whether to average the residuals and over which level. Can be 'everything', 'items', 'respondents' or 'none'. Use 'none' for no average. For example, with 'respondent' the residuals are averaged over all respondents and is thus an average per item. (default is 'none')
            
        Returns
        -------
        torch.Tensor
            The residuals.
        """
        data, z = self._evaluate_data_z_input(data, z, z_estimation_method)

        # 3D tensor with dimensions (respondents, items, item categories)
        if self.model.mc_correct is not None:
            probabilities = self.model.item_probabilities(z)
            # Creating a range tensor for slice indices
            respndents = torch.arange(probabilities.size(0)).view(-1, 1)
            # Expand slices to match the shape of indices
            expanded_respondents = respndents.expand_as(data)
            model_probs = probabilities[expanded_respondents, torch.arange(probabilities.size(1)), data.int()]
            residuals = 1 - model_probs
        else:
            residuals = data - self.model.expected_item_sum_score(z, return_item_scores=True)

        if average_over == "items":
            return residuals.mean(dim=1)
        if average_over == "respondents":
            return residuals.mean(dim=0)
        if average_over == "everything":
            return residuals.mean(dim=None)

        return residuals

    @torch.inference_mode()
    def group_fit_residuals(
        self,
        data: torch.Tensor = None,
        z: torch.Tensor = None,
        scale: str = "z",
        latent_variable: int = 1,
        standardize: bool = True,
        groups: int = 10,
        z_estimation_method: str = "ML",
        population_z: torch.Tensor = None,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Group the respondents based on their ordered latent variable scores.
        Calculate the residuals between the model estimated and observed data within each group.
        See ch. 20 in Handbook of Item Response Theory, Volume Two: Statistical Tools for more details.

        If 'data' is not supplied, the function defaults to using the model's training data.

        Parameters
        ----------
        data : torch.Tensor, optional
            A 2D tensor containing test data. Each row corresponds to one respondent and each column represents a latent variable. (default is None)
        z : torch.Tensor, optional
            A 2D tensor containing the pre-estimated z scores for each respondent in the data. Each row corresponds to one respondent and each column represents a latent variable. (default is None)
        scale : str, optional
            The grouping method scale, which can either be 'bit' or 'z'. Note: for uni-dimensional
            models, 'z' and 'bit' are equivalent. (default is 'z')
        latent_variable: int, optional
            Specifies the latent variable along which ordering and grouping should be performed. (default is 1)
        standardize : bool, optional
            Specifies whether the residuals should be standardized. (default is True)
        groups: int
            The number of groups. (default is 10)
        z_estimation_method : str, optional
            Method used to obtain the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'NN')
        population_z : torch.Tensor, optional
            Only for bit scores. The latent variable z scores for the population. If not provided, they will be computed using z_estimation_method with the model training data. (default is None)
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the bit_scores_from_z method if scale is 'bit'. See :meth:`bit_scores_from_z` for details.
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple with torch tensors. The first one holds the residuals for each group and has dimensions (groups, items, item categories). The second one is a 1D tensor and holds the mid points of the groups.
        """
        grouped_data_probabilties, grouped_model_probabilties, group_mid_points = \
            self.latent_group_probabilities(
                data=data,
                z=z,
                scale=scale,
                latent_variable=latent_variable,
                groups=groups,
                z_estimation_method=z_estimation_method,
                population_z=population_z,
                **kwargs
            )

        raw_residuals = grouped_data_probabilties - grouped_model_probabilties
        if standardize:
            # get number of people in each group
            data = torch.chunk(data, groups)
            group_n = torch.tensor([group.shape[0] for group in data])

            std_residuals = raw_residuals / (grouped_model_probabilties * (1 - grouped_model_probabilties) / group_n.reshape(-1, 1, 1)).sqrt()
            return std_residuals, group_mid_points
        
        for item, categories in enumerate(self.model.item_categories):
            # set non-existing categories to nan
            categories = categories + 1 if self.model.model_missing else categories
            raw_residuals[:, item, categories:] = float("nan")

        return raw_residuals, group_mid_points
    
    @torch.inference_mode()
    def accuracy(
        self,
        data: torch.Tensor = None,
        z: torch.Tensor = None,
        z_estimation_method: str = "ML",
        level: str = "all",
    ):
        """
        Calculate the prediction accuracy of the model for the supplied data.

        Parameters
        ----------
        data : torch.Tensor
            The input data.
        z: torch.Tensor, optional
            The latent variable z scores for the provided data. If not provided, they will be computed using z_estimation_method.
        z_estimation_method : str, optional
            Method used to obtain the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively.
        level: str = "all", optional
            Specifies the level at which the accuracy is calculated. Can be 'all', 'item' or 'respondent'. For example, for 'item' the accuracy is calculated for each item. (default is 'all')

        Returns
        -------
        torch.Tensor
            The accuracy.
        """
        data, z = self._evaluate_data_z_input(data, z, z_estimation_method)

        probabilities = self.model.item_probabilities(z)
        accuracy = (torch.argmax(probabilities, dim=2) == data).float()

        if level == "item":
            dim = 0
        elif level == "respondent":
            dim = 1
        else:
            dim = None
        
        return accuracy.mean(dim=dim)

    @torch.inference_mode()
    def log_likelihood(
        self,
        data: torch.Tensor = None,
        z: torch.Tensor = None,
        z_estimation_method: str = "ML",
        reduction: str = "sum",
        level: str = "all",
    ):
        """
        Calculate the log-likelihood for the provided data.

        If 'data' is not supplied, the function defaults to using the model's training data.

        Parameters
        ----------
        data : torch.Tensor, optional
            A 2D tensor containing test data. Each row corresponds to one respondent and each column represents a latent variable. (default is None)
        z : torch.Tensor, optional
            A 2D tensor containing latent variable z scores. Each row corresponds to one respondent and each column represents a latent variable. (default is None)
        z_estimation_method : str, optional
            Method used to obtain the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'NN')
        reduction : str, optional
            Specifies the reduction method for the log-likelihood. Can be 'sum', 'none' or 'mean'. (default is 'sum')
        level : str, optional
            For reductions other than 'none', specifies the level at which the log-likelihood is summed/averaged. Can be 'all', 'item' or 'respondent'. For example, for 'item' the log-likelihood is summed/averaged for each item. (default is 'all')
            
        Returns
        -------
        torch.Tensor
            The log-likelihood for the provided data.
        """
        data, z = self._evaluate_data_z_input(data, z, z_estimation_method)

        if reduction != "none":
            if level == "item":
                dim = 0
            elif level == "respondent":
                dim = 1
            else:
                dim = None

        likelihoods = self.model.log_likelihood(
            data,
            self.model(z),
            loss_reduction="none"
        )
        
        if reduction in "mean":
            return likelihoods.view(z.shape[0], -1).mean(dim=dim)
        if reduction == "sum":
            return likelihoods.view(z.shape[0], -1).sum(dim=dim)
        
        return likelihoods

    def information(self, z: torch.Tensor, item: bool = True, scale: str = "z", degrees: list[int] = None, **kwargs) -> torch.Tensor:
        """
        Calculate the Fisher information matrix (FIM) for the z scores (or the information in the direction supplied by degrees).

        Parameters
        ----------
        z : torch.Tensor
            A 2D tensor containing latent variable z scores for which to compute the information. Each column represents one latent variable.
        item : bool, optional
            Whether to compute the information for each item (True) or for the test as a whole (False). (default is True)
        scale: str, optional
            The grouping method scale, which can either be 'bit' or 'z'. (default is 'z')
        degrees : list[int], optional
            For multidimensional models. A list of angles in degrees between 0 and 90, one for each latent variable. Specifies the direction in which to compute the information. (default is None)
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the bit_score_gradients method if scale is 'bit'. See :meth:`bit_score_gradients` for details.
            
        Returns
        -------
        torch.Tensor
            A tensor with the information for each z score. Dimensions are:
            
            - By default: (z rows, items, FIM rows, FIM columns).
            - If degrees are specified: (z rows, items).
            - If item is False: (z rows, FIM rows, FIM columns).
            - If degrees are specified and item is False: (z rows).

        Notes
        -----
        In the context of IRT, the Fisher information matrix measures the amount of information
        that a test taker's responses :math:`X` carries about the latent variable(s)
        :math:`\\mathbf{z}`.

        The formula for the Fisher information matrix in the case of multiple parameters is:

        .. math::

            I(\\mathbf{z}) = E\\left[ \\left(\\frac{\\partial \\ell(X; \\mathbf{z})}{\\partial \\mathbf{z}}\\right) \\left(\\frac{\\partial \\ell(X; \\mathbf{z})}{\\partial \\mathbf{z}}\\right)^T \\right] = -E\\left[\\frac{\\partial^2 \\ell(X; \\mathbf{z})}{\\partial \\mathbf{z} \\partial \\mathbf{z}^T}\\right]

        Where:

        - :math:`I(\\mathbf{z})` is the Fisher Information Matrix.
        - :math:`\ell(X; \\mathbf{z})` is the log-likelihood of :math:`X`, given the latent variable vector :math:`\\mathbf{z}`.
        - :math:`\\frac{\\partial \\ell(X; \\mathbf{z})}{\\partial \\mathbf{z}}` is the gradient vector of the first derivatives of the log-likelihood of :math:`X` with respect to :math:`\\mathbf{z}`.
        - :math:`\\frac{\\partial^2 \\log f(X; \\mathbf{z})}{\\partial \\mathbf{z} \\partial \\mathbf{z}^T}` is the Hessian matrix of the second derivatives of the log-likelihood of :math:`X` with respect to :math:`\\mathbf{z}`.
        
        For additional details, see :cite:t:`Chang2017`.
        """
        if degrees is not None and len(degrees) != self.model.latent_variables:
            raise ValueError("There must be one degree for each latent variable.")

        probabilities = self.model.item_probabilities(z.clone())
        gradients = self.model.probability_gradients(z).detach()
        if scale == "bit":
            bit_z_gradients = self.scorer.bit_score_gradients(z, one_dimensional=False, **kwargs)
            # we divide by diagonal of the bit score gradients (the gradients in the direction of the bit score corresponding z scores)
            bit_z_gradients_diag = torch.einsum('...ii->...i', bit_z_gradients)
            gradients = (gradients.permute(1, 2, 0, 3) / bit_z_gradients_diag).permute(2, 0, 1, 3)

        # squared gradient matrices for each latent variable
        # Uses einstein summation with batch permutation ...
        squared_grad_matrices = torch.einsum("...i,...j->...ij", gradients, gradients)
        information_matrices = squared_grad_matrices / probabilities.unsqueeze(-1).unsqueeze(-1).expand_as(squared_grad_matrices)
        information_matrices = information_matrices.nansum(dim=2) # sum over item categories

        if degrees is not None and z.shape[1] > 1:
            cos_degrees = torch.tensor(degrees).float().deg2rad_().cos_()
            # For each z and item: Matrix multiplication cos_degrees^T @ information_matrix @ cos_degrees
            information = torch.einsum('i,...ij,j->...', cos_degrees, information_matrices, cos_degrees)
        else:
            information = information_matrices
            if self.model.latent_variables == 1:
                information.squeeze_()

        if item:
            return information
        else:
            return information.nansum(dim=1) # sum over items


    @torch.inference_mode()
    def group_fit_log_likelihood(
        self,
        data: torch.Tensor = None,
        z: torch.Tensor = None,
        z_estimation_method: str = "ML",
        groups: int = 10,
        latent_variable: int = 1,
    ):
        """
        Group the respondents based on their ordered latent variable scores.
        Calculate the average log-likelihood of the data within each group.

        If 'data' is not supplied, the function defaults to using the model's training data.

        Parameters
        ----------
        data : torch.Tensor, optional
            A 2D tensor containing test data. Each row corresponds to one respondent and each column represents a latent variable. (default is None)
        z : torch.Tensor, optional
            A 2D tensor containing the pre-estimated z scores for each respondent in the data. Each row corresponds to one respondent and each column represents a latent variable. (default is None)
        z_estimation_method : str, optional
            Method used to obtain the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'NN')
        groups: int
            The number of groups. (default is 10)
        latent_variable: int, optional
            Specifies the latent variable along which ordering and grouping should be performed. (default is 1)

        Returns
        -------
        torch.Tensor
            The average log-likelihood for each group.
        """
        data, z = self._evaluate_data_z_input(data, z, z_estimation_method)

        indicies = torch.sort(z[:, latent_variable - 1], dim=0)[1]
        z = z[indicies]
        data = data[indicies]
        likelihoods = self.model.log_likelihood(
            data,
            self.model(z),
            loss_reduction="none"
        )
        respondent_likelihoods = likelihoods.view(z.shape[0], -1).sum(dim=1)
        group_mean_likelihoods = torch.zeros(groups)
        start_index = 0
        for group in range(groups):
            group_size = respondent_likelihoods.shape[0] // groups
            remainder = len(respondent_likelihoods) % groups
            if group < remainder:
                group_size += 1
            group_mean_likelihoods[group] = respondent_likelihoods[start_index: start_index + group_size].mean()
            start_index += group_size
            
        return group_mean_likelihoods

    def latent_group_probabilities(
        self,
        data: torch.Tensor = None,
        z: torch.Tensor = None,
        scale: str = "z",
        latent_variable: int = 1,
        groups: int = 10,
        z_estimation_method: str = "ML",
        population_z: torch.Tensor = None,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Group the respondents based on their ordered latent variable scores.
        Calculate both the observed and IRT model probabilities for each possible item response, within each group.

        If 'data' is not supplied, the function defaults to using the model's training data.

        Parameters
        ----------
        data : torch.Tensor, optional
        z : torch.Tensor, optional
            The latent variable z scores for the provided data. If not provided, they will be computed using z_estimation_method. (default is None)
        scale : str, optional
            The grouping method scale, which can either be 'bit' or 'z'. Note: for uni-dimensional
            models, 'z' and 'bit' are equivalent. (default is 'z')
        latent_variable: int, optional
            Specifies the latent variable along which ordering and grouping should be performed. (default is 1)
        groups: int
            The number of groups. (default is 10)
        z_estimation_method : str, optional
            Method used to obtain the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'NN')
            A 2D tensor containing test data. Each row corresponds to one respondent and each column represents a latent variable. (default is None)
        population_z : torch.Tensor, optional
            Only for bit scores. The latent variable z scores for the population. If not provided, they will be computed using z_estimation_method with the model training data. (default is None)
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the latent_scores method if scores_to_plot is not provided. See :meth:`latent_scores` for details.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A 3D torch tensor with data group averages. The first dimension represents the groups, the second dimension represents the items and the third dimension represents the item categories.

            A 3D torch tensor with model group averages. The first dimension represents the groups, the second dimension represents the items and the third dimension represents the item categories.

            The third tensor contains the average latent variable values within each group along the specified latent_variable.
        """

        if scale not in ["bit", "z"]:
            raise ValueError("Invalid scale. Choose either 'z' or 'bit'.")

        data, z = self._evaluate_data_z_input(data, z, z_estimation_method)

        if scale == "bit":
            if population_z is None and data is self.algorithm.train_data:
                population_z = z
            
            bit_scores, _ = self.scorer.bit_scores_from_z(
                z=z,
                one_dimensional=False,
                z_estimation_method=z_estimation_method,
                **kwargs
            )

            # Sort based on correct column and get the sorted indices
            _, indices = torch.sort(
                bit_scores[:, latent_variable - 1], dim=0
            )
            # Use the indices to sort
            bit_scores = bit_scores[indices]
            z = z[indices]

            grouped_bit = torch.chunk(bit_scores, groups)
            grouped_z = torch.chunk(z, groups)

            group_mid_points = torch.tensor(
                [
                    group[:, latent_variable - 1].mean()
                    for group in grouped_bit
                ]
            )
        elif scale == "z":
            _, indices = torch.sort(z[:, latent_variable - 1], dim=0)
            z = z[indices]
            grouped_z = torch.chunk(z, groups)
            group_mid_points = torch.tensor(
                [group[:, latent_variable - 1].median() for group in grouped_z]
            )

        data = data[indices]
        grouped_data = torch.chunk(data, groups)
        grouped_data_probabilties = self._grouped_data_probabilities(grouped_data)
        grouped_model_probabilties = self._grouped_z_probabilities(grouped_z)
        return grouped_data_probabilties, grouped_model_probabilties, group_mid_points

    @torch.inference_mode()
    def _grouped_z_probabilities(self, grouped_z: tuple[torch.Tensor, ...]):
        """
        Computes the average probabilities for each potential item response for each group.

        Parameters
        ----------
        grouped_z : tuple[torch.Tensor, ...]
            A tuple containing 2D tensors. Each tensor represents a group of respondents, with the first dimension corresponding to the respondents and the second dimension representing their latent variables in the form of z-scores.

        Returns
        -------
        torch.Tensor
            A 3D torch tensor with group averages. The first dimension represents the groups, the second dimension represents the items and the third dimension represents the item categories.
        """
        group_probabilities = torch.zeros(len(grouped_z), len(self.model.modeled_item_responses), max(self.model.modeled_item_responses))
        for group_i, group in enumerate(grouped_z):
            item_probabilities = self.model.item_probabilities(group)
            group_probabilities[group_i, :, :] = item_probabilities.mean(dim=0)
        return group_probabilities

    @torch.inference_mode()
    def _grouped_data_probabilities(self, grouped_data: tuple[torch.Tensor, ...]):
        """
        Computes the average probabilities for each potential item response for each group.

        Parameters
        ----------
        grouped_data : tuple[torch.Tensor, ...]
            A tuple containing 2D tensors. Each tensor represents a group of respondents, with the first dimension representing the respondents and the second dimension representing items.

        Returns
        -------
        torch.Tensor
            A 3D torch tensor with group averages. The first dimension represents the groups, the second dimension represents the items and the third dimension represents the item categories.
        """
        modeled_item_responses = self.model.modeled_item_responses
        group_probabilities = torch.zeros(len(grouped_data), len(modeled_item_responses), max(modeled_item_responses))
        for group_i, group in enumerate(grouped_data):
            for item_i, _ in enumerate(modeled_item_responses):
                counts = torch.bincount(group[:, item_i].int(), minlength=max(modeled_item_responses))
                group_probabilities[group_i, item_i, :] = counts.float() / counts.sum()

        return group_probabilities

    @torch.inference_mode()
    def sum_score_probabilities(
        self,
        latent_density_method: str = "data",
        population_data: torch.Tensor = None,
        trapezoidal_segments: int = 1000,
        sample_size: int = 100000,
    ):
        """
        Computes the marginal probabilities for each sum score, averged over the latent space density. For 'qmvn' and 'gmm' densities, the trapezoidal rule is used for integral approximation.

        Parameters
        ----------
        latent_density_method : str, optional
            Specifies the method used to approximate the latent space density.
            Possible options are
            - 'data' averages over the z scores from the population data.
            - 'encoder sampling' samples z scores from the encoder. Only available for VariationalAutoencoderIRT models
            - 'qmvn' for quantile multivariate normal approximation of a multivariate joint density function (QuantileMVNormal class).
            - 'gmm' for an sklearn gaussian mixture model.
        population_data : torch.Tensor, optional
            The population data used for approximating sum score probabilities. Default is None and uses the training data.
        trapezoidal_segments : int, optional
            The number of integration approximation intervals for each z dimension. (Default is 1000)
        sample_size : int, optional
            Sample size for the 'encoder sampling' method. (Default is 100000)
        Returns
        -------
        torch.Tensor
            A 1D tensor with the probability for each total score.
        """
        self._validate_latent_density_method(latent_density_method)
        z_scores, weights = self._get_z_scores_and_weights(
            latent_density_method, population_data, trapezoidal_segments, sample_size
        )
        probabilities = self.model.item_probabilities(z_scores)
        # sum incorrect response option probabilities if MC
        if self.model.mc_correct is not None:
            probabilities = sum_incorrect_probabilities(
                probabilities=probabilities,
                modeled_item_responses=self.model.modeled_item_responses,
                mc_correct=self.model.mc_correct,
                missing_modeled=self.model.model_missing
            )
            item_categories = [2] * len(self.model.modeled_item_responses)
        else:
            item_categories = self.model.item_categories
            # Add together probabilities for missing response and a score of 0
            if self.model.model_missing:
                summed_slices = probabilities[:, :, 0] + probabilities[:, :, 1]
                probabilities = torch.cat((summed_slices.unsqueeze(-1), probabilities[:, :, 2:]), dim=2)
        
        conditional_total_score_probs = conditional_score_distribution(
            probabilities, item_categories
        )
        sum_score_probabilities = conditional_total_score_probs * weights.view(-1, 1)
        return sum_score_probabilities.sum(dim=0)

    def _validate_latent_density_method(self, latent_density_method: str) -> None:
        valid_methods = ["data", "encoder sampling", "qmvn", "gmm"]
        if latent_density_method not in valid_methods:
            raise ValueError(
                f"Invalid latent density method. Must be one of {valid_methods}."
            )
        if latent_density_method == "encoder sampling" and not isinstance(
            self.algorithm, VAEIRT
        ):
            raise ValueError(
                "Encoder sampling is only available for variational autoencoder models."
            )

    def _get_z_scores_and_weights(
        self,
        latent_density_method: str,
        population_data: torch.Tensor,
        trapezoidal_segments: int,
        sample_size: int,
        ml_map_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lbfgs_learning_rate: float = 0.3,
    ):
        if population_data is None:
            z_scores = self.algorithm.training_z_scores
        else:
            z_scores = self.scorer.latent_scores(population_data, scale="z", z_estimation_method="NN", ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate)

        if latent_density_method in ["data", "encoder sampling"]:
            weights = (
                torch.ones(z_scores.shape[0])
                / z_scores.shape[0]
            )
            if latent_density_method == "encoder sampling":
                z_scores = self.algorithm.sample_latent_variables(
                    sample_size=sample_size, input_data=population_data
                )
        else:
            (
                z_scores,
                weights,
            ) = self._get_z_scores_and_weights_for_latent_density_methods(
                latent_density_method, z_scores, population_data, trapezoidal_segments
            )

        return z_scores, weights

    def _get_z_scores_and_weights_for_latent_density_methods(
        self,
        latent_density_method: str,
        z_scores: torch.Tensor,
        population_data: torch.Tensor,
        trapezoidal_segments: int,
    ):
        # We approximate the density if
        # population_data is not none and the correct density
        # is not already in self.scorer.latent_density
        if population_data is not None or (
            (
                latent_density_method != "qmvn"
                or not isinstance(self.scorer.latent_density, QuantileMVNormal)
            )
            and (
                latent_density_method != "gmm"
                or not isinstance(self.scorer.latent_density, GaussianMixtureTorch)
            )
        ):
            self.scorer.approximate_latent_density(
                z_scores=z_scores, approximation=latent_density_method
            )

        # get the min/max points for integration
        min_z, max_z = self.scorer.min_max_z_for_integration(z_scores)

        # Create a list of linspace tensors for each dimension
        lin_spaces = [
            torch.linspace(min_z[i], max_z[i], trapezoidal_segments)
            for i in range(len(min_z))
        ]

        # Use torch.cartesian_prod to generate all combinations
        z_scores = torch.cartesian_prod(*lin_spaces)

        if z_scores.dim() == 1:
            # Add an extra dimension for 1D models to make it a 2D tensor with 1 column
            z_scores = z_scores.unsqueeze(1)

        weights = self.scorer.latent_density.pdf(z_scores)
        weights = weights / weights.sum()

        return z_scores, weights
