from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import torch
from torch.distributions import MultivariateNormal
from irtorch.estimation_algorithms import VAE, AE, MML
from irtorch.quantile_mv_normal import QuantileMVNormal
from irtorch.gaussian_mixture_model import GaussianMixtureModel
from irtorch._internal_utils import (
    conditional_score_distribution,
    sum_incorrect_probabilities,
    fix_missing_values
)
from irtorch.utils import impute_missing

if TYPE_CHECKING:
    from irtorch.models.base_irt_model import BaseIRTModel

logger = logging.getLogger("irtorch")

class Evaluation:
    """
    Class for evaluating IRT model performance using various metrics.

    Parameters
    ----------
    model : BaseIRTModel
        The IRT model to evaluate.
    """
    def __init__(self, model: BaseIRTModel):
        self.model = model
        self.latent_density = None

    def _evaluate_data_theta_input(
            self,
            data: torch.Tensor = None,
            theta: torch.Tensor = None,
            theta_estimation: str = "ML",
            ml_map_device: str = "cuda" if torch.cuda.is_available() else "cpu",
            lbfgs_learning_rate: float = 0.25,
        ):
        """"
        Helper function for evaluating the data and theta inputs for various performance measure methods.

        Parameters
        ----------
        data : torch.Tensor
            The input data.
        theta: torch.Tensor, optional
            The latent variable theta scores for the provided data. If not provided, they will be computed using theta_estimation.
        theta_estimation : str, optional
            Method used to obtain the theta scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively.
        ml_map_device: str, optional
            For ML and MAP. The device to use for computation. Can be 'cpu' or 'cuda'. (default is "cuda" if available else "cpu")
        lbfgs_learning_rate: float, optional
            For ML and MAP. The learning rate to use for the LBFGS optimizer. (default is 0.3)
        """
        if data is None:
            data = self.model.algorithm.train_data
        else:
            data = data.contiguous()

        if not self.model.model_missing:
            data = impute_missing(data, self.model.mc_correct, self.model.item_categories)

        if theta is None:
            theta = self.model.latent_scores(data=data, theta_estimation=theta_estimation, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate)

        if self.model.model_missing:
            data[data.isnan()] = -1
            data = data + 1

        data = fix_missing_values(data)
        
        return data, theta


    @torch.inference_mode()
    def approximate_latent_density(
        self,
        theta_scores: torch.Tensor,
        approximation: str = "qmvn",
        cv_n_components: list[int] = None,
    ) -> None:
        """
        Approximate the latent space density.

        Parameters
        ----------
        theta_scores : torch.Tensor
            A 2D tensor with theta scores. Each row represents one respondent, and each column an item.
        approximation : str, optional
            The approximation method to use. (default is 'qmvn')
            - 'qmvn' for quantile multivariate normal approximation of a multivariate joint density function (QuantileMVNormal class).
            - 'gmm' for a gaussian mixture model.

        cv_n_components: list[int], optional
            The number of components to use for cross-validation with Gaussian Mixture Models. (default is [2, 3, 4, 5, 10])

        Returns
        -------
        torch.Tensor
            A 2D tensor of latent scores, with latent variables as columns.
        """
        if approximation == "gmm":
            cv_n_components = [2, 3, 4, 5, 10] if cv_n_components is None else cv_n_components
            self.latent_density = self._cv_gaussian_mixture_model(theta_scores, cv_n_components)
        elif approximation == "qmvn":
            self.latent_density = QuantileMVNormal()
            self.latent_density.fit(theta_scores)
        else:
            raise ValueError("Invalid approximation method. Choose either 'qmvn' or 'gmm'.")

    def _cv_gaussian_mixture_model(self, data: torch.Tensor, cv_n_components: list[int]) -> GaussianMixtureModel:
        if len(cv_n_components) > 1:
            logger.info("Performing cross-validation for Gaussian Mixture Model components...")
            n_folds = 5
            average_log_likelihood = torch.zeros(len(cv_n_components))
            for comp_ind, components in enumerate(cv_n_components):
                log_likelihoods = torch.zeros(n_folds)
                data_chunks = data.chunk(n_folds)
                for i, data_val in enumerate(data_chunks):
                    train_chunks = [x for j, x in enumerate(data_chunks) if j != i]
                    data_train = torch.cat(train_chunks, dim=0)

                    gmm = GaussianMixtureModel(n_components=components, n_features=data.shape[1])
                    gmm.fit(data_train)

                    # Compute the log likelihood on the validation data
                    log_likelihoods[i] = gmm(data_val)

                # Compute the average log likelihood over the folds
                average_log_likelihood[comp_ind] = log_likelihoods.mean()

            # Select the number of components that maximizes the average log likelihood
            optimal_n_components = cv_n_components[average_log_likelihood.argmax()]
            logger.info("Fitting Gaussian Mixture Model for latent trait density.")
            gmm = GaussianMixtureModel(n_components=optimal_n_components, n_features=data.shape[1])
        else:
            logger.info("Fitting Gaussian Mixture Model for latent trait density.")
            gmm = GaussianMixtureModel(n_components=cv_n_components[0], n_features=data.shape[1])

        gmm.fit(data)
        return gmm

    @torch.inference_mode()
    def residuals(
        self,
        data: torch.Tensor = None,
        theta: torch.Tensor = None,
        theta_estimation: str = "ML",
        average_over: str = "none",
    ) -> torch.Tensor:
        """
        Compute model residuals using the supplied data. 
        
        For multiple choice models, the residuals are computed as 1 - the probability of the selected response option.
        For other models, the residuals are computed as the difference between the observed and model expected item scores.

        Parameters
        ----------
        data : torch.Tensor
            The input data.
        theta: torch.Tensor, optional
            The latent variable theta scores for the provided data. If not provided, they will be computed using theta_estimation.
        theta_estimation : str, optional
            Method used to obtain the theta scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively.
        average_over: str = "none", optional
            Whether to average the residuals and over which level. Can be 'everything', 'items', 'respondents' or 'none'. Use 'none' for no average. For example, with 'respondent' the residuals are averaged over all respondents and is thus an average per item. (default is 'none')
            
        Returns
        -------
        torch.Tensor
            The residuals.
        """
        data, theta = self._evaluate_data_theta_input(data, theta, theta_estimation)

        # 3D tensor with dimensions (respondents, items, item categories)
        if self.model.mc_correct is not None:
            probabilities = self.model.item_probabilities(theta)
            # Creating a range tensor for slice indices
            respndents = torch.arange(probabilities.size(0)).view(-1, 1)
            # Expand slices to match the shape of indices
            expanded_respondents = respndents.expand_as(data)
            model_probs = probabilities[expanded_respondents, torch.arange(probabilities.size(1)), data.int()]
            residuals = 1 - model_probs
        else:
            residuals = data - self.model.expected_scores(theta, return_item_scores=True)

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
        theta: torch.Tensor = None,
        scale: str = "theta",
        latent_variable: int = 1,
        standardize: bool = True,
        groups: int = 10,
        theta_estimation: str = "ML",
        population_theta: torch.Tensor = None,
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
        theta : torch.Tensor, optional
            A 2D tensor containing the pre-estimated theta scores for each respondent in the data. Each row corresponds to one respondent and each column represents a latent variable. (default is None)
        scale : str, optional
            The grouping method scale, which can either be 'bit' or 'theta'. Note: for uni-dimensional
            models, 'theta' and 'bit' are equivalent. (default is 'theta')
        latent_variable: int, optional
            Specifies the latent variable along which ordering and grouping should be performed. (default is 1)
        standardize : bool, optional
            Specifies whether the residuals should be standardized. (default is True)
        groups: int
            The number of groups. (default is 10)
        theta_estimation : str, optional
            Method used to obtain the theta scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'NN')
        population_theta : torch.Tensor, optional
            Only for bit scores. The latent variable theta scores for the population. If not provided, they will be computed using theta_estimation with the model training data. (default is None)
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the bit_scores_from_theta method if scale is 'bit'. See :meth:`bit_scores_from_theta` for details.
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple with torch tensors. The first one holds the residuals for each group and has dimensions (groups, items, item categories). The second one is a 1D tensor and holds the mid points of the groups.
        """
        grouped_data_probabilties, grouped_model_probabilties, group_mid_points = \
            self.latent_group_probabilities(
                data=data,
                theta=theta,
                scale=scale,
                latent_variable=latent_variable,
                groups=groups,
                theta_estimation=theta_estimation,
                population_theta=population_theta,
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
        theta: torch.Tensor = None,
        theta_estimation: str = "ML",
        level: str = "all",
    ):
        """
        Calculate the prediction accuracy of the model for the supplied data.

        Parameters
        ----------
        data : torch.Tensor
            The input data.
        theta: torch.Tensor, optional
            The latent variable theta scores for the provided data. If not provided, they will be computed using theta_estimation.
        theta_estimation : str, optional
            Method used to obtain the theta scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively.
        level: str = "all", optional
            Specifies the level at which the accuracy is calculated. Can be 'all', 'item' or 'respondent'. For example, for 'item' the accuracy is calculated for each item. (default is 'all')

        Returns
        -------
        torch.Tensor
            The accuracy.
        """
        data, theta = self._evaluate_data_theta_input(data, theta, theta_estimation)

        probabilities = self.model.item_probabilities(theta)
        accuracy = (torch.argmax(probabilities, dim=2) == data).float()

        if level == "item":
            dim = 0
        elif level == "respondent":
            dim = 1
        else:
            dim = None
        
        return accuracy.mean(dim=dim)

    @torch.inference_mode()
    def infit_outfit(
        self,
        data: torch.Tensor = None,
        theta: torch.Tensor = None,
        theta_estimation: str = "ML",
        level: str = "item",
    ):
        """
        Calculate person or item infit and outfit statistics. These statistics help identifying items that do not behave as expected according to the model
        or respondents with unusual response patterns. Items that do not behave as expectedly can be reviewed for possible revision or removal 
        to improve the overall test quality and reliability. Respondents with unusual response patterns can be reviewed for possible cheating or other issues.

        Parameters
        ----------
        data : torch.Tensor
            The input data.
        theta: torch.Tensor, optional
            The latent variable theta scores for the provided data. If not provided, they will be computed using theta_estimation.
        theta_estimation : str, optional
            Method used to obtain the theta scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively.
        level: str = "item", optional
            Specifies whether to compute item or respondent statistics. Can be 'item' or 'respondent'. (default is 'item')

        Returns
        -------
        torch.Tensor
            The infit statistics.

        Notes
        -----
        Infit and outift are computed as follows:

        .. math::
            \\begin{align}
            \\text{Item j infit} = \\frac{\\sum_{i=1}^{n} (O_{ij} - E_{ij})^2}{\\sum_{i=1}^{n} W_{ij}} \\\\
            \\text{Respondent i infit} = \\frac{\\sum_{j=1}^{J} (O_{ij} - E_{ij})^2}{\\sum_{j=1}^{J} W_{ij}} \\\\
            \\text{Item j outfit} = \\frac{\\sum_{i=1}^{n} (O_{ij} - E_{ij})^2/W_{ij}}{n} \\\\
            \\text{Respondent i outfit} = \\frac{\\sum_{j=1}^{J} (O_{ij} - E_{ij})^2/W_{ij}}{J}
            \\end{align}

        Where:

        - :math:`J` is the number of items,
        - :math:`n` is the number of respondents,
        - :math:`O_{ij}` is the observed score on the :math:`j`-th item from the :math:`i`-th respondent.
        - :math:`E_{ij}` is the expected score on the :math:`j`-th item from the :math:`i`-th respondent, calculated from the IRT model.
        - :math:`W_{ij}` is the weight on the :math:`j`-th item from the :math:`j`-th respondent. This is the variance of the item score :math:`W_{ij}=\\sum^{M_j}_{m=0}(m-E_{ij})^2P_{ijk}` where :math:`M_j` is the maximum item score and :math:`P_{ijk}` is the model probability of a score :math:`k` on the :math:`j`-th item from the :math:`i`-th respondent.
        
        """
        if level not in ["item", "respondent"]:
            raise ValueError("Invalid level. Choose either 'item' or 'respondent'.")

        data, theta = self._evaluate_data_theta_input(data, theta, theta_estimation)

        expected_scores = self.model.expected_scores(theta, return_item_scores=True)
        probabilities = self.model.item_probabilities(theta)
        observed_scores = data
        if self.model.mc_correct is not None:
            score_indices = torch.zeros(probabilities.shape[1], probabilities.shape[2])
            score_indices.scatter_(1, (torch.tensor(self.model.mc_correct) - 1 + self.model.model_missing).unsqueeze(1), 1)
            score_indices = score_indices.unsqueeze(0).expand(probabilities.shape[0], -1, -1)
            correct_probabilities = (probabilities*score_indices.int()).sum(dim=2)
            variance = correct_probabilities * (1-correct_probabilities)
            possible_scores = torch.zeros_like(probabilities)
            possible_scores[:, :, 1] = 1
            observed_scores = (data == torch.tensor(self.model.mc_correct) - 1).int()
        else:
            possible_scores = torch.arange(0, probabilities.shape[2]).unsqueeze(0).expand(probabilities.shape[0], probabilities.shape[1], -1)
            variance = ((possible_scores-expected_scores.unsqueeze(2)) ** 2 * probabilities).sum(dim=2)

        mse = (observed_scores - expected_scores) ** 2
        wmse = mse / variance
        wmse[mse == 0] = 0 # if error is 0, set to 0 in case of 0 variance to avoid nans

        if level == "item":
            infit = mse.sum(dim=0) / variance.sum(dim=0)
            outfit = wmse.mean(dim=0)
        elif level == "respondent":
            infit = mse.sum(dim=1) / variance.sum(dim=1)
            outfit = wmse.mean(dim=1)
        
        return infit, outfit

    @torch.inference_mode()
    def log_likelihood(
        self,
        data: torch.Tensor = None,
        theta: torch.Tensor = None,
        theta_estimation: str = "ML",
        reduction: str = "sum",
        level: str = "all",
    ):
        """
        Calculate the log-likelihood for the provided data.

        If 'data' is not supplied, the function defaults to using the model's training data.

        Parameters
        ----------
        data : torch.Tensor, optional
            A 2D tensor containing test data. Each row corresponds to one respondent and each column represents a latent variable. (default is None and uses the model's training data)
        theta : torch.Tensor, optional
            A 2D tensor containing latent variable theta scores. Each row corresponds to one respondent and each column represents a latent variable. If not provided, will be computed using theta_estimation. (default is None)
        theta_estimation : str, optional
            Method used to obtain the theta scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'NN')
        reduction : str, optional
            Specifies the reduction method for the log-likelihood. Can be 'sum', 'none' or 'mean'. (default is 'sum')
        level : str, optional
            For reductions other than 'none', specifies the level at which the log-likelihood is summed/averaged. Can be 'all', 'item' or 'respondent'. For example, for 'item' the log-likelihood is summed/averaged for each item. (default is 'all')
            
        Returns
        -------
        torch.Tensor
            The log-likelihood for the provided data.
        """
        data, theta = self._evaluate_data_theta_input(data, theta, theta_estimation)

        if reduction != "none":
            if level == "item":
                dim = 0
            elif level == "respondent":
                dim = 1
            else:
                dim = None

        likelihoods = self.model.log_likelihood(
            data,
            self.model(theta),
            loss_reduction="none"
        )
        
        if reduction in "mean":
            return likelihoods.view(theta.shape[0], -1).mean(dim=dim)
        if reduction == "sum":
            return likelihoods.view(theta.shape[0], -1).sum(dim=dim)
        
        return likelihoods

    @torch.inference_mode()
    def group_fit_log_likelihood(
        self,
        data: torch.Tensor = None,
        theta: torch.Tensor = None,
        theta_estimation: str = "ML",
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
        theta : torch.Tensor, optional
            A 2D tensor containing the pre-estimated theta scores for each respondent in the data. Each row corresponds to one respondent and each column represents a latent variable. (default is None)
        theta_estimation : str, optional
            Method used to obtain the theta scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'NN')
        groups: int
            The number of groups. (default is 10)
        latent_variable: int, optional
            Specifies the latent variable along which ordering and grouping should be performed. (default is 1)

        Returns
        -------
        torch.Tensor
            The average log-likelihood for each group.
        """
        data, theta = self._evaluate_data_theta_input(data, theta, theta_estimation)

        indicies = torch.sort(theta[:, latent_variable - 1], dim=0)[1]
        theta = theta[indicies]
        data = data[indicies]
        likelihoods = self.model.log_likelihood(
            data,
            self.model(theta),
            loss_reduction="none"
        )
        respondent_likelihoods = likelihoods.view(theta.shape[0], -1).sum(dim=1)
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
        theta: torch.Tensor = None,
        scale: str = "theta",
        latent_variable: int = 1,
        groups: int = 10,
        theta_estimation: str = "ML",
        population_theta: torch.Tensor = None,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Group the respondents based on their ordered latent variable scores.
        Calculate both the observed and IRT model probabilities for each possible item response, within each group.

        If 'data' is not supplied, the function defaults to using the model's training data.

        Parameters
        ----------
        data : torch.Tensor, optional
        theta : torch.Tensor, optional
            The latent variable theta scores for the provided data. If not provided, they will be computed using theta_estimation. (default is None)
        scale : str, optional
            The grouping method scale, which can either be 'bit' or 'theta'. Note: for uni-dimensional
            models, 'theta' and 'bit' are equivalent. (default is 'theta')
        latent_variable: int, optional
            Specifies the latent variable along which ordering and grouping should be performed. (default is 1)
        groups: int
            The number of groups. (default is 10)
        theta_estimation : str, optional
            Method used to obtain the theta scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'NN')
            A 2D tensor containing test data. Each row corresponds to one respondent and each column represents a latent variable. (default is None)
        population_theta : torch.Tensor, optional
            Only for bit scores. The latent variable theta scores for the population. If not provided, they will be computed using theta_estimation with the model training data. (default is None)
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the bit_scores_from_theta method.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A 3D torch tensor with data group averages. The first dimension represents the groups, the second dimension represents the items and the third dimension represents the item categories.

            A 3D torch tensor with model group averages. The first dimension represents the groups, the second dimension represents the items and the third dimension represents the item categories.

            The third tensor contains the average latent variable values within each group along the specified latent_variable.
        """

        if scale not in ["bit", "theta"]:
            raise ValueError("Invalid scale. Choose either 'theta' or 'bit'.")

        data, theta = self._evaluate_data_theta_input(data, theta, theta_estimation)

        if scale == "bit":
            if population_theta is None and data is self.model.algorithm.train_data:
                population_theta = theta
            
            bit_scores, _ = self.model.bit_scales.bit_scores_from_theta(
                theta=theta,
                one_dimensional=False,
                theta_estimation=theta_estimation,
                **kwargs
            )

            # Sort based on correct column and get the sorted indices
            _, indices = torch.sort(
                bit_scores[:, latent_variable - 1], dim=0
            )
            # Use the indices to sort
            bit_scores = bit_scores[indices]
            theta = theta[indices]

            grouped_bit = torch.chunk(bit_scores, groups)
            grouped_theta = torch.chunk(theta, groups)

            group_mid_points = torch.tensor(
                [
                    group[:, latent_variable - 1].mean()
                    for group in grouped_bit
                ]
            )
        elif scale == "theta":
            _, indices = torch.sort(theta[:, latent_variable - 1], dim=0)
            theta = theta[indices]
            grouped_theta = torch.chunk(theta, groups)
            group_mid_points = torch.tensor(
                [group[:, latent_variable - 1].median() for group in grouped_theta]
            )

        data = data[indices]
        grouped_data = torch.chunk(data, groups)
        grouped_data_probabilties = self._grouped_data_probabilities(grouped_data)
        grouped_model_probabilties = self._grouped_theta_probabilities(grouped_theta)
        return grouped_data_probabilties, grouped_model_probabilties, group_mid_points

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
            - 'data' averages over the theta scores from the population data.
            - 'encoder sampling' samples theta scores from the encoder. Only available for VariationalAutoencoderIRT models
            - 'qmvn' for quantile multivariate normal approximation of a multivariate joint density function (QuantileMVNormal class).
            - 'gmm' for a gaussian mixture model.
        population_data : torch.Tensor, optional
            The population data used for approximating sum score probabilities. Default is None and uses the training data.
        trapezoidal_segments : int, optional
            The number of integration approximation intervals for each theta dimension. (Default is 1000)
        sample_size : int, optional
            Sample size for the 'encoder sampling' method. (Default is 100000)
        Returns
        -------
        torch.Tensor
            A 1D tensor with the probability for each total score.
        """
        self._validate_latent_density_method(latent_density_method)
        theta_scores, weights = self._get_theta_scores_and_weights(
            latent_density_method, population_data, trapezoidal_segments, sample_size
        )
        probabilities = self.model.item_probabilities(theta_scores)
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

    @torch.inference_mode()
    def _min_max_theta_for_integration(
        self,
        theta: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the minimum and maximum theta score for approximating integrals over the latent space. Uses one standard deviation below/above the min/max of each theta score vector.

        Parameters
        ----------
        theta : torch.Tensor, optional
            A 2D tensor. Columns are each latent variable, rows are respondents. Default is None and uses training data theta scores.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple with 1D tensors, containing the min and max integration theta scores of each latent variable.
        """
        if theta is None:
            if isinstance(self.model.algorithm, (AE, VAE)):
                theta = self.model.algorithm.training_theta_scores
                theta_min = theta.min(dim=0)[0]
                theta_max = theta.max(dim=0)[0]
                theta_stds = theta.std(dim=0)
            elif isinstance(self.model.algorithm, MML):
                theta_min = torch.full((self.model.latent_variables,), -3)
                theta_max = torch.full((self.model.latent_variables,), 3)
                theta_stds = torch.ones(self.model.latent_variables)
        else:
            theta_min = theta.min(dim=0)[0]
            theta_max = theta.max(dim=0)[0]
            theta_stds = theta.std(dim=0)

        return theta_min - theta_stds, theta_max + theta_stds

    @torch.inference_mode()
    def _grouped_theta_probabilities(self, grouped_theta: tuple[torch.Tensor, ...]):
        """
        Computes the average probabilities for each potential item response for each group.

        Parameters
        ----------
        grouped_theta : tuple[torch.Tensor, ...]
            A tuple containing 2D tensors. Each tensor represents a group of respondents, with the first dimension corresponding to the respondents and the second dimension representing their latent variables in the form of theta-scores.

        Returns
        -------
        torch.Tensor
            A 3D torch tensor with group averages. The first dimension represents the groups, the second dimension represents the items and the third dimension represents the item categories.
        """
        group_probabilities = torch.zeros(len(grouped_theta), len(self.model.modeled_item_responses), max(self.model.modeled_item_responses))
        for group_i, group in enumerate(grouped_theta):
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

    def _validate_latent_density_method(self, latent_density_method: str) -> None:
        valid_methods = ["data", "encoder sampling", "qmvn", "gmm"]
        if latent_density_method not in valid_methods:
            raise ValueError(
                f"Invalid latent density method. Must be one of {valid_methods}."
            )
        if latent_density_method == "encoder sampling" and not isinstance(
            self.model.algorithm, VAE
        ):
            raise ValueError(
                "Encoder sampling is only available for variational autoencoder models."
            )

    def _get_theta_scores_and_weights(
        self,
        latent_density_method: str,
        population_data: torch.Tensor,
        trapezoidal_segments: int,
        sample_size: int,
        ml_map_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lbfgs_learning_rate: float = 0.25,
    ):
        if population_data is None:
            if isinstance(self.model.algorithm, (AE, VAE)):
                theta_scores = self.model.algorithm.training_theta_scores
            elif isinstance(self.model.algorithm, MML):
                logger.info("Sampling from multivariate normal as population theta scores.")
                mvn = MultivariateNormal(torch.zeros(self.model.latent_variables), self.model.algorithm.covariance_matrix)
                theta_scores = mvn.sample((4000,)).to(dtype=torch.float32)
        else:
            theta_scores = self.model.latent_scores(population_data, theta_estimation="NN", ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate)

        if latent_density_method in ["data", "encoder sampling"]:
            weights = (
                torch.ones(theta_scores.shape[0])
                / theta_scores.shape[0]
            )
            if latent_density_method == "encoder sampling":
                theta_scores = self.model.algorithm.sample_latent_variables(
                    sample_size=sample_size, input_data=population_data
                )
        else:
            (
                theta_scores,
                weights,
            ) = self._get_theta_scores_and_weights_for_latent_density_methods(
                latent_density_method, theta_scores, population_data, trapezoidal_segments
            )

        return theta_scores, weights

    def _get_theta_scores_and_weights_for_latent_density_methods(
        self,
        latent_density_method: str,
        theta_scores: torch.Tensor,
        population_data: torch.Tensor,
        trapezoidal_segments: int,
    ):
        # We approximate the density if
        # population_data is not none and the correct density
        # is not already in self.model.evaluation.latent_density
        if population_data is not None or (
            (
                latent_density_method != "qmvn"
                or not isinstance(self.latent_density, QuantileMVNormal)
            )
            and (
                latent_density_method != "gmm"
                or not isinstance(self.latent_density, GaussianMixtureModel)
            )
        ):
            self.approximate_latent_density(
                theta_scores=theta_scores, approximation=latent_density_method
            )

        # get the min/max points for integration
        min_theta, max_theta = self._min_max_theta_for_integration(theta_scores)

        # Create a list of linspace tensors for each dimension
        lin_spaces = [
            torch.linspace(min_theta[i], max_theta[i], trapezoidal_segments)
            for i in range(len(min_theta))
        ]

        # Use torch.cartesian_prod to generate all combinations
        theta_scores = torch.cartesian_prod(*lin_spaces)

        if theta_scores.dim() == 1:
            # Add an extra dimension for 1D models to make it a 2D tensor with 1 column
            theta_scores = theta_scores.unsqueeze(1)

        weights = self.latent_density.pdf(theta_scores)
        weights = weights / weights.sum()

        return theta_scores, weights