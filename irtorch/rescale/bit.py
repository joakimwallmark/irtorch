from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import torch
from torch.distributions import MultivariateNormal
from irtorch.rescale import Scale
from irtorch._internal_utils import entropy, random_guessing_data, linear_regression
from irtorch.outlier_detector import OutlierDetector
from irtorch.estimation_algorithms import AE, VAE, MML

if TYPE_CHECKING:
    from irtorch.models.base_irt_model import BaseIRTModel

logger = logging.getLogger("irtorch")

class Bit(Scale):
    """
    Class for bit scale related features.

    Parameters
    ----------
    model : BaseIRTModel
        The IRT model to use for bit scale computation.
    """
    def __init__(self, model: BaseIRTModel):
        self.model = model

    def bit_score_starting_theta(
        self,
        theta_estimation: str = "ML",
        ml_map_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lbfgs_learning_rate: float = 0.25,
        items: list[int] = None,
        start_all_incorrect: bool = False,
        train_theta: torch.Tensor = None,
        guessing_probabilities: list[float] = None,
        guessing_iterations: int = 10000,
    ):
        r"""
        Computes the starting theta score :math:`\mathbf{\theta}^{(0)}` from which to compute bit scores. See notes under :meth:`bit_scores` for more details.
        
        Parameters
        ----------
        theta_estimation : str, optional
            Method used to obtain the theta scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'ML')
        ml_map_device: str, optional
            For ML and MAP. The device to use for computation. Can be 'cpu' or 'cuda'. (default is "cuda" if available else "cpu")
        lbfgs_learning_rate: float, optional
            For ML and MAP. The learning rate to use for the LBFGS optimizer. (default is 0.3)
        items: list[int], optional
            The item indices for the items to use to compute the bit scores. (default is None and uses all items)
        start_all_incorrect: bool, optional
            Whether to compute the starting theta scores based incorrect responses. If false, starting theta is computed based on relationships between each latent variable and the item responses. (default is False)
        train_theta : torch.Tensor, optional
            A 2D tensor with the training data theta scores. Used to estimate relationships between theta and getting the items correct when start_all_incorrect is False. Columns are latent variables and rows are respondents. (default is None and uses encoder theta scores from the model training data)
        guessing_probabilities: list[float], optional
            The guessing probability for each item. The same length as the number of items. Guessing is not supported for polytomously scored items and the probabilities for them will be ignored. (default is None and uses no guessing or, for multiple choice models, 1 over the number of item categories)
        guessing_iterations: int, optional
            The number of iterations to use for approximating a minimum theta when guessing is incorporated. (default is 200)
        
        Returns
        -------
        torch.Tensor
            A tensor with all the starting theta values.
        """
        items = items or list(range(len(self.model.item_categories)))
        mc_correct = torch.tensor(self.model.mc_correct) if self.model.mc_correct else None
        if not all(isinstance(item, int) for item in items):
            raise ValueError("items must be a list of integers.")
        
        if guessing_probabilities:
            if len(guessing_probabilities) != len(items) or not all(0 <= num < 1 for num in guessing_probabilities):
                raise ValueError("guessing_probabilities must be list of floats with the same length as the number of items and values between 0 and 1.")

        selected_item_categories = [self.model.item_categories[i] for i in items]

        if guessing_probabilities is None and mc_correct is not None:
            guessing_probabilities = [1 / categories for categories in selected_item_categories]

        if not start_all_incorrect:
            if train_theta is None:
                if isinstance(self.model.algorithm, (AE, VAE)):
                    train_theta = self.model.algorithm.training_theta_scores
                elif isinstance(self.model.algorithm, MML):
                    mvn = MultivariateNormal(torch.zeros(self.model.latent_variables), self.model.algorithm.covariance_matrix)
                    train_theta = mvn.sample((4000,)).to(dtype=torch.float32)

            # Which latent variables are inversely related to the test scores?
            item_sum_scores = self.model.expected_scores(train_theta, return_item_scores=False)
            test_weights = linear_regression(train_theta, item_sum_scores.reshape(-1, 1))[1:]
            inverted_scale = torch.where(test_weights < 0, torch.tensor(-1), torch.tensor(1)).reshape(-1)
            
            # Which latent variables are positively related to the item scores?
            directions = self.model.item_theta_relationship_directions(train_theta)
            item_theta_positive = (inverted_scale * directions) >= 0 # Invert item relationship if overall test relationship is inverted

        if guessing_probabilities is None:
            if start_all_incorrect:
                logger.info("Computing theta for all incorrect responses to get minimum bit score.")
                starting_theta = self.model.latent_scores(torch.zeros((1, len(items))).float(), theta_estimation=theta_estimation, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate)
            else:
                # Get minimum score in relation to each latent variable
                min_sum_score = torch.zeros((len(items), self.model.latent_variables))
                min_sum_score[~item_theta_positive] = (torch.tensor(selected_item_categories) - 1).view(-1, 1).float().repeat(1, self.model.latent_variables)[~item_theta_positive]

                # get the minimum theta scores based on the sum scores
                starting_theta = torch.zeros((1, self.model.latent_variables)).float()
                for theta in range(self.model.latent_variables):
                    logger.info("Computing minimum bit score theta for latent variable %s.", theta+1)
                    starting_theta[:, theta] = self.model.latent_scores(min_sum_score.float()[:, theta], theta_estimation=theta_estimation, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate)[:, theta]
        else:
            if mc_correct is None:
                selected_correct = torch.ones(len(items))
                random_data = random_guessing_data(selected_item_categories, guessing_iterations, guessing_probabilities)
            else:
                selected_correct = mc_correct.index_select(0, torch.tensor(items))
                random_data = random_guessing_data(
                    selected_item_categories,
                    guessing_iterations,
                    guessing_probabilities,
                    selected_correct
                )

            if start_all_incorrect:
                # With one-dimensional bit scores, guessing for all items makes sense
                logger.info("Approximating theta from random guessing data to get minimum bit score.")
                guessing_theta = self.model.latent_scores(random_data, theta_estimation=theta_estimation, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate)
            else:
                # Random for positively related and set to correct for others
                guessing_theta = torch.zeros(random_data.shape[0], self.model.latent_variables)
                for theta in range(self.model.latent_variables):
                    random_data_theta = random_data.clone()
                    random_data_theta[:, ~item_theta_positive[:, theta]] = selected_correct[~item_theta_positive[:, theta]].float() - 1
                    logger.info("Approximating minimum bit score theta from random guessing data for latent variable %s.", theta+1)
                    guessing_theta[:, theta] = self.model.latent_scores(random_data_theta, theta_estimation=theta_estimation, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate)[:, theta]

            starting_theta = guessing_theta.detach().median(dim=0).values.reshape(1, self.model.latent_variables)

        return starting_theta


    @torch.inference_mode()
    def transform(
        self,
        theta: torch.Tensor,
        start_theta: torch.Tensor = None,
        population_theta: torch.Tensor = None,
        one_dimensional: bool = False,
        theta_estimation: str = "ML",
        ml_map_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lbfgs_learning_rate: float = 0.25,
        grid_points: int = 300,
        items: list[int] = None,
        start_theta_guessing_probabilities: list[float] = None,
        start_theta_guessing_iterations: int = 10000,
        return_start_theta: bool = False,
    ) -> torch.Tensor:
        r"""
        Transforms :math:`\mathbf{\theta}` scores into bit scores :math:`B(\mathbf{\theta})`.
        
        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor. Columns are latent variables and rows are respondents.
        start_theta : torch.Tensor, optional
            A one row 2D tensor with bit score starting values of each latent variable. Estimated automatically if not provided. (default is None)
        population_theta : torch.Tensor, optional
            A 2D tensor with theta scores of the population. Used to estimate relationships between each theta and sum scores. Columns are latent variables and rows are respondents. (default is None and uses theta_estimation with the model training data)
        one_dimensional: bool, optional
            Whether to estimate one combined bit score for a multidimensional self.model. (default is True)
        theta_estimation : str, optional
            Method used to obtain the theta score grid for bit score computation. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'ML')
        ml_map_device: str, optional
            For ML and MAP. The device to use for computation. Can be 'cpu' or 'cuda'. (default is "cuda" if available else "cpu")
        lbfgs_learning_rate: float, optional
            For ML and MAP. The learning rate to use for the LBFGS optimizer. (default is 0.3)
        grid_points : int, optional
            The number of points to use for computing bit score. More steps lead to more accurate results. (default is 300)
        items: list[int], optional
            The item indices for the items to use to compute the bit scores. (default is None and uses all items)
        start_theta_guessing_probabilities: list[float], optional
            The guessing probability for each item if start_theta is None. The same length as the number of items. Guessing is not supported for polytomously scored items and the probabilities for them will be ignored. (default is None and uses no guessing or, for multiple choice models, 1 over the number of item categories)
        start_theta_guessing_iterations: int, optional
            The number of iterations to use for approximating a minimum theta when guessing is incorporated. (default is 10000)
        return_start_theta: bool, optional
            Whether to return the starting theta scores. (default is False)
        
        Returns
        -------
        torch.Tensor
            A 2D tensor with bit score scale scores for each respondent across the rows together with another tensor with start_theta.
            If return_start_theta is True, the function returns a tuple with the bit scores and the starting theta scores.

        Notes
        -----
        First, item bit scores for each item :math:`j` are computed from :math:`\mathbf{\theta}` scores as follows:

        .. math ::

            \begin{equation}
                \begin{aligned}
                    B_j(\mathbf{\theta})=
                    \int_{t=\mathbf{\theta}^{(0)}}^{\mathbf{\theta}}
                    \left|\frac{dH_j(t)}{dt}\right| dt.
                \end{aligned}
            \end{equation}

        where

        - :math:`\mathbf{\theta}^{(0)}` is the minimum :math:`\mathbf{\theta}`
        - :math:`H(\mathbf{\theta})` is entropy for item :math:`j` as a function of :math:`\mathbf{\theta}`
            
        The total bit scores :math:`B(\mathbf{\theta})` are then the sum of the item scores:

        .. math ::

            \begin{equation}
                \begin{aligned}
                    B(\mathbf{\theta}) = \sum_{j=1}^{J} B_j(\mathbf{\theta}).
                \end{aligned}
            \end{equation}
        """
        if grid_points <= 0:
            raise ValueError("steps must be a positive integer")
        if start_theta is not None and start_theta.shape != (1, self.model.latent_variables):
            raise ValueError(f"start_theta must be a one row tensor with shape (1, {self.model.latent_variables}).")
        if theta_estimation not in ["NN", "ML", "EAP", "MAP"]:
            raise ValueError("Invalid bit_score_theta_grid_method. Choose either 'NN', 'ML', 'EAP' or 'MAP'.")
        if items is None:
            items = list(range(len(self.model.item_categories)))
        elif not isinstance(items, list) or not all(isinstance(item, int) for item in items):
            raise ValueError("items must be a list of integers.")
        
        if population_theta is None:
            if theta_estimation != "NN":
                logger.info("Estimating population theta scores needed for bit score computation.")
                population_theta = self.model.latent_scores(self.model.algorithm.train_data, theta_estimation=theta_estimation, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate)
            else:
                if isinstance(self.model.algorithm, (AE, VAE)):
                    logger.info("Using traning data theta scores as population theta scores for bit score computation.")
                    population_theta = self.model.algorithm.training_theta_scores
                elif isinstance(self.model.algorithm, MML):
                    logger.info("Sampling from multivariate normal as population theta scores for bit score computation.")
                    mvn = MultivariateNormal(torch.zeros(self.model.latent_variables), self.model.algorithm.covariance_matrix)
                    population_theta = mvn.sample((4000,)).to(dtype=torch.float32)

        if start_theta is None:
            start_theta = self.bit_score_starting_theta(
                theta_estimation=theta_estimation,
                ml_map_device=ml_map_device,
                lbfgs_learning_rate=lbfgs_learning_rate,
                items=items,
                start_all_incorrect=one_dimensional,
                train_theta=population_theta,
                guessing_probabilities=start_theta_guessing_probabilities,
                guessing_iterations=start_theta_guessing_iterations,
            )

        inverted_scales = self._inverted_scales(population_theta)
        theta_adjusted, train_theta_adjusted, start_theta_adjusted = self._anti_invert_and_adjust_theta_scores(theta, population_theta, start_theta, inverted_scales)
        grid_start, grid_end, _ = self._get_grid_boundaries(train_theta_adjusted, start_theta_adjusted)
        
        if one_dimensional:
            bit_scores = self._compute_1d_bit_scores(
                theta_adjusted,
                start_theta_adjusted,
                grid_start,
                grid_end,
                inverted_scales,
                grid_points
            )
        else:
            bit_scores = self._compute_multi_dimensional_bit_scores(
                theta_adjusted,
                start_theta_adjusted,
                train_theta_adjusted,
                grid_start,
                grid_end,
                inverted_scales,
                grid_points
            )
        
        if return_start_theta:
            return bit_scores, start_theta
        
        return bit_scores

    @torch.inference_mode(False)
    def gradients(
        self,
        theta: torch.Tensor,
        h: float = None,
        independent_theta: int = None,
        start_theta: torch.Tensor = None,
        population_theta: torch.Tensor = None,
        one_dimensional: bool = False,
        theta_estimation: str = "ML",
        ml_map_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lbfgs_learning_rate: float = 0.3,
        grid_points: int = 300,
        items: list[int] = None,
        start_theta_guessing_probabilities: list[float] = None,
        start_theta_guessing_iterations: int = 10000,
    ) -> torch.Tensor:
        r"""
        For unidimensional models: Computes the gradients of the bit scores for each :math:`j` with respect to the input theta scores

        .. math ::

            \frac{\partial B(\mathbf{\theta})}{\partial \mathbf{\theta}} = 
            \sum_{j=1}^J\frac{\partial B_j(\mathbf{\theta})}{\partial \mathbf{\theta}} = 
            \sum_{j=1}^J\frac{\partial \int_{t=\mathbf{\theta}^{(0)}}^{\mathbf{\theta}}\left|\frac{dH_j(t)}{dt}\right| dt}{\partial \mathbf{\theta}}=
            \sum_{j=1}^J\left|\frac{\partial H_j(\mathbf{\theta})}{\partial \mathbf{\theta}}\right|,

        where 

        - :math:`\mathbf{\theta}^{(0)}` is the minimum :math:`\mathbf{\theta}`
        - :math:`H_j(\mathbf{\theta})` is entropy for item :math:`j` as a function of :math:`\mathbf{\theta}`

        For total bit score gradients, the gradients for each item can be summed.
        
        For multidimmensional models: Approximates the gradients of the total bit scores with respect to the input theta scores using the central difference method:

        .. math ::

            \frac{\partial B(\mathbf{\theta})}{\partial \mathbf{\theta}} \approx \frac{f(B(\mathbf{\theta})+h)-f(B(\mathbf{\theta})-h)}{2 h}

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing latent variable theta scores. Each column represents one latent variable.
        h : float, optional
            The step size for the central difference method. (default uses the difference between the smaller and upper outlier limits (computed using the interquantile range rule) of the training theta scores divided by 1000)
        independent_theta : int, optional
            The latent variable to differentiate with respect to. (default is None and computes gradients with respect to theta)
        start_theta : torch.Tensor, optional
            A one row 2D tensor with bit score starting values of each latent variable. Estimated automatically if not provided. (default is None)
        population_theta : torch.Tensor, optional
            A 2D tensor with theta scores of the population. Used to estimate relationships between each theta and sum scores. Columns are latent variables and rows are respondents. (default is None and uses theta_estimation with the model training data)
        one_dimensional: bool, optional
            Whether to estimate one combined bit score for a multidimensional models. (default is True)
        theta_estimation : str, optional
            Method used to obtain the theta score grid for bit score computation. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'ML')
        ml_map_device: str, optional
            For ML and MAP. The device to use for computation. Can be 'cpu' or 'cuda'. (default is "cuda" if available else "cpu")
        lbfgs_learning_rate: float, optional
            For ML and MAP. The learning rate to use for the LBFGS optimizer. (default is 0.3)
        grid_points : int, optional
            The number of points to use for computing bit score. More steps lead to more accurate results. (default is 300)
        items: list[int], optional
            The item indices for the items to use to compute the bit scores. (default is None and uses all items)
        start_theta_guessing_probabilities: list[float], optional
            The guessing probability for each item if start_theta is None. The same length as the number of items. Guessing is not supported for polytomously scored items and the probabilities for them will be ignored. (default is None and uses no guessing or, for multiple choice models, 1 over the number of item categories)
        start_theta_guessing_iterations: int, optional
            The number of iterations to use for approximating a minimum theta when guessing is incorporated. (default is 10000)

        Returns
        -------
        torch.Tensor
            A torch tensor with the gradients for each theta score. Dimensions are (theta rows, latent variables, latent variables) where the last two are the jacobians.
            If independent_theta is provided, the tensor has dimensions (theta rows, latent variables).
        """
        if self.model.latent_variables == 1:
            if theta.requires_grad:
                theta.requires_grad_(False)

            gradients = torch.zeros(theta.shape[0], len(self.model.item_categories), theta.shape[1])
            theta_scores = theta.clone()
            theta_scores.requires_grad_(True)
            probs = self.model.item_probabilities(theta_scores)
            entropies = entropy(probs)

            # gradient for each item
            for item in range(entropies.shape[1]):
                if theta_scores.grad is not None:
                    theta_scores.grad.zero_()
                entropies[:, item].sum().backward(retain_graph=True)
                for latent_variable in range(theta.shape[1]):
                    gradients[:, item, latent_variable] = theta_scores.grad[:, latent_variable]

            gradients[gradients.isnan()] = 0.
            # sum over items and add dimension for jacobian
            return gradients.abs().sum(dim=1).unsqueeze(dim=-1)
        
        if hasattr(self.model.algorithm, "training_theta_scores") and self.model.algorithm.training_theta_scores is not None:
            q1_q3 = torch.quantile(self.model.algorithm.training_theta_scores, torch.tensor([0.25, 0.75]), dim=0)
        else:
            q1_q3 = torch.tensor([-0.6745, 0.6745]).unsqueeze(1).expand(-1, self.model.latent_variables)

        if start_theta is None:
            start_theta = self.bit_score_starting_theta(
                theta_estimation=theta_estimation,
                items=items,
                start_all_incorrect=one_dimensional,
                train_theta=population_theta,
                guessing_probabilities=start_theta_guessing_probabilities,
                guessing_iterations=start_theta_guessing_iterations,
            )
        iqr = q1_q3[1] - q1_q3[0]
        lower_bound = q1_q3[0] - 1.5 * iqr
        upper_bound = q1_q3[1] + 1.5 * iqr
        h = (upper_bound - lower_bound) / 1000
        theta_low = theta-h
        theta_high = theta+h
        if independent_theta is None:
            gradients = torch.zeros(theta.shape[0],theta.shape[1],theta.shape[1])
            for latent_variable in range(theta.shape[1]):
                theta_low_var = torch.cat((theta[:, :latent_variable], theta_low[:, latent_variable].view(-1, 1), theta[:, latent_variable+1:]), dim=1)
                theta_high_var = torch.cat((theta[:, :latent_variable], theta_high[:, latent_variable].view(-1, 1), theta[:, latent_variable+1:]), dim=1)
                bit_scores_low = self(
                    theta_low_var, start_theta = start_theta, population_theta=population_theta, one_dimensional=one_dimensional, theta_estimation=theta_estimation,
                    ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate, grid_points=grid_points, items=items
                )
                bit_scores_high = self(
                    theta_high_var, start_theta = start_theta, population_theta=population_theta, one_dimensional=one_dimensional, theta_estimation=theta_estimation,
                    ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate, grid_points=grid_points, items=items
                )
                gradients[:, latent_variable, :] = (bit_scores_high - bit_scores_low) / (2 * h[latent_variable])
        else:
            theta_low_var = torch.cat((theta[:, :independent_theta-1], theta_low[:, independent_theta-1].view(-1, 1), theta[:, independent_theta:]), dim=1)
            theta_high_var = torch.cat((theta[:, :independent_theta-1], theta_high[:, independent_theta-1].view(-1, 1), theta[:, independent_theta:]), dim=1)
            bit_scores_low = self(
                theta_low_var, start_theta = start_theta, population_theta=population_theta, one_dimensional=one_dimensional, theta_estimation=theta_estimation,
                ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate, grid_points=grid_points, items=items
            )
            bit_scores_high = self(
                theta_high_var, start_theta = start_theta, population_theta=population_theta, one_dimensional=one_dimensional, theta_estimation=theta_estimation,
                ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate, grid_points=grid_points, items=items
            )
            gradients = (bit_scores_high - bit_scores_low) / (2 * h[independent_theta-1])

        return gradients

    @torch.inference_mode(False)
    def expected_item_score_slopes(
        self,
        theta: torch.Tensor,
        bit_scores: torch.Tensor = None,
        rescale_by_item_score: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        For unidimensional models: Computes the derivatives of the expected item scores with respect to the bit scores, averaged over the provided sample of theta scores.
        
        For multidimensional models: Computes the linear regression slopes of each latent variable with the expected item scores as response variables. 

        Parameters
        ----------
        theta : torch.Tensor, optional
            A 2D tensor with latent theta scores from the population of interest. Each row represents one respondent, and each column represents a latent variable. If not provided, uses the training theta scores. (default is None)
        bit_scores: torch.Tensor, optional
            A 2D tensor with bit scores corresponding to the theta scores. If not provided, computes the bit scores from the theta scores. (default is None)
        rescale_by_item_score : bool, optional
            Whether to rescale the expected items scores to have a max of one by dividing by the max item score. (default is True)
        **kwargs
            Additional keyword arguments for the :meth:`gradients` method.

        Returns
        -------
        torch.Tensor
            A 3D tensor with the expected item score gradients. The first dimension corresponds to the rows in the supplied theta, second is the item, and third is the latent variable.
        """
        if self.model.latent_variables > 1:
            if theta.requires_grad:
                theta.requires_grad_(False)
            expected_item_sum_scores = self.model.expected_scores(theta, return_item_scores=True).detach()
            if not self.model.mc_correct and rescale_by_item_score:
                expected_item_sum_scores = expected_item_sum_scores / (torch.tensor(self.model.item_categories) - 1)
            if bit_scores is None:
                bit_scores = self(theta)
            # item score slopes for each item
            gradients = linear_regression(bit_scores, expected_item_sum_scores).t()[:, 1:]
            # median, _ = torch.median(theta, dim=0)
            # mean_slopes = torch.zeros(theta.shape[0], len(self.model.item_categories),theta.shape[1])
            # for latent_variable in range(theta.shape[1]):
            #     theta_scores = median.repeat(theta.shape[0], 1)
            #     theta_scores[:, latent_variable], _ = theta[:, latent_variable].sort()
            #     theta_scores.requires_grad_(True)
            #     expected_item_sum_scores = self.model.expected_scores(theta_scores, return_item_scores=True)
            #     if not self.model.mc_correct and rescale_by_item_score:
            #         expected_item_sum_scores = expected_item_sum_scores / (torch.tensor(self.model.item_categories) - 1)

            #     # item score slopes for each item
            #     for item in range(expected_item_sum_scores.shape[1]):
            #         if theta_scores.grad is not None:
            #             theta_scores.grad.zero_()
            #         expected_item_sum_scores[:, item].sum().backward(retain_graph=True)
            #         gradients[:, item, latent_variable] = theta_scores.grad[:, latent_variable]
        else:
            dx_dtheta = self.model.expected_item_score_gardients(theta, rescale_by_item_score)
            # sum over items to get dB(theta)/dtheta from dB_j(theta)/dtheta
            dbit_dtheta = self.gradients(theta, **kwargs).sum(dim=1)
            if theta.shape[0] > 1:
                inverted_scales = self._inverted_scales(theta)
            else:
                inverted_scales = torch.tensor([1.])

            # divide by the derivative of the bit scores with respect to the theta scores
            # to get the expected item score slopes with respect to the bit scores (chain rule)
            gradients = inverted_scales.flatten() * dx_dtheta/dbit_dtheta.unsqueeze(1)
            gradients[gradients.isnan()] = 0. # set nans to 0. Sometimes 0/0 can be nan
            gradients[gradients == torch.inf] = 0. # Remove dx_dtheta/0

        return gradients

    def information(self, theta: torch.Tensor, item: bool = True, degrees: list[int] = None, **kwargs) -> torch.Tensor:
        r"""
        Calculate the Fisher information matrix (FIM) for the theta corresponding bit scores (or the information in the direction supplied by degrees).

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing latent variable theta scores for which to compute the information. Each column represents one latent variable.
        item : bool, optional
            Whether to compute the information for each item (True) or for the test as a whole (False). (default is True)
        degrees : list[int], optional
            For multidimensional models. A list of angles in degrees between 0 and 90, one for each latent variable. Specifies the direction in which to compute the information. (default is None and returns the full FIM)
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the :meth:`gradients` method.
            
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
        The Fisher information matrix measures the amount of information
        that a test taker's responses :math:`X` carries about the bit scale transformed latent variable(s)
        :math:`B(\mathbf{\theta})`.

        .. math::

            I(B(\mathbf{\theta})) = E\left[ \left(\frac{\partial \ell(B(\mathbf{\theta})|X)}{\partial B(\mathbf{\theta})}\right) \left(\frac{\partial \ell(B(\mathbf{\theta})|X)}{\partial B(\mathbf{\theta})}\right)^T \right] = -E\left[\frac{\partial^2 \ell(B(\mathbf{\theta})|X)}{\partial B(\mathbf{\theta}) \partial B(\mathbf{\theta})^T}\right]

        Where:

        - :math:`I(B(\mathbf{\theta}))` is the Fisher Information Matrix.
        - :math:`\ell(B(\mathbf{\theta})|X)` is the log-likelihood of :math:`B(\mathbf{\theta})`, given the latent variable vector :math:`X`.
        - :math:`\frac{\partial \ell(B(\mathbf{\theta})|X)}{\partial B(\mathbf{\theta})}` is the gradient vector of the log-likelihood with respect to :math:`B(\mathbf{\theta})`.
        - :math:`\frac{\partial^2 \log f(B(\mathbf{\theta})|X)}{\partial B(\mathbf{\theta}) \partial B(\mathbf{\theta})^T}` is the Hessian matrix (the second derivatives of the log-likelihood with respect to :math:`B(\mathbf{\theta})`).
        
        For additional details, see :cite:t:`Chang2017`.
        """
        if degrees is not None and len(degrees) != self.model.latent_variables:
            raise ValueError("There must be one degree for each latent variable.")

        probabilities = self.model.item_probabilities(theta.clone())
        gradients = self.model.probability_gradients(theta).detach()
        bit_theta_gradients = self.gradients(theta, one_dimensional=False, **kwargs)
        # we divide by diagonal of the bit score gradients (the gradients in the direction of the bit score corresponding theta scores)
        bit_theta_gradients_diag = torch.einsum("...ii->...i", bit_theta_gradients)
        gradients = (gradients.permute(1, 2, 0, 3) / bit_theta_gradients_diag).permute(2, 0, 1, 3)

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
            return information
        else:
            return information.nansum(dim=1) # sum over items

    def _compute_1d_bit_scores(self, theta_adjusted, start_theta_adjusted, grid_start, grid_end, inverted_scale, grid_points):
        """
        Computes the 1D bit scores for the given input theta scores.

        Parameters
        ----------
        theta_adjusted : torch.Tensor
            The adjusted theta-scores. A 2D tensor.
        start_theta_adjusted : torch.Tensor
            The adjusted theta-scores for the starting point. A one row tensor.
        grid_start : torch.Tensor
            The start of the grid. A one row tensor.
        grid_end : torch.Tensor
            The end of the grid. A one row tensor.
        inverted_scale : torch.Tensor
            The inverted scale. A one row tensor with -1 for latent variables inversely related to the test scores and 1 for others.
        grid_points : int
            The number of grid points.

        Returns
        -------
        torch.Tensor
            The computed 1D bit scores.
        """
        # Limit all theta_scores to be within the grid range
        theta_adjusted_capped = torch.clamp(theta_adjusted, grid_start, grid_end)
        
        grid_linespace = torch.linspace(0, 1, steps=grid_points).unsqueeze(-1)
        # Use broadcasting to compute the grid
        grid = grid_start + (grid_linespace * (theta_adjusted_capped - grid_start).unsqueeze(1))

        # set the entire grid for those below the grid to their theta scores
        below_grid_start_mask = theta_adjusted < grid_start
        grid = torch.where(below_grid_start_mask.unsqueeze(1), theta_adjusted.unsqueeze(1), grid)

        # Ensure all values are larger than start_theta_adjusted
        grid = torch.maximum(grid, start_theta_adjusted)

        # set the last slot in each grid to the outliers theta scores
        grid[:, -1, :] = torch.where(theta_adjusted > grid_end, theta_adjusted, grid[:, -1])
        # set the first slot in each grid to the start_theta_adjusted (required when theta_adjusted is an outlier)
        grid[:, 0, :] = start_theta_adjusted.squeeze()

        # convert back to non-inverted theta scale and compute grid entropies
        grid = grid.view(-1, grid.shape[2]) * inverted_scale
        output = self.model(grid)
        entropies = entropy(self.model.probabilities_from_output(output))

        # the goal is to get the entropies to the dimensions required for bit_score_distance
        # we need to transpose for view to order them correctly
        # and change positioning of each dimension in the end using permute
        entropies = (
            entropies.t()
            .view(entropies.shape[1], int(entropies.shape[0] / grid_points), grid_points)
            .permute(1, 0, 2)
        )

        # Compute the absolute difference
        diff = torch.abs(entropies - torch.roll(entropies, shifts=1, dims=2))
        # Note that for each sub-tensor in the third dimension, the first element will
        # be the difference with the last element of the previous sub-tensor because
        # of the roll function. We set this to 0
        diff[:, :, 0] = 0
        return diff.sum(dim=(1, 2)).unsqueeze(1)

    def _compute_multi_dimensional_bit_scores(self, theta_adjusted, start_theta_adjusted, train_theta_adjusted, grid_start, grid_end, inverted_scale, grid_points):
        """
        Computes the multi-dimensional bit scores for the given input theta scores.

        Parameters
        -----------
        theta_adjusted : torch.Tensor
            The input theta scores.
        start_theta_adjusted : torch.Tensor
            The minimum theta score to be used in the grid. A one row tensor.
        train_theta_adjusted : torch.Tensor
            The theta scores of the training data. Used for computing the median of each latent variable.
        grid_start : float
            The minimum value of the grid. A one row tensor.
        grid_end : float
            The maximum value of the grid. A one row tensor.
        inverted_scale : torch.Tensor
            The inverted scale. A one row tensor with -1 for latent variables inversely related to the test scores and 1 for others.
        grid_points : int
            The number of points in the grid.

        Returns
        --------
        bit_scores : torch.Tensor
            The multi-dimensional bit scores for the input theta scores.
        """
        # Construct grid
        ratio = torch.linspace(0, 1, steps=grid_points).view(-1, 1)
        grid = (1 - ratio) * grid_start + ratio * grid_end
        # Add theta scores to the grid and sort the columns
        grid = torch.cat([grid, theta_adjusted], dim=0)
        # set all theta scores smaller than start_theta_adjusted to start_theta_adjusted
        grid = torch.max(grid, start_theta_adjusted)
        # set the first slot in each grid to the start_theta_adjusted
        # required when any value in theta_adjusted is an outlier
        grid[0, :] = start_theta_adjusted

        grid, sorted_indices = torch.sort(grid, dim=0)
        # Fill each column in grid we are not not computing bit scores for with the median
        median, _ = torch.median(train_theta_adjusted, dim=0)
        bit_scores = torch.zeros_like(theta_adjusted)
        for theta_var in range(grid.shape[1]):
            # Only compute once per unique value
            unique_grid, inverse_indices = grid[:, theta_var].unique(return_inverse=True)
            latent_variable_grid = median.repeat(unique_grid.shape[0], 1)
            latent_variable_grid[:, theta_var] = unique_grid

            # Convert back to non-inverted theta scale and compute grid entropies
            output = self.model(latent_variable_grid * inverted_scale)
            entropies = entropy(self.model.probabilities_from_output(output))

            # Compute the absolute difference between each grid point entropy and the previous one
            diff = torch.zeros_like(entropies)
            diff[1:,:] = torch.abs(entropies[:-1, :] - entropies[1:, :])

            # cummulative sum over grid points and then sum the item scores
            bit_score_grid = diff.sum(dim=1).cumsum(dim=0)

            # add duplicates, unsort and take only the bit scores for the input theta scores
            bit_scores[:, theta_var] = bit_score_grid[inverse_indices][torch.sort(sorted_indices[:, theta_var])[1]][-theta_adjusted.shape[0]:]
            
        return bit_scores


    def _inverted_scales(self, train_theta):
        """
        Compute a tensor with information about whether each latent variable is positively or negatively related to the test scores.

        Parameters
        ----------
        train_theta : torch.Tensor
            The training data in the latent space.

        Returns
        -------
        torch.Tensor
            A one row tensor with elements corresponding to latent variables. 1's for positive and -1's for negative relationships.

        Notes
        -----
        If the neural network is set to handle multiple choice correct answers, 
        the method computes the scores based on the correct responses. 
        Otherwise, it simply sums up the training data. 
        The method then performs a linear regression between the latent variables 
        and the scores, and inverts the scales based on the linear weights.
        """
        scores = self.model.expected_scores(train_theta).sum(dim=1).reshape(-1, 1)
        linear_weights = linear_regression(train_theta, scores)[1:]
        inverted_scale = torch.where(linear_weights < 0, torch.tensor(-1), torch.tensor(1)).reshape(1, -1)
        return inverted_scale

    def _anti_invert_and_adjust_theta_scores(self, theta, train_theta, start_theta, inverted_scale):
        """
        Adjust the input and training theta scores based on the starting theta scores and the inverted scale.

        Parameters
        ----------
        theta : torch.Tensor
            The input theta scores.
        train_theta : torch.Tensor
            The training theta scores.
        start_theta : torch.Tensor
            The starting theta scores.
        inverted_scale : torch.Tensor
            The inverted scale.

        Returns
        -------
        tuple of torch.Tensor
            The adjusted input theta scores, the adjusted training theta scores and the adjusted starting theta scores.

        Notes
        -----
        The method first anti-inverts the theta scores by multiplying them with the inverted scale. 
        Then, If the anti-inverted theta scores are smaller than the starting theta scores, we set them to the starting theta scores.
        """
        start_theta_adjusted = start_theta * inverted_scale
        theta_adjusted = torch.max(theta * inverted_scale, start_theta_adjusted)
        train_theta_adjusted = torch.max(train_theta * inverted_scale, start_theta_adjusted)

        return theta_adjusted, train_theta_adjusted, start_theta_adjusted


    def _get_grid_boundaries(self, train_theta_adjusted: torch.Tensor, start_theta_adjusted):
        """
        Determines the start and end points of the grid used for computing bit scores.

        Parameters
        ----------
        train_theta_adjusted : torch.Tensor
            A 2D array containing the adjusted theta scores of the training data. Each row represents one respondent, and each column represents a latent variable.
        start_theta_adjusted : torch.Tensor
            A 1 row tensor containing starting theta scores for the bit scores.

        Returns
        -------
        grid_start : torch.Tensor
            A 1D tensor containing the start points of the grid for each latent variable.
        grid_end : torch.Tensor
            A 1D tensor containing the end points of the grid for each latent variable.
        start_is_outlier : torch.Tensor
            A 1 row boolean tensor where each column corresponds to one latent variable. True if the starting theta score is an outlier.
        """        
        outlier_detector = OutlierDetector(factor=4)
        start_is_outlier = outlier_detector.is_outlier(start_theta_adjusted, data=train_theta_adjusted, lower=True)[0, :]
        if any(start_is_outlier):
            smallest_non_outlier = outlier_detector.smallest_largest_non_outlier(train_theta_adjusted, smallest=True)
            grid_start = torch.max(start_theta_adjusted, smallest_non_outlier)
        else:
            grid_start = start_theta_adjusted
        grid_end = outlier_detector.smallest_largest_non_outlier(train_theta_adjusted, smallest=False)
        return grid_start, grid_end, start_is_outlier
