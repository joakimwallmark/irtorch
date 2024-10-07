from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import torch
from irtorch.rescale import Scale
from irtorch._internal_utils import entropy
from irtorch.outlier_detector import OutlierDetector
from irtorch._internal_utils import random_guessing_data

if TYPE_CHECKING:
    from irtorch.models.base_irt_model import BaseIRTModel

logger = logging.getLogger("irtorch")

class Bit(Scale):
    """
    Bit scale transformation, as introduced by :cite:t:`Wallmark2024`.

    Parameters
    ----------
    model : BaseIRTModel
        The IRT model to use for bit scale computation.
    population_theta : torch.Tensor, optional
        Theta scores from the population. Usually the training data.
        Used to find good starting values for the grid of theta scores used for the bit transformation.
        Recommended to use for models with theta distributions for which values far from 0 are common. (default is None)
    start_theta : torch.Tensor, optional
        The starting theta scores for the bit scale computation. If None, the minimum theta scores are used. (default is None)
    mc_start_theta_approx : bool, optional
        For multiple choice models. Whether to approximate the starting theta scores using simulated random guesses.
        If True, runs :meth:`bit_score_starting_theta_mc`. (default is False)
    **kwargs
        Additional keyword arguments for the starting theta approximation method. See :meth:`bit_score_starting_theta_mc`.
    
    Examples
    --------
    >>> import irtorch
    >>> from irtorch.models import MonotoneNN
    >>> from irtorch.estimation_algorithms import AE
    >>> from irtorch.rescale import Bit
    >>> data, mc_correct = irtorch.load_dataset.swedish_sat_quantitative()
    >>> model = MonotoneNN(data, mc_correct=mc_correct)
    >>> model.fit(train_data=data, algorithm=AE())
    >>> thetas = model.latent_scores(data, theta_estimation="NN")
    >>> # Initalize the scale transformation
    >>> # mc_start_theta_approx sets the starting theta to the approximate score of a randomly guessing respondent
    >>> bit = Bit(model, population_theta=thetas, mc_start_theta_approx=True)
    >>> # Supply the new scale to the model
    >>> model.rescale(bit)
    >>> # Estimate thetas on the transformed scale
    >>> rescaled_thetas = model.latent_scores(data)
    >>> # Or alternatively by directly converting the old ones
    >>> rescaled_thetas = model.scale(thetas)
    >>> # Plot the differences
    >>> model.plot.plot_latent_score_distribution(thetas).show()
    >>> model.plot.plot_latent_score_distribution(rescaled_thetas).show()
    >>> # Plot an item on the bit transformed scale
    >>> model.plot.plot_item_probabilities(1).show()
    """
    def __init__(
        self,
        model: BaseIRTModel,
        population_theta: torch.Tensor = None,
        start_theta: torch.Tensor = None,
        mc_start_theta_approx: bool = False,
        **kwargs
    ):
        self.model = model
        self._invert_scale_multiplier = self._get_inverted_scale_multiplier()
        self._population_theta = population_theta
        if start_theta is not None:
            if start_theta.flatten().shape != (self.model.latent_variables,):
                raise ValueError("start_theta must have the same number of elements as the number of latent variables.")
            self._start_theta = start_theta.flatten()
        else:
            if mc_start_theta_approx:
                if self.model.mc_correct is None:
                    raise ValueError("Random sampling start theta approximation is only supported for multiple choice models.")
                self.bit_score_starting_theta_mc(**kwargs)
            else:
                self._start_theta = self._invert_scale_multiplier * torch.full((self.model.latent_variables, ), -10000)

    def set_start_theta(self, start_theta: torch.Tensor):
        """
        Sets the starting theta scores for the bit scale computation.

        Parameters
        ----------
        start_theta : torch.Tensor
            The starting theta scores for the bit scale computation.
        """
        if start_theta.shape != (self.model.latent_variables,):
            raise ValueError("start_theta must have the same number of elements as the number of latent variables.")
        self._start_theta = start_theta

    @torch.inference_mode()
    def transform(
        self,
        theta: torch.Tensor,
        grid_points: int = 300,
        items: list[int] = None,
    ) -> torch.Tensor:
        r"""
        Transforms :math:`\mathbf{\theta}` scores into bit scores :math:`B(\mathbf{\theta})`.
        
        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor. Columns are latent variables and rows are respondents.
        grid_points : int, optional
            The number of points to use for computing bit score. More steps lead to more accurate results. (default is 300)
        items: list[int], optional
            The item indices for the items to use to compute the bit scores. (default is None and uses all items)
        
        Returns
        -------
        torch.Tensor
            A 2D tensor with bit score scale scores for each respondent across the rows together with another tensor with start_theta.

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
        if items is None:
            items = list(range(len(self.model.item_categories)))
        elif not isinstance(items, list) or not all(isinstance(item, int) for item in items):
            raise ValueError("items must be a list of integers.")

        start_theta_adjusted = self._start_theta * self._invert_scale_multiplier 
        theta_adjusted = torch.max(theta * self._invert_scale_multiplier, start_theta_adjusted)
        if self._population_theta is None:
            grid_start = torch.full((1, self.model.latent_variables), -7)
            grid_end = torch.full((1, self.model.latent_variables), 7)
            medians = torch.zeros(1, self.model.latent_variables)
        else:
            population_theta_adjusted = torch.max(self._population_theta * self._invert_scale_multiplier , start_theta_adjusted)
            grid_start, grid_end = self._get_grid_boundaries(population_theta_adjusted, start_theta_adjusted)
            medians, _ = torch.median(population_theta_adjusted, dim=0)
        
        bit_scores = self._compute_multi_dimensional_bit_scores(
            theta_adjusted,
            start_theta_adjusted,
            medians,
            grid_start,
            grid_end,
            self._invert_scale_multiplier,
            grid_points
        )
        
        return bit_scores

    @torch.inference_mode()
    def transform_to_1D(
        self,
        theta: torch.Tensor,
        grid_points: int = 300,
        items: list[int] = None,
    ) -> torch.Tensor:
        r"""
        Transforms :math:`\mathbf{\theta}` scores of a multi-dimensional model into one-dimensional bit scores :math:`B(\mathbf{\theta})`.
        Equivalent to :meth:`transform` for one-dimensional models.
        
        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor. Columns are latent variables and rows are respondents.
        grid_points : int, optional
            The number of points to use for computing bit score. More steps lead to more accurate results. (default is 300)
        items: list[int], optional
            The item indices for the items to use to compute the bit scores. (default is None and uses all items)
        
        Returns
        -------
        torch.Tensor
            A 2D tensor with bit score scale scores for each respondent across the rows together with another tensor with start_theta.

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
        if items is None:
            items = list(range(len(self.model.item_categories)))
        elif not isinstance(items, list) or not all(isinstance(item, int) for item in items):
            raise ValueError("items must be a list of integers.")
        if self.model.latent_variables == 1:
            return self.transform(theta=theta, grid_points=grid_points, items=items)

        start_theta_adjusted = self._start_theta * self._invert_scale_multiplier 
        theta_adjusted = torch.max(theta * self._invert_scale_multiplier, start_theta_adjusted)
        if self._population_theta is None:
            grid_start = torch.full((1, self.model.latent_variables), -7)
            grid_end = torch.full((1, self.model.latent_variables), 7)
        else:
            population_theta_adjusted = torch.max(self._population_theta * self._invert_scale_multiplier , start_theta_adjusted)
            grid_start, grid_end = self._get_grid_boundaries(population_theta_adjusted, start_theta_adjusted)
        
        bit_scores = self._compute_1d_bit_scores(
            theta_adjusted,
            start_theta_adjusted,
            grid_start,
            grid_end,
            self._invert_scale_multiplier,
            grid_points
        )
        
        return bit_scores

    @torch.inference_mode(False)
    def gradients(
        self,
        theta: torch.Tensor,
        items: list[int] = None,
    ) -> torch.Tensor:
        r"""
        Computes the gradients of the bit scores with respect to the input theta scores

        .. math ::

            \frac{\partial B(\mathbf{\theta})}{\partial \mathbf{\theta}} = 
            \sum_{j=1}^J\frac{\partial B_j(\mathbf{\theta})}{\partial \mathbf{\theta}} = 
            \sum_{j=1}^J\frac{\partial \int_{t=\mathbf{\theta}^{(0)}}^{\mathbf{\theta}}\left|\frac{dH_j(t)}{dt}\right| dt}{\partial \mathbf{\theta}}=
            \sum_{j=1}^J\left|\frac{\partial H_j(\mathbf{\theta})}{\partial \mathbf{\theta}}\right|,

        where 

        - :math:`\mathbf{\theta}^{(0)}` is the minimum :math:`\mathbf{\theta}`
        - :math:`H_j(\mathbf{\theta})` is entropy for item :math:`j` as a function of :math:`\mathbf{\theta}`

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing latent variable theta scores. Each column represents one latent variable.
        items: list[int], optional
            The item indices for the items to use to compute the bit scores. (default is None and uses all items)

        Returns
        -------
        torch.Tensor
            A torch tensor with the gradients for each theta score. Dimensions are (theta rows, latent variables, latent variables) where the last two are the jacobians.
        """
        if self.model.latent_variables == 1:
            if theta.requires_grad:
                theta.requires_grad_(False)

            gradients = torch.zeros(theta.shape[0], len(self.model.item_categories), theta.shape[1])
            theta_scores = theta.clone()
            theta_scores.requires_grad_(True)
            probs = self.model.item_probabilities(theta_scores)
            entropies = entropy(probs)

            for item in range(entropies.shape[1]):
                if theta_scores.grad is not None:
                    theta_scores.grad.zero_()
                entropies[:, item].sum().backward(retain_graph=True)
                for latent_variable in range(theta.shape[1]):
                    gradients[:, item, latent_variable] = theta_scores.grad[:, latent_variable]

            gradients[gradients.isnan()] = 0.
            if items is not None:
                gradients = gradients[:, items, :]

            # sum over items and add dimension for jacobian
            # multiply by -1 if we are inversely related to the theta scores
            return self._invert_scale_multiplier.flatten() * gradients.abs().sum(dim=1).unsqueeze(dim=-1)
        else:
            raise NotImplementedError("Multidimensional bit scale gradients is not implemented.")
        # if hasattr(self.model.algorithm, "training_theta_scores") and self.model.algorithm.training_theta_scores is not None:
        #     q1_q3 = torch.quantile(self.model.algorithm.training_theta_scores, torch.tensor([0.25, 0.75]), dim=0)
        # else:
        #     q1_q3 = torch.tensor([-0.6745, 0.6745]).unsqueeze(1).expand(-1, self.model.latent_variables)

        # iqr = q1_q3[1] - q1_q3[0]
        # lower_bound = q1_q3[0] - 1.5 * iqr
        # upper_bound = q1_q3[1] + 1.5 * iqr
        # h = (upper_bound - lower_bound) / 1000
        # theta_low = theta-h
        # theta_high = theta+h
        # if independent_theta is None:
        #     gradients = torch.zeros(theta.shape[0],theta.shape[1],theta.shape[1])
        #     for latent_variable in range(theta.shape[1]):
        #         theta_low_var = torch.cat((theta[:, :latent_variable], theta_low[:, latent_variable].view(-1, 1), theta[:, latent_variable+1:]), dim=1)
        #         theta_high_var = torch.cat((theta[:, :latent_variable], theta_high[:, latent_variable].view(-1, 1), theta[:, latent_variable+1:]), dim=1)
        #         bit_scores_low = self.transform(
        #             theta_low_var, one_dimensional=one_dimensional, grid_points=grid_points, items=items
        #         )
        #         bit_scores_high = self.transform(
        #             theta_high_var, one_dimensional=one_dimensional, grid_points=grid_points, items=items
        #         )
        #         gradients[:, latent_variable, :] = (bit_scores_high - bit_scores_low) / (2 * h[latent_variable])
        # else:
        #     theta_low_var = torch.cat((theta[:, :independent_theta-1], theta_low[:, independent_theta-1].view(-1, 1), theta[:, independent_theta:]), dim=1)
        #     theta_high_var = torch.cat((theta[:, :independent_theta-1], theta_high[:, independent_theta-1].view(-1, 1), theta[:, independent_theta:]), dim=1)
        #     bit_scores_low = self(theta_low_var, one_dimensional=one_dimensional, grid_points=grid_points, items=items)
        #     bit_scores_high = self(theta_high_var, one_dimensional=one_dimensional, grid_points=grid_points, items=items)
        #     gradients = (bit_scores_high - bit_scores_low) / (2 * h[independent_theta-1])

        # return gradients

    def bit_score_starting_theta_mc(
        self,
        theta_estimation: str = "ML",
        ml_map_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lbfgs_learning_rate: float = 0.25,
        items: list[int] = None,
        guessing_probabilities: list[float] = None,
        guessing_iterations: int = 10000,
    ):
        r"""
        For multiple choice models, approximate the starting theta score :math:`\mathbf{\theta}^{(0)}` from which to compute bit scores.
        See notes under :meth:`bit_scores` for the bit score formula.
        
        Parameters
        ----------
        model : BaseIRTModel
            The IRT model to use for computing the starting theta scores.
        theta_estimation : str, optional
            Method used to obtain the theta scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'ML')
        ml_map_device: str, optional
            For ML and MAP. The device to use for computation. Can be 'cpu' or 'cuda'. (default is "cuda" if available else "cpu")
        lbfgs_learning_rate: float, optional
            For ML and MAP. The learning rate to use for the LBFGS optimizer. (default is 0.3)
        items: list[int], optional
            The item indices for the items to use to compute the bit scores. (default is None and uses all items)
        guessing_probabilities: list[float], optional
            The guessing probability for each item. The same length as the number of items. Guessing is not supported for polytomously scored items and the probabilities for them will be ignored. (default is None and uses no guessing or, for multiple choice models, 1 over the number of item categories)
        guessing_iterations: int, optional
            The number of iterations to use for approximating a minimum theta when guessing is incorporated. (default is 200)
        
        Returns
        -------
        torch.Tensor
            A tensor with all the starting theta values.
        """
        if self.model.mc_correct is None:
            raise ValueError("This method is only supported for multiple choice models.")

        items = items or list(range(len(self.model.item_categories)))
        mc_correct = torch.tensor(self.model.mc_correct)
        if not all(isinstance(item, int) for item in items):
            raise ValueError("items must be a list of integers.")
        
        selected_item_categories = [self.model.item_categories[i] for i in items]

        if guessing_probabilities:
            if len(guessing_probabilities) != len(items) or not all(0 <= num < 1 for num in guessing_probabilities):
                raise ValueError("guessing_probabilities must be list of floats with the same length as the number of items and values between 0 and 1.")
        else:
            guessing_probabilities = [1 / categories for categories in selected_item_categories]

        selected_correct = mc_correct.index_select(0, torch.tensor(items))
        random_data = random_guessing_data(
            selected_item_categories,
            guessing_iterations,
            guessing_probabilities,
            selected_correct
        )

        logger.info("Approximating theta from random guessing data to get minimum bit score.")
        guessing_theta = self.model.latent_scores(random_data, theta_estimation=theta_estimation, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate, rescale=False)

        starting_theta = guessing_theta.detach().median(dim=0).values.reshape(1, self.model.latent_variables)

        self._start_theta = starting_theta.flatten()

    def _compute_1d_bit_scores(self, theta_adjusted, start_theta_adjusted, grid_start, grid_end, invert_scale_multiplier, grid_points):
        """
        Computes the 1D bit scores for multidimensional models for the given input theta scores.

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
        invert_scale_multiplier : torch.Tensor
            A one row tensor with -1 for latent variables inversely related to the test scores and 1 for others.
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
        grid = grid.view(-1, grid.shape[2]) * invert_scale_multiplier
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

    def _compute_multi_dimensional_bit_scores(self, theta_adjusted, start_theta_adjusted, median_thetas, grid_start, grid_end, invert_scale_multiplier, grid_points):
        """
        Computes the multi-dimensional bit scores for the given input theta scores (or more efficient one-dimensional scores for 1D models).

        Parameters
        -----------
        theta_adjusted : torch.Tensor
            The input theta scores.
        start_theta_adjusted : torch.Tensor
            The minimum theta score to be used in the grid. A one row tensor.
        median_thetas : torch.Tensor
            The median theta score of each latent variable.
        grid_start : float
            The minimum value of the grid. A one row tensor.
        grid_end : float
            The maximum value of the grid. A one row tensor.
        invert_scale_multiplier : torch.Tensor
            A one row tensor with -1 for latent variables inversely related to the test scores and 1 for others.
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
        bit_scores = torch.zeros_like(theta_adjusted)
        for theta_var in range(grid.shape[1]):
            # Only compute once per unique value
            unique_grid, inverse_indices = grid[:, theta_var].unique(return_inverse=True)
            latent_variable_grid = median_thetas.repeat(unique_grid.shape[0], 1)
            latent_variable_grid[:, theta_var] = unique_grid

            # Convert back to non-inverted theta scale and compute grid entropies
            output = self.model(latent_variable_grid * invert_scale_multiplier)
            entropies = entropy(self.model.probabilities_from_output(output))

            # Compute the absolute difference between each grid point entropy and the previous one
            diff = torch.zeros_like(entropies)
            diff[1:,:] = torch.abs(entropies[:-1, :] - entropies[1:, :])

            # cummulative sum over grid points and then sum the item scores
            bit_score_grid = diff.sum(dim=1).cumsum(dim=0)

            # add duplicates, unsort and take only the bit scores for the input theta scores
            bit_scores[:, theta_var] = bit_score_grid[inverse_indices][torch.sort(sorted_indices[:, theta_var])[1]][-theta_adjusted.shape[0]:]
            
        return bit_scores


    def _get_inverted_scale_multiplier(self):
        """
        Compute a tensor with information about whether each latent variable is positively or negatively related to the test scores.
        Assumes monotonicity of the IRT model in the theta range (-1, 1).

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
        inverted_scale_multiplier = torch.ones(1, self.model.latent_variables)
        for latent_variable in range(self.model.latent_variables):
            thetas = torch.zeros(2, self.model.latent_variables)
            thetas[0, latent_variable] = -1.
            thetas[1, latent_variable] = 1.
            scores = self.model.expected_scores(thetas).sum(dim=1)
            if scores[0] > scores[1]:
                inverted_scale_multiplier[0, latent_variable] = -1.

        return inverted_scale_multiplier


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
            A 1 row tensor containing the start points of the grid for each latent variable.
        grid_end : torch.Tensor
            A 1 row tensor containing the end points of the grid for each latent variable.
        """        
        outlier_detector = OutlierDetector(factor=4)
        start_is_outlier = outlier_detector.is_outlier(start_theta_adjusted, data=train_theta_adjusted, lower=True)[0, :]
        if any(start_is_outlier):
            smallest_non_outlier = outlier_detector.smallest_largest_non_outlier(train_theta_adjusted, smallest=True)
            grid_start = torch.max(start_theta_adjusted, smallest_non_outlier)
        else:
            grid_start = start_theta_adjusted
        grid_end = outlier_detector.smallest_largest_non_outlier(train_theta_adjusted, smallest=False)
        return grid_start, grid_end


    # def bit_score_starting_theta(
    #     self,
    #     theta_estimation: str = "ML",
    #     ml_map_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    #     lbfgs_learning_rate: float = 0.25,
    #     items: list[int] = None,
    #     start_all_incorrect: bool = False,
    #     train_theta: torch.Tensor = None,
    #     guessing_probabilities: list[float] = None,
    #     guessing_iterations: int = 10000,
    # ):
    #     r"""
    #     Computes the starting theta score :math:`\mathbf{\theta}^{(0)}` from which to compute bit scores. See notes under :meth:`bit_scores` for more details.
        
    #     Parameters
    #     ----------
    #     train_theta : torch.Tensor
    #         A 2D tensor with the training data theta scores. Used to estimate relationships between theta and getting the items correct when start_all_incorrect is False. Columns are latent variables and rows are respondents.
    #     theta_estimation : str, optional
    #         Method used to obtain the theta scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'ML')
    #     ml_map_device: str, optional
    #         For ML and MAP. The device to use for computation. Can be 'cpu' or 'cuda'. (default is "cuda" if available else "cpu")
    #     lbfgs_learning_rate: float, optional
    #         For ML and MAP. The learning rate to use for the LBFGS optimizer. (default is 0.3)
    #     items: list[int], optional
    #         The item indices for the items to use to compute the bit scores. (default is None and uses all items)
    #     start_all_incorrect: bool, optional
    #         Whether to compute the starting theta scores based incorrect responses. If false, starting theta is computed based on relationships between each latent variable and the item responses. (default is False)
    #     guessing_probabilities: list[float], optional
    #         The guessing probability for each item. The same length as the number of items. Guessing is not supported for polytomously scored items and the probabilities for them will be ignored. (default is None and uses no guessing or, for multiple choice models, 1 over the number of item categories)
    #     guessing_iterations: int, optional
    #         The number of iterations to use for approximating a minimum theta when guessing is incorporated. (default is 200)
        
    #     Returns
    #     -------
    #     torch.Tensor
    #         A tensor with all the starting theta values.
    #     """
    #     items = items or list(range(len(self.model.item_categories)))
    #     mc_correct = torch.tensor(self.model.mc_correct) if self.model.mc_correct else None
    #     if not all(isinstance(item, int) for item in items):
    #         raise ValueError("items must be a list of integers.")
        
    #     if guessing_probabilities:
    #         if len(guessing_probabilities) != len(items) or not all(0 <= num < 1 for num in guessing_probabilities):
    #             raise ValueError("guessing_probabilities must be list of floats with the same length as the number of items and values between 0 and 1.")

    #     selected_item_categories = [self.model.item_categories[i] for i in items]

    #     if guessing_probabilities is None and mc_correct is not None:
    #         guessing_probabilities = [1 / categories for categories in selected_item_categories]

    #     if not start_all_incorrect:
    #         # Which latent variables are inversely related to the test scores?
    #         item_sum_scores = self.model.expected_scores(train_theta, return_item_scores=False)
    #         test_weights = linear_regression(train_theta, item_sum_scores.reshape(-1, 1))[1:]
    #         inverted_scale = torch.where(test_weights < 0, torch.tensor(-1), torch.tensor(1)).reshape(-1)
            
    #         # Which latent variables are positively related to the item scores?
    #         directions = self.model.item_theta_relationship_directions(train_theta)
    #         item_theta_positive = (inverted_scale * directions) >= 0 # Invert item relationship if overall test relationship is inverted

    #     if guessing_probabilities is None:
    #         if start_all_incorrect:
    #             logger.info("Computing theta for all incorrect responses to get minimum bit score.")
    #             starting_theta = self.model.latent_scores(torch.zeros((1, len(items))).float(), theta_estimation=theta_estimation, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate, rescale=False)
    #         else:
    #             # Get minimum score in relation to each latent variable
    #             min_sum_score = torch.zeros((len(items), self.model.latent_variables))
    #             min_sum_score[~item_theta_positive] = (torch.tensor(selected_item_categories) - 1).view(-1, 1).float().repeat(1, self.model.latent_variables)[~item_theta_positive]

    #             # get the minimum theta scores based on the sum scores
    #             starting_theta = torch.zeros((1, self.model.latent_variables)).float()
    #             for theta in range(self.model.latent_variables):
    #                 logger.info("Computing minimum bit score theta for latent variable %s.", theta+1)
    #                 starting_theta[:, theta] = self.model.latent_scores(min_sum_score.float()[:, theta], theta_estimation=theta_estimation, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate, rescale=False)[:, theta]
    #     else:
    #         if mc_correct is None:
    #             selected_correct = torch.ones(len(items))
    #             random_data = random_guessing_data(selected_item_categories, guessing_iterations, guessing_probabilities)
    #         else:
    #             selected_correct = mc_correct.index_select(0, torch.tensor(items))
    #             random_data = random_guessing_data(
    #                 selected_item_categories,
    #                 guessing_iterations,
    #                 guessing_probabilities,
    #                 selected_correct
    #             )

    #         if start_all_incorrect:
    #             # With one-dimensional bit scores, guessing for all items makes sense
    #             logger.info("Approximating theta from random guessing data to get minimum bit score.")
    #             guessing_theta = self.model.latent_scores(random_data, theta_estimation=theta_estimation, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate, rescale=False)
    #         else:
    #             # Random for positively related and set to correct for others
    #             guessing_theta = torch.zeros(random_data.shape[0], self.model.latent_variables)
    #             for theta in range(self.model.latent_variables):
    #                 random_data_theta = random_data.clone()
    #                 random_data_theta[:, ~item_theta_positive[:, theta]] = selected_correct[~item_theta_positive[:, theta]].float() - 1
    #                 logger.info("Approximating minimum bit score theta from random guessing data for latent variable %s.", theta+1)
    #                 guessing_theta[:, theta] = self.model.latent_scores(random_data_theta, theta_estimation=theta_estimation, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate, rescale=False)[:, theta]

    #         starting_theta = guessing_theta.detach().median(dim=0).values.reshape(1, self.model.latent_variables)

    #     return starting_theta