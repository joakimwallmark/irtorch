import logging
import torch
from torch.distributions import MultivariateNormal
from irtorch.models import BaseIRTModel
from irtorch._internal_utils import output_to_item_entropy, random_guessing_data, linear_regression
from irtorch.outlier_detector import OutlierDetector
from irtorch.estimation_algorithms import AE, VAE, MML

logger = logging.getLogger("irtorch")

def bit_score_starting_z(
    model: BaseIRTModel,
    z_estimation_method: str = "ML",
    ml_map_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    lbfgs_learning_rate: float = 0.3,
    items: list[int] = None,
    start_all_incorrect: bool = False,
    train_z: torch.Tensor = None,
    guessing_probabilities: list[float] = None,
    guessing_iterations: int = 10000,
):
    """
    Computes the starting z score from which to compute bit scores.
    
    Parameters
    ----------
    model : BaseIRTModel
        The model for which to compute the starting z scores.
    z_estimation_method : str, optional
        Method used to obtain the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'ML')
    ml_map_device: str, optional
        For ML and MAP. The device to use for computation. Can be 'cpu' or 'cuda'. (default is "cuda" if available else "cpu")
    lbfgs_learning_rate: float, optional
        For ML and MAP. The learning rate to use for the LBFGS optimizer. (default is 0.3)
    items: list[int], optional
        The item indices for the items to use to compute the bit scores. (default is None and uses all items)
    start_all_incorrect: bool, optional
        Whether to compute the starting z scores based incorrect responses. If false, starting z is computed based on relationships between each latent variable and the item responses. (default is False)
    train_z : torch.Tensor, optional
        A 2D tensor with the training data z scores. Used to estimate relationships between z and getting the items correct when start_all_incorrect is False. Columns are latent variables and rows are respondents. (default is None and uses encoder z scores from the model training data)
    guessing_probabilities: list[float], optional
        The guessing probability for each item. The same length as the number of items. Guessing is not supported for polytomously scored items and the probabilities for them will be ignored. (default is None and uses no guessing or, for multiple choice models, 1 over the number of item categories)
    guessing_iterations: int, optional
        The number of iterations to use for approximating a minimum z when guessing is incorporated. (default is 200)
    
    
    Returns
    -------
    torch.Tensor
        A tensor with all the starting z values.
    """
    items = items or list(range(len(model.modeled_item_responses)))
    mc_correct = torch.tensor(model.mc_correct) if model.mc_correct else None
    if not all(isinstance(item, int) for item in items):
        raise ValueError("items must be a list of integers.")
    
    if guessing_probabilities:
        if len(guessing_probabilities) != len(items) or not all(0 <= num < 1 for num in guessing_probabilities):
            raise ValueError("guessing_probabilities must be list of floats with the same length as the number of items and values between 0 and 1.")

    selected_item_categories = [model.item_categories[i] for i in items]

    if guessing_probabilities is None and mc_correct is not None:
        guessing_probabilities = [1 / categories for categories in selected_item_categories]

    if not start_all_incorrect:
        if train_z is None:
            if isinstance(model.algorithm, (AE, VAE)):
                train_z = model.algorithm.training_z_scores
            elif isinstance(model.algorithm, MML):
                mvn = MultivariateNormal(torch.zeros(model.latent_variables), model.algorithm.covariance_matrix)
                train_z = mvn.sample((4000,)).to(dtype=torch.float32)

        # Which latent variables are inversely related to the test scores?
        item_sum_scores = model.expected_scores(train_z, return_item_scores=False)
        test_weights = linear_regression(train_z, item_sum_scores.reshape(-1, 1))[1:]
        inverted_scale = torch.where(test_weights < 0, torch.tensor(-1), torch.tensor(1)).reshape(-1)
        
        # Which latent variables are positively related to the item scores?
        directions = model.item_z_relationship_directions(train_z)
        item_z_positive = (inverted_scale * directions) >= 0 # Invert item relationship if overall test relationship is inverted

    if guessing_probabilities is None:
        if start_all_incorrect:
            logger.info("Computing z for all incorrect responses to get minimum bit score.")
            starting_z = model.latent_scores(torch.zeros((1, len(items))).float(), scale="z", z_estimation_method=z_estimation_method, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate)
        else:
            # Get minimum score in relation to each latent variable
            min_sum_score = torch.zeros((len(items), model.latent_variables))
            if model.model_missing:
                min_sum_score[~item_z_positive] = (torch.tensor(selected_item_categories) - 2).view(-1, 1).float().repeat(1, model.latent_variables)[~item_z_positive]
            else:
                min_sum_score[~item_z_positive] = (torch.tensor(selected_item_categories) - 1).view(-1, 1).float().repeat(1, model.latent_variables)[~item_z_positive]

            # get the minimum z scores based on the sum scores
            starting_z = torch.zeros((1, model.latent_variables)).float()
            for z in range(model.latent_variables):
                logger.info("Computing minimum bit score z for latent variable %s.", z+1)
                starting_z[:, z] = model.latent_scores(min_sum_score.float()[:, z], scale="z", z_estimation_method=z_estimation_method, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate)[:, z]
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
            logger.info("Approximating z from random guessing data to get minimum bit score.")
            guessing_z = model.latent_scores(random_data, scale="z", z_estimation_method=z_estimation_method, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate)
        else:
            # Random for positively related and set to correct for others
            guessing_z = torch.zeros(random_data.shape[0], model.latent_variables)
            for z in range(model.latent_variables):
                random_data_z = random_data.clone()
                random_data_z[:, ~item_z_positive[:, z]] = selected_correct[~item_z_positive[:, z]].float() - 1
                logger.info("Approximating minimum bit score z from random guessing data for latent variable %s.", z+1)
                guessing_z[:, z] = model.latent_scores(random_data_z, scale="z", z_estimation_method=z_estimation_method, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate)[:, z]

        starting_z = guessing_z.detach().median(dim=0).values.reshape(1, model.latent_variables)

    return starting_z


@torch.inference_mode()
def bit_scores_from_z(
    model: BaseIRTModel,
    z: torch.Tensor,
    start_z: torch.Tensor = None,
    population_z: torch.Tensor = None,
    one_dimensional: bool = False,
    z_estimation_method: str = "ML",
    ml_map_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    lbfgs_learning_rate: float = 0.3,
    grid_points: int = 300,
    items: list[int] = None,
    start_z_guessing_probabilities: list[float] = None,
    start_z_guessing_iterations: int = 10000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the bit scores from z scores.

    Parameters
    ----------
    model : BaseIRTModel
        The model for which to compute the bit scores.
    z : torch.Tensor
        A 2D tensor. Columns are latent variables and rows are respondents.
    start_z : torch.Tensor, optional
        A one row 2D tensor with bit score starting values of each latent variable. Estimated automatically if not provided. (default is None)
    population_z : torch.Tensor, optional
        A 2D tensor with z scores of the population. Used to estimate relationships between each z and sum scores. Columns are latent variables and rows are respondents. (default is None and uses z_estimation_method with the model training data)
    one_dimensional: bool, optional
        Whether to estimate one combined bit score for a multidimensional model. (default is True)
    z_estimation_method : str, optional
        Method used to obtain the z score grid for bit score computation. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'ML')
    ml_map_device: str, optional
        For ML and MAP. The device to use for computation. Can be 'cpu' or 'cuda'. (default is "cuda" if available else "cpu")
    lbfgs_learning_rate: float, optional
        For ML and MAP. The learning rate to use for the LBFGS optimizer. (default is 0.3)
    grid_points : int, optional
        The number of points to use for computing bit score. More steps lead to more accurate results. (default is 300)
    items: list[int], optional
        The item indices for the items to use to compute the bit scores. (default is None and uses all items)
    start_z_guessing_probabilities: list[float], optional
        The guessing probability for each item if start_z is None. The same length as the number of items. Guessing is not supported for polytomously scored items and the probabilities for them will be ignored. (default is None and uses no guessing or, for multiple choice models, 1 over the number of item categories)
    start_z_guessing_iterations: int, optional
        The number of iterations to use for approximating a minimum z when guessing is incorporated. (default is 10000)
    
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A 2D tensor with bit score scale scores for each respondent across the rows together with another tensor with start_z.
    """
    if grid_points <= 0:
        raise ValueError("steps must be a positive integer")
    if start_z is not None and start_z.shape != (1, model.latent_variables):
        raise ValueError(f"start_z must be a one row tensor with shape (1, {model.latent_variables}).")
    if z_estimation_method not in ["NN", "ML", "EAP", "MAP"]:
        raise ValueError("Invalid bit_score_z_grid_method. Choose either 'NN', 'ML', 'EAP' or 'MAP'.")
    if items is None:
        items = list(range(len(model.modeled_item_responses)))
    elif not isinstance(items, list) or not all(isinstance(item, int) for item in items):
        raise ValueError("items must be a list of integers.")
    
    if population_z is None:
        if z_estimation_method != "NN":
            logger.info("Estimating population z scores needed for bit score computation.")
            population_z = model.latent_scores(model.algorithm.train_data, scale="z", z_estimation_method=z_estimation_method, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate)
        else:
            if isinstance(model.algorithm, (AE, VAE)):
                logger.info("Using traning data z scores as population z scores for bit score computation.")
                population_z = model.algorithm.training_z_scores
            elif isinstance(model.algorithm, MML):
                logger.info("Sampling from multivariate normal as population z scores for bit score computation.")
                mvn = MultivariateNormal(torch.zeros(model.latent_variables), model.algorithm.covariance_matrix)
                population_z = mvn.sample((4000,)).to(dtype=torch.float32)

    if start_z is None:
        start_z = bit_score_starting_z(
            model=model,
            z_estimation_method=z_estimation_method,
            items=items,
            start_all_incorrect=one_dimensional,
            train_z=population_z,
            guessing_probabilities=start_z_guessing_probabilities,
            guessing_iterations=start_z_guessing_iterations,
        )

    inverted_scales = _inverted_scales(model, population_z)
    z_adjusted, train_z_adjusted, start_z_adjusted = _anti_invert_and_adjust_z_scores(z, population_z, start_z, inverted_scales)
    grid_start, grid_end, _ = _get_grid_boundaries(train_z_adjusted, start_z_adjusted)
    
    if one_dimensional:
        bit_scores = _compute_1d_bit_scores(
            model,
            z_adjusted,
            start_z_adjusted,
            grid_start,
            grid_end,
            inverted_scales,
            grid_points
        )
    else:
        bit_scores = _compute_multi_dimensional_bit_scores(
            model,
            z_adjusted,
            start_z_adjusted,
            train_z_adjusted,
            grid_start,
            grid_end,
            inverted_scales,
            grid_points
        )
    
    return bit_scores, start_z

@torch.inference_mode(False)
def bit_score_gradients(
    model: BaseIRTModel,
    z: torch.Tensor,
    h: float = None,
    independent_z: int = None,
    start_z: torch.Tensor = None,
    population_z: torch.Tensor = None,
    one_dimensional: bool = False,
    z_estimation_method: str = "ML",
    ml_map_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    lbfgs_learning_rate: float = 0.3,
    grid_points: int = 300,
    items: list[int] = None,
    start_z_guessing_probabilities: list[float] = None,
    start_z_guessing_iterations: int = 10000,
) -> torch.Tensor:
    r"""
    Computes the gradients of the bit scores with respect to the input z scores using the central difference method: 
    .. math ::

        f^{\prime}(z) \approx \frac{f(z+h)-f(z-h)}{2 h}

    Parameters
    ----------
    model : BaseIRTModel
        The model for which to compute the bit score gradients.
    z : torch.Tensor
        A 2D tensor containing latent variable z scores. Each column represents one latent variable.
    h : float, optional
        The step size for the central difference method. (default is uses the difference between the smaller and upper outlier limits (computed using the interquantile range rule) of the training z scores divided by 1000)
    independent_z : int, optional
        The latent variable to differentiate with respect to. (default is None and computes gradients with respect to z)
    start_z : torch.Tensor, optional
        A one row 2D tensor with bit score starting values of each latent variable. Estimated automatically if not provided. (default is None)
    population_z : torch.Tensor, optional
        A 2D tensor with z scores of the population. Used to estimate relationships between each z and sum scores. Columns are latent variables and rows are respondents. (default is None and uses z_estimation_method with the model training data)
    one_dimensional: bool, optional
        Whether to estimate one combined bit score for a multidimensional model. (default is True)
    z_estimation_method : str, optional
        Method used to obtain the z score grid for bit score computation. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'ML')
    ml_map_device: str, optional
        For ML and MAP. The device to use for computation. Can be 'cpu' or 'cuda'. (default is "cuda" if available else "cpu")
    lbfgs_learning_rate: float, optional
        For ML and MAP. The learning rate to use for the LBFGS optimizer. (default is 0.3)
    grid_points : int, optional
        The number of points to use for computing bit score. More steps lead to more accurate results. (default is 300)
    items: list[int], optional
        The item indices for the items to use to compute the bit scores. (default is None and uses all items)
    start_z_guessing_probabilities: list[float], optional
        The guessing probability for each item if start_z is None. The same length as the number of items. Guessing is not supported for polytomously scored items and the probabilities for them will be ignored. (default is None and uses no guessing or, for multiple choice models, 1 over the number of item categories)
    start_z_guessing_iterations: int, optional
        The number of iterations to use for approximating a minimum z when guessing is incorporated. (default is 10000)

    Returns
    -------
    torch.Tensor
        A torch tensor with the gradients for each z score. Dimensions are (z rows, bit scores, z scores) where the last two dimensions represent the jacobian.
        If independent_z is provided, the tensor has dimensions (z rows, bit scores).
    """
    if start_z is None:
        start_z = bit_score_starting_z(
            model=model,
            z_estimation_method=z_estimation_method,
            items=items,
            start_all_incorrect=one_dimensional,
            train_z=population_z,
            guessing_probabilities=start_z_guessing_probabilities,
            guessing_iterations=start_z_guessing_iterations,
        )

    if hasattr(model.algorithm, "training_z_scores") and model.algorithm.training_z_scores is not None:
    # if isinstance(self.algorithm, (AE, VAE)):
        q1_q3 = torch.quantile(model.algorithm.training_z_scores, torch.tensor([0.25, 0.75]), dim=0)
    else:
    # elif isinstance(self.algorithm, MML):
        q1_q3 = torch.tensor([-0.6745, 0.6745]).unsqueeze(1).expand(-1, model.latent_variables)

    
    iqr = q1_q3[1] - q1_q3[0]
    lower_bound = q1_q3[0] - 1.5 * iqr
    upper_bound = q1_q3[1] + 1.5 * iqr
    h = (upper_bound - lower_bound) / 1000
    z_low = z-h
    z_high = z+h
    if independent_z is None:
        gradients = torch.zeros(z.shape[0], z.shape[1], z.shape[1])
        for latent_variable in range(z.shape[1]):
            z_low_var = torch.cat((z[:, :latent_variable], z_low[:, latent_variable].view(-1, 1), z[:, latent_variable+1:]), dim=1)
            z_high_var = torch.cat((z[:, :latent_variable], z_high[:, latent_variable].view(-1, 1), z[:, latent_variable+1:]), dim=1)
            bit_scores_low = bit_scores_from_z(
                model, z_low_var, start_z = start_z, population_z=population_z, one_dimensional=one_dimensional, z_estimation_method=z_estimation_method,
                ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate, grid_points=grid_points, items=items
            )[0]
            bit_scores_high = bit_scores_from_z(
                model, z_high_var, start_z = start_z, population_z=population_z, one_dimensional=one_dimensional, z_estimation_method=z_estimation_method,
                ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate, grid_points=grid_points, items=items
            )[0]
            gradients[:, latent_variable, :] = (bit_scores_high - bit_scores_low) / (2 * h[latent_variable])
    else:
        z_low_var = torch.cat((z[:, :independent_z-1], z_low[:, independent_z-1].view(-1, 1), z[:, independent_z:]), dim=1)
        z_high_var = torch.cat((z[:, :independent_z-1], z_high[:, independent_z-1].view(-1, 1), z[:, independent_z:]), dim=1)
        bit_scores_low = bit_scores_from_z(
            model, z_low_var, start_z = start_z, population_z=population_z, one_dimensional=one_dimensional, z_estimation_method=z_estimation_method,
            ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate, grid_points=grid_points, items=items
        )[0]
        bit_scores_high = bit_scores_from_z(
            model, z_high_var, start_z = start_z, population_z=population_z, one_dimensional=one_dimensional, z_estimation_method=z_estimation_method,
            ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate, grid_points=grid_points, items=items
        )[0]
        gradients = (bit_scores_high - bit_scores_low) / (2 * h[independent_z-1])

    return gradients


@torch.inference_mode(False)
def bit_expected_item_score_slopes(
    model: BaseIRTModel,
    z: torch.Tensor,
    bit_scores: torch.Tensor = None,
    rescale_by_item_score: bool = True,
    **kwargs
) -> torch.Tensor:
    """
    Computes the slope of the expected item scores with respect to the bit scores, averaged over the sample in z. Similar to loadings in traditional factor analysis. For each separate latent variable, the slope is computed as the average of the slopes of the expected item scores for each item, using the median z scores for the other latent variables.

    Parameters
    ----------
    model : BaseIRTModel
        The model for which to compute the expected item score slopes.
    z : torch.Tensor, optional
        A 2D tensor with latent z scores from the population of interest. Each row represents one respondent, and each column represents a latent variable. If not provided, uses the training z scores. (default is None)
    scale : str, optional
        The latent trait scale to differentiate with respect to. Can be 'bit' or 'z'. 
        'bit' is only a linear approximation for multidimensional models since multiple z scores can lead to the same bit scores, 
        and thus there are no unique derivatives of the item scores with respect to the bit scores for multidimensional models. (default is 'z')
    bit_scores: torch.Tensor, optional
        A 2D tensor with bit scores corresponding to the z scores. If not provided, computes the bit scores from the z scores. (default is None)
    rescale_by_item_score : bool, optional
        Whether to rescale the expected items scores to have a max of one by dividing by the max item score. (default is True)
    **kwargs
        Additional keyword arguments for the bit_score_gradients method.

    Returns
    -------
    torch.Tensor
        A tensor with the expected item score slopes.
    """
    if z.shape[0] < 2:
        raise ValueError("z must have at least 2 rows.")
    if z.requires_grad:
        z.requires_grad_(False)

    if model.latent_variables > 1:
        expected_item_sum_scores = model.expected_scores(z, return_item_scores=True).detach()
        if not model.mc_correct and rescale_by_item_score:
            expected_item_sum_scores = expected_item_sum_scores / (torch.tensor(model.modeled_item_responses) - 1)
        if bit_scores is None:
            bit_scores = bit_scores_from_z(model, z)[0]
        # item score slopes for each item
        mean_slopes = linear_regression(bit_scores, expected_item_sum_scores).t()[:, 1:]
    else:
        median, _ = torch.median(z, dim=0)
        mean_slopes = torch.zeros(z.shape[0], len(model.modeled_item_responses), z.shape[1])
        for latent_variable in range(z.shape[1]):
            z_scores = median.repeat(z.shape[0], 1)
            z_scores[:, latent_variable], _ = z[:, latent_variable].sort()
            z_scores.requires_grad_(True)
            expected_item_sum_scores = model.expected_scores(z_scores, return_item_scores=True)
            if not model.mc_correct and rescale_by_item_score:
                expected_item_sum_scores = expected_item_sum_scores / (torch.tensor(model.modeled_item_responses) - 1)

            # item score slopes for each item
            for item in range(expected_item_sum_scores.shape[1]):
                if z_scores.grad is not None:
                    z_scores.grad.zero_()
                expected_item_sum_scores[:, item].sum().backward(retain_graph=True)
                mean_slopes[:, item, latent_variable] = z_scores.grad[:, latent_variable]

        if model.latent_variables == 1:
            dbit_dz = bit_score_gradients(model, z, independent_z=1, **kwargs)
            # divide by the derivative of the bit scores with respect to the z scores
            # to get the expected item score slopes with respect to the bit scores
            mean_slopes = torch.einsum("ab...,a...->ab...", mean_slopes, 1/dbit_dz)
        else:
            mean_slopes = mean_slopes.mean(dim=0)

    return mean_slopes


def bit_information(model: BaseIRTModel, z: torch.Tensor, item: bool = True, degrees: list[int] = None, **kwargs) -> torch.Tensor:
    r"""
    Calculate the Fisher information matrix (FIM) for the z corresponding bit scores (or the information in the direction supplied by degrees).

    Parameters
    ----------
    z : torch.Tensor
        A 2D tensor containing latent variable z scores for which to compute the information. Each column represents one latent variable.
    item : bool, optional
        Whether to compute the information for each item (True) or for the test as a whole (False). (default is True)
    degrees : list[int], optional
        For multidimensional models. A list of angles in degrees between 0 and 90, one for each latent variable. Specifies the direction in which to compute the information. (default is None and returns the full FIM)
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
    :math:`\mathbf{z}`.

    The formula for the Fisher information matrix in the case of multiple parameters is:

    .. math::

        I(\mathbf{z}) = E\left[ \left(\frac{\partial \ell(X; \mathbf{z})}{\partial \mathbf{z}}\right) \left(\frac{\partial \ell(X; \mathbf{z})}{\partial \mathbf{z}}\right)^T \right] = -E\left[\frac{\partial^2 \ell(X; \mathbf{z})}{\partial \mathbf{z} \partial \mathbf{z}^T}\right]

    Where:

    - :math:`I(\mathbf{z})` is the Fisher Information Matrix.
    - :math:`\ell(X; \mathbf{z})` is the log-likelihood of :math:`X`, given the latent variable vector :math:`\mathbf{z}`.
    - :math:`\frac{\partial \ell(X; \mathbf{z})}{\partial \mathbf{z}}` is the gradient vector of the first derivatives of the log-likelihood of :math:`X` with respect to :math:`\mathbf{z}`.
    - :math:`\frac{\partial^2 \log f(X; \mathbf{z})}{\partial \mathbf{z} \partial \mathbf{z}^T}` is the Hessian matrix of the second derivatives of the log-likelihood of :math:`X` with respect to :math:`\mathbf{z}`.
    
    For additional details, see :cite:t:`Chang2017`.
    """
    if degrees is not None and len(degrees) != model.latent_variables:
        raise ValueError("There must be one degree for each latent variable.")

    probabilities = model.item_probabilities(z.clone())
    gradients = model.probability_gradients(z).detach()
    bit_z_gradients = bit_score_gradients(z, one_dimensional=False, **kwargs)
    # we divide by diagonal of the bit score gradients (the gradients in the direction of the bit score corresponding z scores)
    bit_z_gradients_diag = torch.einsum("...ii->...i", bit_z_gradients)
    gradients = (gradients.permute(1, 2, 0, 3) / bit_z_gradients_diag).permute(2, 0, 1, 3)

    # squared gradient matrices for each latent variable
    # Uses einstein summation with batch permutation ...
    squared_grad_matrices = torch.einsum("...i,...j->...ij", gradients, gradients)
    information_matrices = squared_grad_matrices / probabilities.unsqueeze(-1).unsqueeze(-1).expand_as(squared_grad_matrices)
    information_matrices = information_matrices.nansum(dim=2) # sum over item categories

    if degrees is not None and z.shape[1] > 1:
        cos_degrees = torch.tensor(degrees).float().deg2rad_().cos_()
        # For each z and item: Matrix multiplication cos_degrees^T @ information_matrix @ cos_degrees
        information = torch.einsum("i,...ij,j->...", cos_degrees, information_matrices, cos_degrees)
    else:
        information = information_matrices

    if item:
        return information
    else:
        return information.nansum(dim=1) # sum over items

def _compute_1d_bit_scores(model: BaseIRTModel, z_adjusted, start_z_adjusted, grid_start, grid_end, inverted_scale, grid_points):
    """
    Computes the 1D bit scores for the given input z scores.

    Parameters
    ----------
    model : BaseIRTModel
        The model for which to compute the bit scores.
    z_adjusted : torch.Tensor
        The adjusted z-scores. A 2D tensor.
    start_z_adjusted : torch.Tensor
        The adjusted z-scores for the starting point. A one row tensor.
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
    # Limit all z_scores to be within the grid range
    z_adjusted_capped = torch.clamp(z_adjusted, grid_start, grid_end)
    
    grid_linespace = torch.linspace(0, 1, steps=grid_points).unsqueeze(-1)
    # Use broadcasting to compute the grid
    grid = grid_start + (grid_linespace * (z_adjusted_capped - grid_start).unsqueeze(1))

    # set the entire grid for those below the grid to their z scores
    below_grid_start_mask = z_adjusted < grid_start
    grid = torch.where(below_grid_start_mask.unsqueeze(1), z_adjusted.unsqueeze(1), grid)

    # Ensure all values are larger than start_z_adjusted
    grid = torch.maximum(grid, start_z_adjusted)

    # set the last slot in each grid to the outliers z scores
    grid[:, -1, :] = torch.where(z_adjusted > grid_end, z_adjusted, grid[:, -1])
    # set the first slot in each grid to the start_z_adjusted (required when z_adjusted is an outlier)
    grid[:, 0, :] = start_z_adjusted.squeeze()

    # convert back to non-inverted z scale and compute grid entropies
    grid = grid.view(-1, grid.shape[2]) * inverted_scale
    output = model(grid)
    entropies = output_to_item_entropy(output, model.modeled_item_responses)
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

def _compute_multi_dimensional_bit_scores(model: BaseIRTModel, z_adjusted, start_z_adjusted, train_z_adjusted, grid_start, grid_end, inverted_scale, grid_points):
    """
    Computes the multi-dimensional bit scores for the given input z scores.

    Parameters
    -----------
    model : BaseIRTModel
        The model for which to compute the bit scores.
    z_adjusted : torch.Tensor
        The input z scores.
    start_z_adjusted : torch.Tensor
        The minimum z score to be used in the grid. A one row tensor.
    train_z_adjusted : torch.Tensor
        The z scores of the training data. Used for computing the median of each latent variable.
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
        The multi-dimensional bit scores for the input z scores.
    """
    # Construct grid
    ratio = torch.linspace(0, 1, steps=grid_points).view(-1, 1)
    grid = (1 - ratio) * grid_start + ratio * grid_end
    # Add z scores to the grid and sort the columns
    grid = torch.cat([grid, z_adjusted], dim=0)
    # set all z scores smaller than start_z_adjusted to start_z_adjusted
    grid = torch.max(grid, start_z_adjusted)
    # set the first slot in each grid to the start_z_adjusted
    # required when any value in z_adjusted is an outlier
    grid[0, :] = start_z_adjusted

    grid, sorted_indices = torch.sort(grid, dim=0)
    # Fill each column in grid we are not not computing bit scores for with the median
    median, _ = torch.median(train_z_adjusted, dim=0)
    bit_scores = torch.zeros_like(z_adjusted)
    for z_var in range(grid.shape[1]):
        # Only compute once per unique value
        unique_grid, inverse_indices = grid[:, z_var].unique(return_inverse=True)
        latent_variable_grid = median.repeat(unique_grid.shape[0], 1)
        latent_variable_grid[:, z_var] = unique_grid

        # Convert back to non-inverted z scale and compute grid entropies
        output = model(latent_variable_grid * inverted_scale)
        entropies = output_to_item_entropy(output, model.modeled_item_responses)

        # Compute the absolute difference between each grid point entropy and the previous one
        diff = torch.zeros_like(entropies)
        diff[1:,:] = torch.abs(entropies[:-1, :] - entropies[1:, :])

        # cummulative sum over grid points and then sum the item scores
        bit_score_grid = diff.sum(dim=1).cumsum(dim=0)

        # add duplicates, unsort and take only the bit scores for the input z scores
        bit_scores[:, z_var] = bit_score_grid[inverse_indices][torch.sort(sorted_indices[:, z_var])[1]][-z_adjusted.shape[0]:]
        
    return bit_scores


def _inverted_scales(model: BaseIRTModel, train_z):
    """
    Compute a tensor with information about whether each latent variable is positively or negatively related to the test scores.

    Parameters
    ----------
    train_z : torch.Tensor
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
    scores = model.expected_scores(train_z).sum(dim=1).reshape(-1, 1)
    linear_weights = linear_regression(train_z, scores)[1:]
    inverted_scale = torch.where(linear_weights < 0, torch.tensor(-1), torch.tensor(1)).reshape(1, -1)
    return inverted_scale

def _anti_invert_and_adjust_z_scores(z, train_z, start_z, inverted_scale):
    """
    Adjust the input and training z scores based on the starting z scores and the inverted scale.

    Parameters
    ----------
    z : torch.Tensor
        The input z scores.
    train_z : torch.Tensor
        The training z scores.
    start_z : torch.Tensor
        The starting z scores.
    inverted_scale : torch.Tensor
        The inverted scale.

    Returns
    -------
    tuple of torch.Tensor
        The adjusted input z scores, the adjusted training z scores and the adjusted starting z scores.

    Notes
    -----
    The method first anti-inverts the z scores by multiplying them with the inverted scale. 
    Then, If the anti-inverted z scores are smaller than the starting z scores, we set them to the starting z scores.
    """
    start_z_adjusted = start_z * inverted_scale
    z_adjusted = torch.max(z * inverted_scale, start_z_adjusted)
    train_z_adjusted = torch.max(train_z * inverted_scale, start_z_adjusted)

    return z_adjusted, train_z_adjusted, start_z_adjusted


def _get_grid_boundaries(train_z_adjusted: torch.Tensor, start_z_adjusted):
    """
    Determines the start and end points of the grid used for computing bit scores.

    Parameters
    ----------
    train_z_adjusted : torch.Tensor
        A 2D array containing the adjusted z scores of the training data. Each row represents one respondent, and each column represents a latent variable.
    start_z_adjusted : torch.Tensor
        A 1 row tensor containing starting z scores for the bit scores.

    Returns
    -------
    grid_start : torch.Tensor
        A 1D tensor containing the start points of the grid for each latent variable.
    grid_end : torch.Tensor
        A 1D tensor containing the end points of the grid for each latent variable.
    start_is_outlier : torch.Tensor
        A 1 row boolean tensor where each column corresponds to one latent variable. True if the starting z score is an outlier.
    """        
    outlier_detector = OutlierDetector(factor=4)
    start_is_outlier = outlier_detector.is_outlier(start_z_adjusted, data=train_z_adjusted, lower=True)[0, :]
    if any(start_is_outlier):
        smallest_non_outlier = outlier_detector.smallest_largest_non_outlier(train_z_adjusted, smallest=True)
        grid_start = torch.max(start_z_adjusted, smallest_non_outlier)
    else:
        grid_start = start_z_adjusted
    grid_end = outlier_detector.smallest_largest_non_outlier(train_z_adjusted, smallest=False)
    return grid_start, grid_end, start_is_outlier


def _get_bit_midpoints(
    model: BaseIRTModel,
    data: torch.Tensor = None,
    z: torch.Tensor = None,
    latent_variable: int = 1,
    groups: int = 10,
    population_z: torch.Tensor = None,
    z_estimation_method: str = "ML",
    **kwargs
):
    """
    Find the middle bit scores within test taker groups grouped by the specified latent variable.

    Parameters
    ----------
    model : BaseIRTModel
        The IRT model.
    data : torch.Tensor, optional
        A 2D tensor containing test data. (default is None)
    z : torch.Tensor, optional
        The latent variable z scores for the provided data. If not provided, they will be computed using z_estimation_method. (default is None)
    latent_variable: int, optional
        Specifies the latent variable along which ordering and grouping should be performed. (default is 1)
    groups: int
        The number of groups. (default is 10)
    population_z : torch.Tensor, optional
        The z scores of the population. Provide if the populations differs from the data argument. If not provided, will be assumed equal to data or z. (default is None)
    z_estimation_method : str, optional
        The method used to estimate the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP'. (default is 'ML')
    **kwargs : dict, optional
        Additional keyword arguments to be passed to bit_scores_from_z.

    Returns
    -------

    torch.Tensor, torch.Tensor
        A tensor with the group mid points and a tensor with the z corresponding bit scores.
    """
    if population_z is None and data is model.algorithm.train_data:
        population_z = z
    
    bit_scores, _ = bit_scores_from_z(
        model=model,
        z=z,
        population_z=population_z,
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
    group_mid_points = torch.tensor(
        [
            group[:, latent_variable - 1].mean()
            for group in grouped_bit
        ]
    )

    return group_mid_points, bit_scores
