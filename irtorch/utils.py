import logging
import random
import copy
from itertools import product
import torch
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
from irtorch.models import BaseIRTModel
from irtorch.timer import start_timer, end_timer_and_print

__all__ = ["cross_validation", "gauss_hermite", "get_item_categories", "impute_missing", "fit_multiple_models_cpu", "split_data", "set_seed", "imv"]

logger = logging.getLogger("irtorch")

def _prepare_cross_validation_data(data: torch.Tensor, folds: int) -> list[torch.Tensor]:
    """Prepare data for cross-validation by shuffling and splitting into folds."""
    shuffled_data = data[torch.randperm(data.shape[0])]
    return torch.chunk(shuffled_data, folds, dim=0)

def _create_param_grid(params_grid: dict, kwargs: dict) -> list[dict]:
    """Create list of parameter combinations from grid."""
    param_combinations = list(product(*params_grid.values()))
    param_names = list(params_grid.keys())
    param_dicts = [
        {name: value for name, value in zip(param_names, combination)}
        for combination in param_combinations
    ]
    for param_dict in param_dicts:
        param_dict.update(kwargs)
    return param_dicts

def cross_validation(
    model: BaseIRTModel,
    data: torch.Tensor,
    folds: int,
    params_grid: dict,
    theta_estimation: str = "ML",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    cores_to_use: int = None,
    multiprocessing: bool = True,
    **kwargs
) -> pd.DataFrame:
    r"""
    Perform cross-validation on the given model and data. Uses log-likelihood for model evaluation. Note that for running on the CPU on windows, `if __name__ == '__main__':` needs to be added to the main script before calling this function, see examples.

    Parameters
    ----------
    model : IRT
        The irt model to train. Note that this should be an untrained model.
    data : torch.Tensor
        The data to use for cross-validation. The data is randomly shuffled before splitting into folds.
    folds : int
        The number of folds to use for cross-validation.
    params_grid : dict
        The hyperparameters to use for cross-validation. All need to be arguments for the model fit method.
    theta_estimation : str, optional
        Method used to obtain the theta scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'ML')
    device : str, optional
        The device to use for training. Can be 'cpu' for CPU or 'cuda' for GPU (if available). The default is 'cuda' if a GPU is available, otherwise 'cpu'.
    **kwargs
        Additional keyword arguments to pass to the model fit method.

    Returns
    -------
    list
        A list of dictionaries containing the hyperparameters and the corresponding cross-validation scores.

    Examples
    --------
    >>> # Import necessary modules, load the data and split it into a training and testing set:
    >>> import irtorch
    >>> from irtorch.models import GeneralizedPartialCredit
    >>> from irtorch.estimation_algorithms import MML
    >>> from irtorch.utils import split_data, cross_validation
    >>> data_math = irtorch.load_dataset.swedish_national_mathematics_2()
    >>> train_data, test_data = split_data(data_math, 0.8)
    >>> # Initialize the IRT model:
    >>> model = GeneralizedPartialCredit(data=data_math)
    >>> # Set up a grid of parameters for cross-validation:
    >>> params_grid = {
    ...     "learning_rate": [0.1, 0.15],
    ...     "learning_rate_update_patience": [3, 7],
    ... }
    >>> # Perform cross-validation to find a good set of parameters:
    >>> if __name__ == '__main__':
    >>>     result = cross_validation(
    ...         model,
    ...         data=train_data,
    ...         folds=5,
    ...         params_grid=params_grid,
    ...         theta_estimation='ML',
    ...         device='cpu',
    ...         algorithm = MML(),
    ...         multiprocessing=True
    ...     )
    >>>     print(result)
    >>>     print(f"--------------------\nBest hyperparameters\n--------------------")
    >>>     best_params = result.loc[result["log_likelihood"].idxmax(), ["learning_rate", "learning_rate_update_patience"]]
    >>>     print(f"learning rate: {best_params['learning_rate']} \nlearning rate update patience: {best_params['learning_rate_update_patience']}")
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this machine, use device = 'cpu'.")

    data_folds = _prepare_cross_validation_data(data, folds)
    param_dicts = _create_param_grid(params_grid, kwargs)
    
    logger.info(f"Performing cross-validation with {len(param_dicts)} parameter combinations")
    start_timer()
    
    results = (
        _run_multiprocessing(model, data_folds, param_dicts, theta_estimation, device, cores_to_use)
        if device == "cpu" and multiprocessing
        else _run_sequential(model, data_folds, param_dicts, theta_estimation, device)
    )

    end_timer_and_print("Cross-validation completed")
    
    results_df = pd.DataFrame(results)
    results_df.drop(list(kwargs.keys()), axis=1, inplace=True)
    return results_df

def _run_multiprocessing(model, data_folds, param_dicts, theta_estimation, device, cores_to_use):
    if cores_to_use is None:
        cores_to_use = min(mp.cpu_count(), len(param_dicts))
    
    try:
        logger.info(f"Attempting to initialize multiprocessing with {cores_to_use} cores")
        mp.set_start_method('spawn', force=True)
        
        jobs = []
        for params in param_dicts:
            jobs.append((copy.deepcopy(model), data_folds, params, theta_estimation, device))
        
        original_threads = torch.get_num_threads()
        try:
            threads_per_worker = max(1, original_threads // cores_to_use)
            torch.set_num_threads(threads_per_worker)
            
            with mp.Pool(cores_to_use) as pool:
                results = pool.starmap(_cv_param_combination, jobs)
            
            logger.info("Multiprocessing completed successfully")
            
        finally:
            torch.set_num_threads(original_threads)
            
    except Exception as e:
        logger.warning(f"Multiprocessing initialization failed with error: {str(e)}. Falling back to sequential processing.")
        return _run_sequential(model, data_folds, param_dicts, theta_estimation, device)

    return results

def _run_sequential(model, data_folds, param_dicts, theta_estimation, device):
    logger.info("Using sequential processing")
    return [
        _cv_param_combination(
            copy.deepcopy(model), 
            data_folds, 
            params, 
            theta_estimation, 
            device
        )
        for params in param_dicts
    ]

def _cv_param_combination(model, data_folds, params, theta_estimation, device):
    """Worker function for cross-validation parameter combinations."""
    try:
        results = []
        for i, validation_fold in enumerate(data_folds):
            train_folds = [x for j, x in enumerate(data_folds) if j != i]
            train_data = torch.cat(train_folds, dim=0)
            
            # Create a fresh copy of the params for each fold
            fold_params = params.copy()
            
            # Handle algorithm separately to avoid it being included in averaging
            algorithm = None
            if "algorithm" in fold_params:
                algorithm = copy.deepcopy(fold_params.pop("algorithm"))
            
            result = _cv_fold(model, train_data, validation_fold, fold_params, theta_estimation, device, algorithm)
            
            # Add algorithm back to result if it was present
            if algorithm is not None:
                result["algorithm"] = algorithm
                
            results.append(result)
        
        # Average results across folds, excluding non-numeric values
        avg_result = {}
        for k in results[0].keys():
            if k == "algorithm":
                avg_result[k] = results[0][k]  # Just take the first one
            else:
                avg_result[k] = sum(d[k] for d in results) / len(results)
        
        return avg_result
    except Exception as e:
        logger.error(f"Error in _cv_param_combination: {str(e)}")
        raise

def _cv_fold(irt_model, train_data, validation_data, params, theta_estimation, device, algorithm=None):
    """Worker function for a single cross-validation fold."""
    try:
        # Fit model using provided parameters
        irt_model.fit(train_data, device=device, algorithm=algorithm, **params)
        log_likelihood = irt_model.evaluate.log_likelihood(validation_data, theta_estimation=theta_estimation, reduction="sum").item()

        return {**params, "log_likelihood": log_likelihood}
    except Exception as e:
        logger.error(f"Error in _cv_fold: {str(e)}")
        raise


def gauss_hermite(n, mean, covariance):
    r"""
    Calculate the Gauss-Hermite quadrature points and weights for a multivariate normal distribution.

    Parameters
    ----------
    n : int
        Number of quadrature points.
    mean : torch.Tensor
        Mean of the distribution.
    covariance : torch.Tensor
        Covariance matrix of the distribution.

    Returns
    -------
    x : torch.Tensor
        Quadrature points.
    w : torch.Tensor
        Quadrature weights.

    Notes
    -----
    To integrate a function against a multivariate Gaussian distribution, one can employ Gauss-Hermite quadrature with an appropriate change of variables. Beginning with the integral:

    .. math::
        \int \frac{1}{\sqrt{\operatorname{det}(2 \pi \mathbf{\Sigma})}} e^{-\frac{1}{2}(\mathbf{y}-\boldsymbol{\mu})^{\top} \mathbf{\Sigma}^{-1}(\mathbf{y}-\boldsymbol{\mu})} f(\mathbf{y}) d \mathbf{y}

    We use the transformation :math:`\mathbf{x}=\frac{1}{\sqrt{2}} \mathbf{L}^{-1}(\mathbf{y}-\boldsymbol{\mu})`, where :math:`\boldsymbol{\Sigma}=\mathbf{L} \mathbf{L}^{\top}`, leading to:

    .. math::
        \int \frac{1}{\sqrt{\operatorname{det}(2 \pi \mathbf{\Sigma})}} e^{\mathbf{x}^{\top} \mathbf{x}} f(\sqrt{2} \mathbf{L} \mathbf{x}+\boldsymbol{\mu}) \operatorname{det}(\sqrt{2} \mathbf{L}) d \mathbf{x}=\int \pi^{-\frac{N}{2}} e^{-\mathbf{x}^{\top} \mathbf{x}} f(\sqrt{2} \mathbf{L} \mathbf{x}+\boldsymbol{\mu}) d \mathbf{x}

    For an :math:`N`-dimensional vector :math:`\boldsymbol{x}`, the integral can be decomposed into :math:`N` nested Gauss-Hermite integrals, since the inner product in the exponent, :math:`\exp \left(\sum_{n=1}^N x_n^2\right)`, can be represented as a product.

    Examples
    --------
    Computing mean and variance of a multivariate normal distribution using 8 Gauss-Hermite quadrature points:

    >>> import torch
    >>> from irtorch.utils import gauss_hermite
    >>> mu = torch.tensor([1, 0])
    >>> cov = torch.tensor([[1.3, -0.213], [-0.213, 1.2]])
    >>> points, weights = gauss_hermite(8, mu, cov)
    >>> mean = torch.sum(weights[:, None] * points, dim=0)
    >>> variance = torch.sum(weights[:, None] * (points - mu)**2, dim=0)
    >>> print(f"Mean: {mean}")
    >>> print(f"Variance: {variance}")
    """
    mvn_dim = len(mean) #dimesions of the multivariate normal distribution
    x, w = np.polynomial.hermite.hermgauss(n)
    x = torch.tensor(x, dtype=torch.float64)
    w = torch.tensor(w, dtype=torch.float64)
    const = torch.pi**(-0.5 * mvn_dim)
    xn = torch.tensor(list(product(*(x,)*mvn_dim)), dtype=torch.float64)
    wn = torch.prod(torch.tensor(list(product(*(w,)*mvn_dim)), dtype=torch.float64), dim=1)
    chol_decomp = torch.linalg.cholesky(covariance).to(torch.float64)
    # Transformation of the quadrature points
    # See change of variables in the docstring
    yn = 2.0**0.5 * (chol_decomp @ xn.T).T + mean[None, :]
    return yn, wn * const

def get_item_categories(data: torch.Tensor):
    """
    Get the number of possible responses for each item in the data.

    Parameters
    -----------
    data : torch.Tensor
        A 2D tensor where each row represents one respondent and each column represents an item.
        The values should be the scores/possible responses on the items, starting from 0.
        Missing item responses need to be coded as -1 or 'nan'.

    Returns
    ----------
    list
        A list of integers where each integer is the number of possible responses for the corresponding item.
    """
    return [int(data[~data.isnan().any(dim=1)][:, col].max()) + 1 for col in range(data.shape[1])]

@torch.no_grad()
def impute_missing(
    data: torch.Tensor,
    method: str = "zero",
    model: BaseIRTModel = None,
    mc_correct: list[int] = None,
    item_categories: list[int] = None
) -> torch.Tensor:
    """
    Impute missing values.

    Parameters
    ----------
    data : torch.Tensor
        A 2D tensor where each row is a response vector and each column is an item.
    method : str, optional
        The imputation method to use. Options are 'zero', 'mean', 'random_incorrect'. (default is 'zero')

        - 'zero': Impute missing values with 0.
        - 'mean': Impute missing values with the item means.
        - 'random incorrect': Impute missing values with a random incorrect response. This method is only valid for multiple choice data.
        - 'prior expected': Impute missing values with the expected scores for the latent space prior distribution mean.
    model : BaseIRTModel, optional
        Only for method='random_incorrect' or 'prior expected'. The IRT model to use for imputation. (default is None)
    mc_correct : list[int], optional
        Only for method='random_incorrect'. A list of integers where each integer is the correct response for the corresponding item. If None, the data is assumed to be non multiple choice (or dichotomously scored multiple choice with only 0's and 1's). (default is None)
    item_categories : list[int], optional
        Only for method='random_incorrect'. A list of integers where each integer is the number of possible responses for the corresponding item. If None, the number of possible responses is calculated from the data. (default is None)
    """
    imputed_data = data.clone()

    if (imputed_data == -1).any():
        imputed_data[imputed_data == -1] = torch.nan

    if method == "zero":
        imputed_data = torch.where(torch.isnan(imputed_data), torch.tensor(0.0), imputed_data)
    elif method == "mean":
        means = imputed_data.nanmean(dim=0)
        mask = torch.isnan(imputed_data)
        imputed_data[mask] = means.repeat(data.shape[0], 1)[mask]
    elif method == "random incorrect":
        if model is not None:
            if model.mc_correct is not None:
                raise ValueError("The model provided must be a multiple choice item model when using random_incorrect imputation")
            item_categories = model.item_categories
            mc_correct = model.mc_correct
        else:
            if mc_correct is None:
                raise ValueError("mc_correct must be provided when using random_incorrect imputation without a model")
            if item_categories is None:
                item_categories = (torch.where(~imputed_data.isnan(), data, torch.tensor(float('-inf'))).max(dim=0).values + 1).int().tolist()

        for col in range(imputed_data.shape[1]):
            # Get the incorrect non-missing responses from the column
            incorrect_responses = torch.arange(0, item_categories[col], device=imputed_data.device).float()
            incorrect_responses = incorrect_responses[incorrect_responses != mc_correct[col]]
            # Find the indices of missing values in the column
            missing_indices = (imputed_data[:, col].isnan()).squeeze()
            # randomly sample from the incorrect responses and replace missing
            imputed_data[missing_indices, col] = incorrect_responses[torch.randint(0, incorrect_responses.size(0), (missing_indices.sum(),))]
    elif method == "prior expected":
        if model is None:
            raise ValueError("The model must be provided when using prior mean imputation")
        if model.mc_correct is not None:
            raise ValueError("The model provided must be a non-multiple choice item model when using prior mean imputation")
        prior_scores = model.expected_scores(
            torch.zeros(1, model.latent_variables).to(next(model.parameters()).device),
            return_item_scores=True
        ).round()
        mask = torch.isnan(imputed_data)
        imputed_data[mask] = prior_scores.repeat(imputed_data.shape[0], 1).to(
            next(model.parameters()).device
        )[mask]
    else:
        raise ValueError(
            f"{method} imputation is not implmented"
        )

    return imputed_data

@torch.no_grad()
def imv(
    model1: BaseIRTModel,
    model2: BaseIRTModel,
    data: torch.Tensor,
    model1_theta: torch.Tensor = None,
    model2_theta: torch.Tensor = None,
    max_iterations = 100,
    **kwargs
):
    r"""
    Quantifies the improvement in model predictions from model 2 over model 1 by computing the InterModel Vigorish (IMV) metric by :cite:t:`Domingue2024`.
    The IMV is the expected gain if one were to place fair bets on model 1 making correct predictions on the observed data,
    but the actual predictions are made by model 2.
    
    Parameters
    ----------
    model1 : BaseIRTModel
        The first model to compare.
    model2 : BaseIRTModel
        The second model to compare.
    data : torch.Tensor, optional
        The data to use for evaluating the models. Typically not the same as the model training data to avoid overfitting.
    model1_theta : torch.Tensor, optional
        The latent scores for the first model. If None, the latent scores are calculated using the first model. (default is None)
    model2_theta : torch.Tensor, optional
        The latent scores for the second model. If None, the latent scores are calculated using the second model. (default is None)
    max_iterations : int, optional
        The maximum number of iterations for the Newton-Raphson method. (default is 100)
    **kwargs
        Additional keyword arguments used for theta estimation. Refer to :meth:`irtorch.models.BaseIRTModel.latent_scores` for additional details.

    Returns
    -------
    torch.Tensor
        The IMV value. A positive value indicates that model 2 predicts better than model 1.

    Examples
    --------
    >>> import irtorch
    >>> from irtorch.estimation_algorithms import MML
    >>> from irtorch.models import OneParameterLogistic, TwoParameterLogistic, ThreeParameterLogistic
    >>> data = irtorch.load_dataset.swedish_sat_binary()[:, :80]
    >>> train_data, test_data = irtorch.split_data(data, 0.8)
    >>> model1 = OneParameterLogistic(train_data)
    >>> model2 = TwoParameterLogistic(train_data)
    >>> model3 = ThreeParameterLogistic(train_data)
    >>> model1.fit(train_data, MML())
    >>> model2.fit(train_data, MML())
    >>> model3.fit(train_data, MML())
    >>> imv12 = irtorch.utils.imv(model1, model2, test_data)
    >>> imv23 = irtorch.utils.imv(model2, model3, test_data)
    >>> imv13 = irtorch.utils.imv(model1, model3, test_data)
    >>> print(f"IMV 1-2: {imv12:.4f}")
    >>> print(f"IMV 2-3: {imv23:.4f}")
    >>> print(f"IMV 1-3: {imv13:.4f}")
    """
    # Same as the log of the geometric mean of the likelihoods
    log_geometric_mean1 = model1.evaluate.log_likelihood(data, model1_theta, reduction="mean", **kwargs)
    log_geometric_mean2 = model2.evaluate.log_likelihood(data, model2_theta, reduction="mean", **kwargs)
    # Probability of the entropy that leads to the geometric mean
    log_geometric_means = torch.stack((log_geometric_mean1, log_geometric_mean2))
    w = _imv_newton_raphson(log_geometric_means, max_iter=max_iterations)
    imv = (w[1]-w[0])/w[0]
    return imv

def _imv_newton_raphson(negative_entropy, max_iter=100, tol=1e-6):
    negative_entropy = negative_entropy.clone().detach()
    w = torch.full_like(negative_entropy, 0.75)
    for _ in range(max_iter):
        fx = w * torch.log(w) + (1 - w) * torch.log(1 - w) - negative_entropy
        fpx = torch.log(w) - torch.log(1 - w)
        
        # Avoid division by zero
        fpx = torch.where(fpx == 0, torch.tensor(1e-6), fpx)
        
        delta_x = -fx / fpx
        w = w + delta_x
        w = torch.clamp(w, 0.5 + 1e-6, 1 - 1e-6)
        if torch.max(torch.abs(delta_x)) < tol:
            break
    return w


def fit_multiple_models_cpu(
    models: list[BaseIRTModel],
    train_data: torch.Tensor,
    cores_to_use: int = None,
    **kwargs
) -> list[BaseIRTModel]:
    """
    Train multiple models on the same data using multiprocessing.

    Parameters
    ----------
    models : BaseIRTModel
        The IRT models to train. Note that these should be untrained model instances.
    train_data : torch.Tensor
        The data to use for training.
    cores_to_use : int, optional
        Number of cores to use for multiprocessing when device is 'cpu'. (default uses one per model)
    **kwargs
        Additional keyword arguments to pass to the model fit method.

    Returns
    -------
    list[BaseIRTModel]
        A list of trained models.

    Examples
    --------
    >>> from irtorch.models import GeneralizedPartialCredit
    >>> from irtorch.estimation_algorithms import VAE
    >>> from irtorch.load_dataset import swedish_national_mathematics_1
    >>> from irtorch.utils import train_multiple_models_cpu
    >>> data = swedish_national_mathematics_1()
    >>> # train 3 models with 1, 2 and 3 latent variables
    >>> models = [GeneralizedPartialCredit(latent_variables = i+1, data=data) for i in range(3)]
    >>> if __name__ == '__main__':
    ...     train_multiple_models_cpu(models, data, algorithm = VAE())
    """
    if cores_to_use is None:
        cores_to_use = min(len(models), mp.cpu_count())

    print(f"Using {cores_to_use} cores")
    original_threads = torch.get_num_threads()
    try:
        # Ensure each subprocess uses a single thread for PyTorch.
        torch.set_num_threads(1)
        with mp.Pool(cores_to_use) as pool:
            trained_models = pool.starmap(_fit_model, [(model, train_data, kwargs) for model in models])
    finally:
        torch.set_num_threads(original_threads)

    return trained_models

def _fit_model(model: BaseIRTModel, train_data: torch.Tensor, kwargs):
    """
    This is a worker function used in multiprocessing
    """
    model.fit(train_data, device="cpu", **kwargs)
    return model

def split_data(data, train_ratio=0.8, shuffle=True):
    """
    Splits a tensor into training and testing datasets.

    Parameters
    ----------
    data : torch.Tensor
        The dataset to be split. It should be a tensor where the first dimension corresponds to the number of samples.
    train_ratio : float, optional
        The proportion of the dataset to include in the train split. This should be a decimal representing
        the percentage of data used for training (e.g., 0.8 for 80% training and 20% testing). The default is 0.8.
    shuffle : bool, optional
        Whether to shuffle the data before splitting. It is highly recommended to shuffle the data
        to ensure that the training and testing datasets are representative of the overall dataset. The default is True.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing two tensors. The first tensor is the training dataset, and the second tensor is the testing dataset.
        The split is based on the `train_ratio` parameter, and if `shuffle` is True, the split is performed on the shuffled data.

    Examples
    --------
    >>> data = torch.rand(100, 10)  # Example tensor with 100 samples and 10 features each
    >>> train_data, test_data = split_tensor(data, train_ratio=0.8, shuffle=True)
    >>> print("Training data size:", train_data.shape)
    >>> print("Testing data size:", test_data.shape)
    """
    if shuffle:
        indices = torch.randperm(data.size(0))
        data_shuffled = data[indices]
    else:
        data_shuffled = data

    train_size = int(data.size(0) * train_ratio)
    train_data = data_shuffled[:train_size]
    test_data = data_shuffled[train_size:]

    return train_data, test_data

def set_seed(seed: int):
    """
    Seed all random number generators for reproducibility (random, numpy and torch).

    Parameters
    ----------
    seed : int
        The seed value to use for seeding the random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU, but also works for single.
    # CuDNN (CUDA Deep Neural Network library) provides highly optimized implementations for many neural network operations.
    # Some of these implementations are non-deterministic due to parallelism and optimization techniques.
    # Setting torch.backends.cudnn.deterministic = True ensures that CuDNN selects deterministic algorithms, eliminating randomness from these operations.
    torch.backends.cudnn.deterministic = True
    # CuDNN's benchmarking feature selects the most efficient algorithm for your specific hardware configuration, but introduces randomness
    torch.backends.cudnn.benchmark = False
