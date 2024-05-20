import logging
import torch
import copy
import pandas as pd
from itertools import product
import torch.multiprocessing as mp
from irtorch.irt import IRT

logger = logging.getLogger("irtorch")

def cross_validation(
    irt_model: IRT,
    data: torch.Tensor,
    folds: int,
    params_grid: dict,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    z_estimation_method: str = "ML",
    **kwargs
) -> pd.DataFrame:
    """
    Perform cross-validation on the given model and data. Uses log-likelihood for model evaluation. Note that for running on the CPU on windows, `if __name__ == '__main__':` needs to be added to the main script before calling this function, see examples.

    Parameters
    ----------
    irt_model : IRT
        The irt model to train. Note that this should be an untrained model.
    data : torch.Tensor
        The data to use for cross-validation. The data is randomly shuffled before splitting into folds.
    folds : int
        The number of folds to use for cross-validation.
    params_grid : dict
        The hyperparameters to use for cross-validation. All need to be arguments for the model fit method.
    z_estimation_method : str, optional
        Method used to obtain the z scores. Also used for bit scores as they require the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'ML')
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
    This example demonstrates how to use cross_validation() function with the Swedish National Mathematics dataset.

    First, we import necessary modules, load the data and split it into a training and testing set:

    >>> from irtorch import IRT
    >>> from irtorch.load_dataset import swedish_national_mathematics_2
    >>> from irtorch.utils import split_data
    >>> from irtorch.cross_validation import cross_validation
    >>> data_math = swedish_national_mathematics_2()
    >>> train_data, test_data = split_data(data_math, 0.8)

    Next, we initialize the IRT model:

    >>> irt_model = IRT(data=data_math)

    We then set up a grid of parameters for cross-validation:

    >>> params_grid = {
    ...     'learning_rate': [0.05, 0.1],
    ...     'batch_size': [64, 128],
    ... }

    Finally, we perform cross-validation to find a good set of parameters:

    >>> if __name__ == '__main__':
    ...     result = cross_validation(irt_model, data=train_data, folds=5, params_grid=params_grid, z_estimation_method='NN', device='cpu')
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this machine, use device = 'cpu'.")

    # randomly scrable the data
    data = data[torch.randperm(data.shape[0])]
    data_folds = torch.chunk(data, folds, dim=0)

    param_combinations = list(product(*params_grid.values()))
    param_comb_names = list(params_grid.keys())
    param_dicts = [
        {name: value for name, value in zip(param_comb_names, combination)}
        for combination in param_combinations
    ]
    # Add additional kwargs to each parameter dictionary
    for param_dict in param_dicts:
        param_dict.update(kwargs)

    logger.info(f"Performing cross-validation with {len(param_dicts)} parameter combinations")

    # Prepare arguments for multiprocessing
    jobs = []
    for params in param_dicts:
        for fold in range(folds):
            train_data = torch.cat([data_folds[i] for i in range(folds) if i != fold])
            validation_data = data_folds[fold]
            jobs.append((copy.deepcopy(irt_model), train_data, validation_data, params, z_estimation_method, device))

    if device == "cpu":
        cores_to_use = mp.cpu_count()
        print(f"Using {cores_to_use} cores")
        original_threads = torch.get_num_threads()
        try:
            with mp.Pool(cores_to_use) as pool:
                results = pool.starmap(_cv_fold, jobs)
        finally:
            torch.set_num_threads(original_threads)

    elif device == "cuda":
        results = []
        for job in jobs:
            results.append(_cv_fold(*job))

    # average over folds
    results = pd.DataFrame(results)
    results.drop(list(kwargs.keys()), axis=1, inplace=True)
    results = results.groupby(param_comb_names).mean().reset_index()
    return results

def _cv_fold(irt_model : IRT, train_data, validation_data, params, z_estimation_method, device):
    if device == "cpu":
        torch.set_num_threads(1) # One thread per core, to avoid overloading the CPU

    irt_model.fit(train_data, device=device, **params)
    log_likelihood = irt_model.log_likelihood(validation_data, z_estimation_method = z_estimation_method, reduction="sum").item()

    return {**params, "log_likelihood": log_likelihood}
