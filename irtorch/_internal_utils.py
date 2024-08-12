import torch

def linear_regression(x, y):
    """
    Performs linear regression.

    Parameters
    -----------
    x : torch.Tensor
        Design matrix of shape (m, n), where m is the number of samples and n is the number of features.
    y : torch.Tensor
        Target vector of shape (m, o), where m is the number of samples and o is the number response variables. If o > 1, this implies a multivariate linear regression.

    Returns
    -----------
    w : torch.Tensor
        A tensor vector with the bias and the weights of shape (n+1, o).
    """
    # Add a bias term (1) to each sample in x
    bias = torch.cat([torch.ones(x.size(0), 1), x], dim=1)

    # Compute the weights using the normal equation
    w = torch.inverse(bias.t().mm(bias)).mm(bias.t()).mm(y)

    return w

def random_guessing_data(
    item_categories: list[int],
    size: int,
    guessing_probabilities: list[float] = None,
    mc_correct: list[int] = None
):
    """
    Create random test data based on guessing.

    Parameters
    ----------
    item_categories : list[int]
        A list of integers where each integer is the number of possible responses for the corresponding item, excluding missing responses.
    size : int
        The number of rows in the returned tensor.
    guessing_probabilities: list[float], optional
        The guessing probability for each item. The same length as the number of items. Guessing is not supported for polytomously scored items and the probabilities for them will be ignored by setting all responses to 0. (default is None and uses no guessing or, for multiple choice models, 1 over the number of item categories)
    mc_correct : list[int], optional
        Only for multiple choice data with guessing_probabilities supplied. A list of integers where each integer is the correct response for the corresponding item. If None, the data is assumed to be non multiple choice (or dichotomously scored multiple choice with only 0's and 1's). (default is None)

    Returns
    -------
    torch.Tensor
        A 2D tensor with test data. Rows are respondents and columns are items.
    """
    num_items = len(item_categories)
    if (
        guessing_probabilities is not None and \
        isinstance(guessing_probabilities, list) and \
        all(isinstance(item, float) for item in guessing_probabilities) and \
        len(guessing_probabilities) == num_items
    ):
        if not all(0 <= num < 1 for num in guessing_probabilities):
            raise ValueError("The guessing probabilities must be between 0 and 1.")
    else:
        raise ValueError("guessing_probabilities must be list of floats with the same length as the number of items.")


    if guessing_probabilities is None:
        # Use default guessing probabilities
        guessing_probabilities = [1.0 / cats for cats in item_categories]

    # Initialize the response matrix with zeros
    response_matrix = torch.zeros(size, num_items)

    # Process each item
    for item_idx, num_categories in enumerate(item_categories):
        guessing_prob = guessing_probabilities[item_idx]

        if num_categories == 2:
            # Dichotomous item, generate responses based on the guessing probability
            responses = torch.bernoulli(torch.full((size,), guessing_prob))
        else:
            if mc_correct is not None:
                # Create a tensor to hold probabilities for all choices
                probs = torch.full((num_categories,), (1 - guessing_prob) / (num_categories - 1))
                
                # Assign the guessing probability to the correct option
                probs[mc_correct[item_idx]] = guessing_prob

                # Now, for the incorrect answers, we distribute the remaining probability
                # We've taken the guessing_prob for the correct answer, so we distribute what's left
                incorrect_total_prob = 1.0 - guessing_prob
                probs[probs != guessing_prob] = incorrect_total_prob / (num_categories - 1)  # distribute among others

                # Randomly choose options based on the probabilities
                responses = torch.multinomial(probs.repeat(size, 1), 1).squeeze()
            else:
                # Polytomous item which is not mcmc, keep at 0
                responses = torch.zeros(size, dtype=torch.long)

        response_matrix[:, item_idx] = responses

    return response_matrix

def conditional_score_distribution(
    probabilities: torch.Tensor,
    item_categories: list[int],
):
    """
    Compute total score distribution conditional on theta as by (Thissen et. al. 1995).

    Parameters
    ----------
    probabilities : torch.Tensor
        A 3D tensor containing item score probabilities for each theta quadrature point.
        The first dimension represents the quadrature points, the second dimension represents the items and the third dimension represents the item categories.
    item_categories : list[list]
        A list of integers where each integer is the number of possible responses for the corresponding item.

    Returns
    -------
    torch.Tensor
        A 2D torch tensor of total score probabilities, rows being quadrature theta points and columns being total scores. (rows sum to 1)
    """
    no_items = probabilities.shape[1]
    q_points = probabilities.shape[0]
    max_score = sum(item_categories) - no_items

    # Rows are quadrature points, columns are total scores
    nsprobs = torch.zeros(q_points, max_score + 1).to(probabilities.device)
    nsprobs[:, : item_categories[0]] = probabilities[:, 0, :item_categories[0]]

    if no_items > 1:
        for item in range(1, no_items):
            item_max = sum(item_categories[: item + 1]) - len(
                item_categories[: item + 1]
            )
            sprobs = nsprobs.clone()
            nsprobs = torch.zeros_like(nsprobs)
            for j in range(item_max - item_categories[item] + 2):
                nsprobs[:, j : item_categories[item] + j] = (
                    torch.t(sprobs[:, j] * torch.t(probabilities[:, item, :item_categories[item]]))
                    + nsprobs[:, j : item_categories[item] + j]
                )
    return nsprobs


def sum_incorrect_probabilities(
    probabilities: list[torch.Tensor],
    item_responses: list[int],
    mc_correct: list[int],
):
    """
    Sum incorrect score probabilities for multiple choice items. Useful for approximating sum scores.

    Parameters
    ----------
    probabilities : list[torch.Tensor]
        A list of 2D tensors containing item score probabilities for each item.
        Rows are theta quadrature points from the theta density and columns correspond to item responses. (rows sum to 1)
    item_responses : list[int]
        A list of integers where each integer is the number of possible responses for the corresponding item.
    mc_correct : list[int]
        A list of integers where each integer is correct response for the corresponding item.

    Returns
    -------
    torch.Tensor
        A 2D torch tensor correct/incorrect response probabilities
    """
    new_probs = torch.zeros(probabilities.shape[0], probabilities.shape[1], 2)
    for item in range(0, len(item_responses)):
        item_score_0_probs = torch.cat(
            (
                probabilities[:, item, :mc_correct[item]],
                probabilities[:, item, mc_correct[item]+1:]
            ),
            dim=1,
        ).sum(dim=1)
        new_probs[:, item, :] = torch.cat(
            (
                item_score_0_probs.unsqueeze(1),
                probabilities[:, item, mc_correct[item]].unsqueeze(1),
            ),
            dim=1,
        )
    return new_probs


def entropy(probabilities: torch.Tensor, log_base: int = 2):
    """
    Calculate the entropy of a set of probabilities.

    The entropy is a measure of the uncertainty or randomness of a set of probabilities. It is calculated as the sum of the product of each probability and its surprisal (negative log probability).

    Parameters
    -------------
    probabilities: torch.Tensor
        A tensor of probabilities. The last tensor dimension should represent a discrete probability distribution and should sum to 1.
    log_base: int, optional
        The base of the logarithm used in the calculation. (default is 2, which gives entropy in bits)

    Returns
    -------------
    entropy: torch.Tensor
        The entropy of each row of probabilities.
    """
    if not torch.allclose(probabilities.sum(dim=-1), torch.tensor(1.0)):
        raise RuntimeError("Each row of probabilities must sum to 1.")

    surprisal = -torch.log(probabilities) / torch.log(torch.tensor(log_base))
    surprisal[surprisal.isinf()] = 0 # is surprisal is Inf, set to finite value to get correct entropy instead of NaN
    entropies = (probabilities * surprisal).sum(dim=-1)
    return entropies

def is_jupyter():
    try:
        from IPython import get_ipython
        if "IPKernelApp" not in get_ipython().config:
            raise ImportError("Not in IPython")
        return True
    except Exception:
        return False

def dynamic_print(string_to_print):
    """
    Dynamically update terminal printout.

    Parameters
    ----------
    *args : str
        Strings to print

    """
    formatted_string = f"\r{string_to_print} " # small space after to make it look better in terminal
    print(formatted_string, end="", flush=True)

def one_hot_encode_test_data(
    data: torch.Tensor, item_categories: list
):
    """
    One-hot encodes test data for each item based on the number of response categories for that item. Missing item responses need to be coded as -1 or nan.

    Parameters
    ----------
    data : torch.Tensor
        A 2D tensor where each row represents one respondent and each column represents an item.
        The values should be the scores achieved by the respondents on the items, starting from 0.
        Missing item responses need to be coded as -1 or 'nan'.
    item_categories : list
        A list of integers where each integer is the number of response categories for the corresponding item.

    Returns
    -------
    torch.Tensor
        A 2D tensor where each group of columns corresponds to the one-hot encoded scores for one of the items.
        The number of columns in each group is equal to the maximum possible score plus one for that item.

    Notes
    -----
    The input data tensor should contain integer values representing the scores. If it contains non-integer values,
    they will be rounded to the nearest integer.
    """
    if data.dim() != 2:
        raise ValueError("Input data must be a 2D tensor.")
    if data.shape[1] != len(item_categories):
        raise ValueError(
            "The number of columns in the data tensor must match the length of the item_categories list."
        )
    if data.isnan().any():
        data[data.isnan()] = -1

    one_hot_list = []
    if data.dtype != torch.long:
        data = data.round().long()

    for i in range(data.shape[1]):
        one_hot = torch.zeros((data.shape[0], item_categories[i]), dtype=torch.long).to(
            data.device
        )
        # Fill in the appropriate column with ones based on the scores
        # Only for those rows where the score is not -1
        valid_rows = data[:, i] != -1
        one_hot[valid_rows, data[valid_rows, i]] = 1
        # Append the one-hot encoded tensor to the list
        one_hot_list.append(one_hot)

    # Concatenate the one-hot encoded columns back into a single tensor
    return torch.cat(one_hot_list, dim=1).float()

def get_missing_mask(data: torch.Tensor) -> torch.Tensor:
    """
    Get a mask for missing values in the data.

    Parameters
    ----------
    data : torch.Tensor
        The data tensor.

    Returns
    -------
    torch.Tensor
        A tensor of the same shape as the input data tensor, with 1s where the data is missing and 0s where it is not.
    """
    return (data == -1) | data.isnan()

@torch.inference_mode()
def impute_missing_internal(
    data: torch.Tensor,
    method: str = "zero",
    mc_correct: list[int] = None,
    item_categories: list[int] = None,
) -> torch.Tensor:
    """
    Impute missing values. Separate from the external version to not rely on a fitted model. 

    Parameters
    ----------
    data : torch.Tensor
        A 2D tensor where each row is a response vector and each column is an item.
    method : str, optional
        The imputation method to use. Options are 'zero', 'mean'. (default is 'zero')

        - 'zero': Impute missing values with 0.
        - 'mean': Impute missing values with the item means.
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
    else:
        raise ValueError(
            f"{method} imputation is not implmented"
        )
        

    return imputed_data