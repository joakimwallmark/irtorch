import torch

def linear_regression(x, y):
    """
    Performs linear regression.

    Parameters
    -----------
    x : torch.Tensor
        Design matrix of shape (m, n), where m is the number of samples and n is the number of features.
    y : torch.Tensor
        Target vector of shape (m, 1).

    Returns
    -----------
    w : torch.Tensor
        A tensor vector with the bias and the weights of shape (n+1, 1).
    """
    # Add a bias term (1) to each sample in x
    bias = torch.cat([torch.ones(x.size(0), 1), x], dim=1)

    # Compute the weights using the normal equation
    w = torch.inverse(bias.t().mm(bias)).mm(bias.t()).mm(y)

    return w


def impute_missing(data: torch.tensor, mc_correct: list[int] = None, item_categories: list[int] = None):
    """
    Impute missing values in the data. For multiple choice data for which missing is not modeled, imputes randomly from incorrect responses.

    Parameters
    -----------
    data : torch.Tensor
        A 2D tensor where each row represents one respondent and each column represents an item.
        The values should be the scores/possible responses on the items, starting from 0.
        Missing item responses need to be coded as -1 or 'nan'.
    mc_correct : list[int], optional
        A list of integers where each integer is the correct response for the corresponding item. If None, the data is assumed to be non multiple choice (or dichotomously scored multiple choice with only 0's and 1's). (default is None)
    item_categories : list[int], optional
        A list of integers where each integer is the number of possible responses for the corresponding item. If None, the number of possible responses is calculated from the data. (default is None)

    Returns
    ----------
    torch.Tensor
        A 2D tensor with missing values imputed. Rows are respondents and columns are items.
    """
    if data.isnan().any():
        data[data.isnan()] = -1

    if mc_correct is None:
        data[data==-1] = 0
        return data
    else: 
        if item_categories is None:
            raise ValueError("item_categories are required for multiple choice imputation")
        for col in range(data.shape[1]):
            # Get the incorrect non-missing responses from the column
            incorrect_responses = torch.arange(0, item_categories[col])
            incorrect_responses = incorrect_responses[incorrect_responses != mc_correct[col]-1]

            # Find the indices of -1 values in the column
            missing_indices = (data[:, col] == -1).squeeze()

            # randomly sample from the incorrect responses and replace missing
            data[missing_indices, col] = incorrect_responses[torch.randint(0, incorrect_responses.size(0), (missing_indices.sum(),))].float()

        return data

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
                # TODO what if we encoded missing?
                # Create a tensor to hold probabilities for all choices
                probs = torch.full((num_categories,), (1 - guessing_prob) / (num_categories - 1))
                
                # Assign the guessing probability to the correct option
                probs[mc_correct[item_idx]-1] = guessing_prob

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
    Compute total score distribution conditional on z as by (Thissen et. al. 1995).

    Parameters
    ----------
    probabilities : torch.Tensor
        A 3D tensor containing item score probabilities for each z quadrature point.
        The first dimension represents the quadrature points, the second dimension represents the items and the third dimension represents the item categories.
    item_categories : list[list]
        A list of integers where each integer is the number of possible responses for the corresponding item.

    Returns
    -------
    torch.Tensor
        A 2D torch tensor of total score probabilities, rows being quadrature z points and columns being total scores. (rows sum to 1)
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
    modeled_item_responses: list[int],
    mc_correct: list[int],
    missing_modeled: bool,
):
    """
    Sum incorrect score probabilities for multiple choice items. Useful for approximating sum scores.

    Parameters
    ----------
    probabilities : list[torch.Tensor]
        A list of 2D tensors containing item score probabilities for each item.
        Rows are z quadrature points from the z density and columns correspond to item responses. (rows sum to 1)
    modeled_item_responses : list[int]
        A list of integers where each integer is the number of possible responses for the corresponding item.
    mc_correct : list[int]
        A list of integers where each integer is correct response for the corresponding item.
    missing_modeled : bool
        Whether probabilities come from a model that modeled missing responses.

    Returns
    -------
    torch.Tensor
        A 2D torch tensor correct/incorrect response probabilities
    """
    mc_correct = [corr + 1 for corr in mc_correct] if missing_modeled else mc_correct
    new_probs = torch.zeros(probabilities.shape[0], probabilities.shape[1], 2)
    for item in range(0, len(modeled_item_responses)):
        item_score_0_probs = torch.cat(
            (
                probabilities[:, item, :mc_correct[item] - 1],
                probabilities[:, item, mc_correct[item]:]
            ),
            dim=1,
        ).sum(dim=1)
        new_probs[:, item, :] = torch.cat(
            (
                item_score_0_probs.unsqueeze(1),
                probabilities[:, item, mc_correct[item] - 1].unsqueeze(1),
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


def output_to_item_entropy(output, item_categories: list[int]):
    """
    Calculate the entropy for each respondent for each item from autoencoder output logits.

    Parameters
    ----------
    output : torch.Tensor
        The outputted logits from the autoencoder. It should be a 2D tensor where the first dimension
        is the number of respondents and the second dimension is the number of items times the number of categories.
    item_categories : list of int
        A list of integers where each integer represents the number of categories for an item. The list length should
        be equal to the number of items.

    Returns
    -------
    torch.Tensor
        A 2D tensor with the entropy for each respondent for each item. The first dimension is the number of test
        takers and the second dimension is the number of items.

    Raises
    ------
    ValueError
        If the length of item_categories is not equal to the second dimension of output divided by the sum of item_categories.
    """
    if output.shape[1] != max(item_categories)*len(item_categories):
        raise ValueError(
            "Length of item_categories must be equal to the second dimension of output divided by the sum of item_categories."
        )

    reshaped_output = output.reshape(output.shape[0], len(item_categories), max(item_categories))
    probabilities = reshaped_output.softmax(dim=2)
    return entropy(probabilities)

class PytorchIRTDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor):
        super().__init__()
        self.data = data
        # set missing responses to 0 in the response mask (all non-missing are ones)
        self.mask = torch.zeros_like(data, dtype=torch.int)
        self.mask[data == -1] = 1

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.mask[index]

def is_jupyter():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
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
    print(formatted_string, end='', flush=True)