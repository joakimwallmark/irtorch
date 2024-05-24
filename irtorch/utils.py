import torch

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

def impute_missing(data: torch.tensor, mc_correct: list[int] = None, item_categories: list[int] = None):
    """
    Impute missing values in the data. 
    For multiple choice data for which missing is not modeled, imputes randomly from incorrect responses.
    For non-multiple choice data, imputes missing values as 0.

    Parameters
    -----------
    data : torch.Tensor
        A 2D tensor where each row represents one respondent and each column represents an item.
        The values should be the scores/possible responses on the items, starting from 0.
        Missing item responses need to be coded as -1 or "nan".
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
            item_categories = (torch.where(~data.isnan(), data, torch.tensor(float('-inf'))).max(dim=0).values + 1).int().tolist()
        for col in range(data.shape[1]):
            # Get the incorrect non-missing responses from the column
            incorrect_responses = torch.arange(0, item_categories[col])
            incorrect_responses = incorrect_responses[incorrect_responses != mc_correct[col]-1]

            # Find the indices of -1 values in the column
            missing_indices = (data[:, col] == -1).squeeze()

            # randomly sample from the incorrect responses and replace missing
            data[missing_indices, col] = incorrect_responses[torch.randint(0, incorrect_responses.size(0), (missing_indices.sum(),))].float()

        return data

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
        # Shuffle data
        indices = torch.randperm(data.size(0))
        data_shuffled = data[indices]
    else:
        data_shuffled = data

    # Calculate the number of training samples
    train_size = int(data.size(0) * train_ratio)

    # Split the data
    train_data = data_shuffled[:train_size]
    test_data = data_shuffled[train_size:]

    return train_data, test_data

def one_hot_encode_test_data(
    data: torch.Tensor, item_categories: list, encode_missing: bool
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
    encode_missing: bool
        Encode missing values in a separate category. If False, they are coded as 0 for all items.

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
    if encode_missing:
        data = data + 1
        item_categories = [item_cat + 1 for item_cat in item_categories]
    else:
        data[data == -1] = 0

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

def decode_one_hot_test_data(one_hot_data: torch.Tensor, item_categories: list):
    """
    Decodes one-hot encoded test data back to the original scores.

    Parameters
    ----------
    one_hot_data : torch.Tensor
        A 2D tensor where each group of columns corresponds to the one-hot encoded scores for one of the items.
        The number of columns in each group is equal to the number of possible responses for that item.
    item_categories : list
        A list of integers where each integer is the number of possible responses for the corresponding item.

    Returns
    -------
    torch.Tensor
        A 2D tensor where each row represents one respondent and each column represents an item.
        The values are the scores achieved by the respondents on the items.
    """
    if one_hot_data.dim() != 2:
        raise ValueError("Input one_hot_data must be a 2D tensor.")
    if sum(item_categories) != one_hot_data.shape[1]:
        raise ValueError(
            "The total number of categories must match the number of columns in the one_hot_data tensor."
        )

    # Preallocate a tensor for the scores
    scores = []
    start = 0
    for _, item_cat in enumerate(item_categories):
        # Extract the one-hot encoded scores for this item
        # Decode the one-hot encoded scores back to the original scores
        scores.append(torch.argmax(one_hot_data[:, start : start + item_cat], dim=1))
        start += item_cat

    return torch.stack(scores, dim=1).float()
