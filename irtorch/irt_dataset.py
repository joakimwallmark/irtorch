import torch
from irtorch._internal_utils import one_hot_encode_test_data
from irtorch._internal_utils import impute_missing_internal

class PytorchIRTDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for IRT data.

    Parameters
    ----------
    data : torch.Tensor
        A 2D tensor with the data. Rows are respondents and columns are items.
    one_hot_encoded : bool, optional
        Only relevant for autoencoder models. Whether the input data is one-hot encoded. (default is False)
    item_categories : list[int], optional
        For one_hot_encoding. A list of integers where each integer is the number of possible responses for the corresponding item. (default is None)
    imputation_method : str, optional
        The method to use for imputing missing values into the autoencoder input data.
        The data itself is not modified. 
        See :func:`irtorch.utils.impute_missing` for available methods.
        Note that only methods not relying on a fitted model can be used. (The default is None)
    **kwargs
        Additional keyword arguments passed to :func:`irtorch.utils.impute_missing`.
    """
    def __init__(self,
        data: torch.Tensor,
        one_hot_encoded: bool = False,
        item_categories: list[int] = None,
        imputation_method: str = None,
        **kwargs
    ):
        super().__init__()
        self.data = data.clone() # clone to avoid modifying the original data

        # set missing responses to 0 in the response mask (all non-missing are ones)
        self.mask = (data == -1) | data.isnan()

        if imputation_method is not None:
            data = impute_missing_internal(data = data, method=imputation_method, item_categories=item_categories, **kwargs)
        
        if one_hot_encoded:
            if item_categories is None:
                raise ValueError("item_categories must be supplied for one-hot encoded data.")
            self.input_data = one_hot_encode_test_data(data, item_categories)
        else:
            self.input_data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Get a sample from the dataset.

        Parameters
        ----------
        index : int
            The index of the sample to get.

        Returns
        -------
        torch.Tensor
            The data for the sample.
        torch.Tensor
            The mask for the sample.
        torch.Tensor
            Only relevant for autoencoder models. The autoencoder input data for the sample.
        """
        return self.data[index], self.mask[index], self.input_data[index]