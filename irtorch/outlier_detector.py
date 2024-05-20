import logging
import torch

logger = logging.getLogger("irtorch")

class OutlierDetector:
    """
    A class for identifying and detecting outliers in a dataset.
    """

    def __init__(self, factor: int = 1.5):
        """
        Initializes an instance of the OutlierDetector class.

        Parameters
        ------------
        factor : int, optional
            The factor used to determine the outliers. Default is 1.5.
        """
        self.factor = factor

    def __setattr__(self, name: str, value: any) -> None:
        """
        Prevents the factor from being set to a value less than 1.

        Parameters
        ------------
        name : str
            The name of the attribute being set.
        value : any
            The value to set the attribute to.
        """
        if name == "factor" and value < 1:
            raise ValueError("Factor must be greater than or equal to 1.")
        super().__setattr__(name, value)

    def identify_outliers(self, data: torch.Tensor, upper: bool = True, lower: bool = True):
        """
        Identifies outliers in the input data using the interquartile range method.

        Parameters
        ------------
        data : torch.Tensor
            A 2D tensor. Columns are items and rows are observations.
        upper : bool, optional
            Whether to identify upper outliers. Default is True.
        lower : bool, optional
            Whether to identify lower outliers. Default is True.

        Returns
        ----------
        torch.Tensor
            A 2D boolean tensor indicating the outlier status for each element in 'data'.
        """
        # Validate inputs
        if len(data.shape) != 2:
            raise ValueError("Input data must be a 2D tensor")
        if not upper and not lower:
            raise ValueError("At least one of upper and lower must be True")

        # Calculate quartiles and IQR for all columns
        q1, q3 = torch.quantile(data, torch.tensor([0.25, 0.75]), dim=0, keepdim=True)
        iqr = q3 - q1

        # Calculate lower and upper bounds
        lower_bound = q1 - self.factor * iqr
        upper_bound = q3 + self.factor * iqr

        # Identify outliers
        lower_outliers = data < lower_bound if lower else torch.full_like(data, False, dtype=torch.bool)
        upper_outliers = data > upper_bound if upper else torch.full_like(data, False, dtype=torch.bool)

        # Combine the conditions
        outliers = lower_outliers | upper_outliers

        return outliers

    def is_outlier(self, new_observations: torch.Tensor, data: torch.Tensor, upper: bool = True, lower: bool = True):
        """
        Determines if new observations are outliers in any of the columns based on the existing dataset.

        Parameters
        -------------
        new_observations : torch.Tensor
            A 2D tensor representing new observations to be tested. Its number of columns should match the number of columns in 'data'.
        data : torch.Tensor
            A 2D tensor representing the existing dataset. Columns are items, and rows are observations.
        upper : bool, optional
            Whether to identify upper outliers. Default is True.
        lower : bool, optional
            Whether to identify lower outliers. Default is True.

        Returns
        ------------
        torch.Tensor
            A 2D boolean tensor where each row corresponds to an observation and each column corresponds to a latent variable. 
            The value is True if the observation is an outlier for that latent variable.
        """
        # Validate inputs
        if len(data.shape) != 2 or len(new_observations.shape) != 2:
            raise ValueError("'data' and 'new_observations' must be 2D tensors.")
        if data.shape[1] != new_observations.shape[1]:
            raise ValueError("'new_observations' number of columns must match the number of columns in 'data'.")
        if not upper and not lower:
            raise ValueError("At least one of upper and lower must be True")

        # Calculate quartiles and IQR for all columns
        q1, q3 = torch.quantile(data, torch.tensor([0.25, 0.75]), dim=0, keepdim=True)
        iqr = q3 - q1

        # Calculate lower and upper bounds
        lower_bound = q1 - self.factor * iqr
        upper_bound = q3 + self.factor * iqr

        # Check if the new observations are outliers
        is_lower_outlier = (new_observations < lower_bound) if lower else torch.full_like(new_observations, False, dtype=torch.bool)
        is_upper_outlier = (new_observations > upper_bound) if upper else torch.full_like(new_observations, False, dtype=torch.bool)

        # An observation is an outlier if it is either a lower or upper outlier
        outlier_status = is_lower_outlier | is_upper_outlier

        return outlier_status

    def smallest_largest_non_outlier(self, data: torch.tensor, smallest = True):
        """
        Identifies the smallest non-outlier value in each column of the input data.

        Parameters
        -------------
        data : torch.Tensor
            A 2D tensor. Columns are features, and rows are observations.
        factor : int, optional
            The factor used to determine the outliers using the interquartile range method. (default is 1.5)
        smallest : bool, optional
            Whether to identify the smallest non-outlier value (True) or the largest non-outlier value (False). (default is True)

        Returns
        ------------
        torch.Tensor
            A 1D tensor containing the smallest non-outlier values for each column.
        """

        # Ensure the input data is a 2D tensor
        if len(data.shape) != 2:
            raise ValueError("Input data must be a 2D tensor")

        # Calculate quartiles and IQR for all columns, resulting in 1D tensors for q1, q3, and iqr
        q1, q3 = torch.quantile(data, torch.tensor([0.25, 0.75]), dim=0, keepdim=True)
        iqr = q3 - q1

        # Calculate lower and upper bounds
        lower_bound = q1 - self.factor * iqr
        upper_bound = q3 + self.factor * iqr

        # Mask the original data with the non-outliers mask and replace outliers with "inf"/"-inf" (to ignore during min/max calculation)
        non_outliers_mask = (data >= lower_bound) & (data <= upper_bound)
        if smallest:
            masked_data = torch.where(non_outliers_mask, data, torch.full_like(data, float("inf")))
        else:
            masked_data = torch.where(non_outliers_mask, data, torch.full_like(data, float("-inf")))
        resulting_values, _ = torch.min(masked_data, dim=0) if smallest else torch.max(masked_data, dim=0)

        return resulting_values.unsqueeze(0)