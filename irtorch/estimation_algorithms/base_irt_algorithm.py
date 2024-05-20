import logging
from abc import ABC, abstractmethod
import torch
from irtorch.models import BaseIRTModel

logger = logging.getLogger("irtorch")

class BaseIRTAlgorithm(ABC):
    """
    Abstract base class for IRT algorithms. All IRT algorithms should inherit from this class.

    Parameters
    ----------
    model : BaseIRTModel, optional
        The model to train. Needs to inherit irtorch.models.BaseIRTModel.
    one_hot_encoded : bool, optional
        Whether the algorithm uses one-hot encoded data. (default is False)
    """
    def __init__(
        self,
        model: BaseIRTModel,
        one_hot_encoded: bool = False,
    ):
        super().__init__()
        self.model = model
        self.imputation_method = "zero"
        self.train_data = None

        if model.mc_correct is not None:
            self.one_hot_encoded = True
        else:
            self.one_hot_encoded = one_hot_encoded

    @abstractmethod
    def fit(self, train_data: torch.Tensor, **kwargs):
        """
        Fit the model to the data.

        Parameters
        ----------
        train_data : torch.Tensor
            The training data.
        """
        # Store all training data within the model instance
        if self.train_data is not None:
            self.train_data = torch.cat((self.train_data, train_data), dim=0).contiguous()
        else:
            self.train_data = train_data.contiguous()

    @abstractmethod
    def _impute_missing_with_prior(self, batch, missing_mask):
        """
        Impute missing values with the prior.

        Parameters
        ----------
        batch : torch.Tensor
            The batch of data.
        missing_mask : torch.Tensor
            The mask of missing values.

        Returns
        -------
        torch.Tensor
            The imputed data.
        """
        raise NotImplementedError(
            "Prior imputation not implemented for this algorithm."
        )

    def fix_missing_values(self, data: torch.Tensor):
        """
        Deal with missing values so that the data can be used for fitting.

        Parameters
        ----------
        data : torch.Tensor
            The data to fix.

        Returns
        -------
        torch.Tensor
            The data with missing values imputed.
        """
        if data.isnan().any():
            data[data.isnan()] = -1

        if self.model.model_missing:
            data = data + 1 # handled in z_scores for nn
        else:
            if self.imputation_method == "zero":
                data[data == -1] = 0
            else:
                # self._impute_missing(data, data.isnan())
                # see also helper_function impute_missing
                raise NotImplementedError("Imputation methods not implemented yet")

        return data
