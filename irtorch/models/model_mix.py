import torch
import torch.nn as nn
import torch.nn.functional as F
from irtorch.models.base_irt_model import BaseIRTModel

class ModelMix(BaseIRTModel):
    r"""
    A mix of models tied together by a single set of latent variables. 
    This allows for a mix of different IRT models to be used when items have different types or some items don't fit a single model.
    Note that the items in the test data must be ordered in the same way as the models are ordered in the list.

    Parameters
    ----------
    models : list[BaseIRTModel]
        A list of IRT models to be mixed together.

    Examples
    --------
    Example of fitting a mix of Generalized Partial Credit (GPC) and Monotone Neural Network (MNN) models to the Swedish National Mathematics 2 dataset.
    The GPC model is used for the first 14 items and MNN for the last 14 items.
    
    >>> from irtorch.models import GeneralizedPartialCredit, MonotoneNN, ModelMix
    >>> from irtorch.estimation_algorithms import AE
    >>> from irtorch.load_dataset import swedish_national_mathematics_2
    >>> data_math = swedish_national_mathematics_2()
    >>> gpc = GeneralizedPartialCredit(data=data_math[:, :14])
    >>> mnn = MonotoneNN(data = data_math[:, 14:])
    >>> mix_model = ModelMix([gpc, mnn])
    >>> mix_model.fit(train_data=data_math, algorithm=AE())
    """
    def __init__(
        self,
        models: list[BaseIRTModel],
    ):
        item_categories = [item for model in models for item in model.item_categories]
        latent_variables = [model.latent_variables for model in models]
        if not all([latent_variables[0] == latent_variables[i] for i in range(1, len(latent_variables))]):
            raise ValueError("All models must have the same number of latent variables.")

        super().__init__(
            latent_variables=latent_variables[0],
            item_categories = item_categories
        )
        # register each model as a submodule
        self.models = nn.ModuleList(models)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight_param, mean=1., std=0.01)
        nn.init.zeros_(self.bias_param)
    
    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        theta : torch.Tensor
            2D tensor with latent variables. Rows are respondents and latent variables are columns. 

        Returns
        -------
        output : torch.Tensor
            2D tensor. Rows are respondents and columns are item category logits.
        """
        outputs = [model(theta) for model in self.models]
        concatenated_output = torch.cat(outputs, dim=1)
        return concatenated_output

    def probabilities_from_output(self, output: torch.Tensor) -> torch.Tensor:
        """
        Compute probabilities from the output tensor from the forward method.

        Parameters
        ----------
        output : torch.Tensor
            Output tensor from the forward pass.

        Returns
        -------
        torch.Tensor
            3D tensor with dimensions (respondents, items, item categories)
        """
        outputs = torch.split(
            output,
            [model.max_item_responses * model.items for model in self.models],
            dim=1
        )
        probabilities = [model.probabilities_from_output(output_part) for model, output_part in zip(self.models, outputs)]
        return self._pad_and_concat(probabilities, dim=1, pad_dim=2)

    def _pad_and_concat(self, tensors, dim, pad_dim):
        # Calculate the maximum size along the pad dimension
        max_size = max(tensor.size(pad_dim) for tensor in tensors)
        
        # Create a list to store padded tensors
        padded_tensors = []
        
        # Iterate over the tensors and pad them to the maximum size
        for tensor in tensors:
            pad_size = max_size - tensor.size(pad_dim)
            if pad_size > 0:
                pad = [0] * (2 * tensor.dim())
                pad[-(2 * pad_dim + 1)] = pad_size
                tensor = F.pad(tensor, pad)
            padded_tensors.append(tensor)
        
        # Concatenate the padded tensors along the specified dimension
        return torch.cat(padded_tensors, dim=dim)

    def item_probabilities(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Returns the probabilities for each possible response for all items.

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor. Rows are respondents and latent variables are columns.

        Returns
        -------
        torch.Tensor
            3D tensor with dimensions (respondents, items, item categories)
        """
        output = self(theta)
        return self.probabilities_from_output(output)

    def log_likelihood(
        self,
        data: torch.Tensor,
        output: torch.Tensor,
        missing_mask: torch.Tensor = None,
        loss_reduction: str = "sum",
    ) -> torch.Tensor:
        """
        Compute the log likelihood of the data given the model. This is equivalent to the negative cross entropy loss.

        Parameters
        ----------
        data: torch.Tensor
            A 2D tensor with test data. Columns are items and rows are respondents
        output: torch.Tensor
            A 2D tensor with output. Columns are item response categories and rows are respondents
        missing_mask: torch.Tensor, optional
            A 2D tensor with missing data mask. (default is None)
        loss_reduction: str, optional 
            The reduction argument for torch.nn.functional.cross_entropy(). (default is 'sum')
        
        Returns
        -------
        torch.Tensor
            The log likelihood.
        """
        # run model.log_likelihood for each model
        outputs = torch.split(
            output,
            [model.max_item_responses * model.items for model in self.models],
            dim=1
        )
        data_splits = torch.split(
            data,
            [model.items for model in self.models],
            dim=1
        )
        if missing_mask is None:
            mask_splits = [None] * len(self.models)
        else:
            mask_splits = torch.split(
                missing_mask,
                [model.items for model in self.models],
                dim=1
            )
            mask_splits = [mask.contiguous() for mask in mask_splits]
        log_likelihoods = [
            model.log_likelihood(data_split, output, mask_split, loss_reduction)
            for model, data_split, mask_split, output in zip(self.models, data_splits, mask_splits, outputs)
        ]

        if loss_reduction == "mean":
            raise NotImplementedError("loss_reduction='mean' is not implemented for ModelMix.")
        elif loss_reduction == "sum":
            return torch.sum(torch.stack(log_likelihoods), dim=0)
        elif loss_reduction == "none":
            return torch.cat(log_likelihoods, dim=0)
