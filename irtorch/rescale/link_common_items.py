import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from irtorch.rescale import Scale
from irtorch.models import BaseIRTModel
from irtorch.torch_modules import RationalQuadraticSpline
from irtorch.torch_modules import SoftplusLinear
from irtorch.activation_functions import BoundedELU

logger = logging.getLogger("irtorch")

class LinkCommonItems(Scale):
    r"""
    Link theta scales from two different IRT models to the same scale using common (anchor) items.
    Either rational quadratic splines :cite:p:`Durkan2019` or monotonic neural networks :cite:p:`Runje2023` can be used to link the scales.
    Currently only supports unidimensional models.
    
    Parameters
    ----------
    model_from : BaseIRTModel
        The IRT model which scale to transform.
    model_to : BaseIRTModel
        The scale of model_from will be linked to the scale of model_to.
    model_from_common_item_indices : list[int]
        The indices of the items in model_from that are also in model_to (first item is index 0).
    model_to_common_item_indices : list[int]
        The indices of the items in model_to that are also in model_from (first item is index 0).
    method : str, optional
        The method to use for linking the scales. Either "spline" or "neuralnet". Default is "spline".
        Note that the splines uses a fixed range of -5.5 to 5.5 for input values and a learned output range with initial
        values of -5.5 to 5.5. If latent scores are outside this range are common for your models, you may need to adjust the bounds.
        See :class:`irtorch.torch_modules.RationalQuadraticSpline` for more information.
    inverted : bool, optional
        Set to true if the theta scale of one model is inverted. Default is False.
    **kwargs
        Additional keyword arguments for :class:`irtorch.torch_modules.RationalQuadraticSpline` constructor when method is "spline".
        When method is "neuralnet", the number of neurons in the hidden layer can be set with the neurons argument.
        Note that the number of neurons must be divisible by 3. Default is 9.
        By default, the spline is set to have 50 bins and the input bounds are set to -5.5 and 5.5 and output bounds to -3.0 and 3.0.

    Notes
    -----
    We have two models fitted using data from two different populations, P and Q. We also have some items in common between the models for the purpose of linking. Let :math:`\theta_P` and :math:`\theta_Q` be points from the latent trait scales from the models fitted to P and Q respectively. Our goal is to find a linking function :math:`g\left(\theta_P\right)` which takes a :math:`\theta_P` and outputs the equivalent :math:`\theta_Q`. This is done by finding the :math:`g\left(\theta_P\right)` that minimizes the KL divergence between the transformed and linked item curves.

    .. math::

        \int \sum_{j \in \text { common }} \sum_x D_{K L}\left[P_P\left(X_j=x|\theta_P\right) \Vert P_Q \left(X_j=x| \theta_Q = g\left(\theta_P \right) \right)\right] f(\theta_P) d \theta_P

    - :math:`P_P\left(X_j=x|\theta_P\right)` is the probability for a score :math:`x` on item :math:`j` from the model fitted to population P given :math:`\theta_P`.
    - :math:`P_Q \left(X_j=x| \theta_Q = g\left(\theta_P \right)\right)` is the probability for a score :math:`x` on item :math:`j` from the model fitted to population Q given :math:`\theta_Q = g\left(\theta_P \right)`.
    - :math:`f(\theta_P)` is the density of the latent trait distribution in population P.
    - The sums are over the common items and their possible responses.
    
    Examples
    --------
    >>> import irtorch
    >>> from irtorch.rescale import LinkCommonItems
    >>> from irtorch.models import ThreeParameterLogistic
    >>> from irtorch.estimation_algorithms import JML, MML
    >>> data = irtorch.load_dataset.swedish_sat_binary()[:, :80]
    >>> # As an illustration, we split the dataset into two parts and use 20 common items.
    >>> # In practice, we would of course use different datasets for each model.
    >>> data1 = data[:2500, :50]
    >>> data2 = data[2500:, 30:]
    >>> model1 = ThreeParameterLogistic(items=50)
    >>> model2 = ThreeParameterLogistic(items=50)
    >>> model1.fit(train_data=data1, algorithm=MML())
    >>> model2.fit(train_data=data2, algorithm=JML())
    >>> # Link the scale of model 2 to the model 1 scale using common items.
    >>> link = LinkCommonItems(model2, model1, list(range(20)), list(range(30, 50)))
    >>> link.fit(theta_from = model2.latent_scores(data2), learning_rate=0.01, max_epochs=1000)
    >>> model2.add_scale_transformation(link)
    >>> # Plot the transformation
    >>> model2.plot.scale_transformations(input_theta_range=(-5, 5)).show()
    """
    def __init__(self,
        model_from: BaseIRTModel,
        model_to: BaseIRTModel,
        model_from_common_item_indices: list[int],
        model_to_common_item_indices: list[int],
        method: str = "spline",
        inverted: bool = False,
        **kwargs
    ):
        if model_from.latent_variables != model_to.latent_variables:
            raise ValueError("The models must have the same number of latent variables.")
        if model_from.latent_variables != 1:
            raise ValueError("Linking is currently only supported for unidimensional models.")
        if model_from_common_item_indices is None or model_to_common_item_indices is None:
            raise ValueError("Common item indices must be provided.")
        if len(model_from_common_item_indices) != len(model_to_common_item_indices):
            raise ValueError("The number of common items must be the same in both models.")
        if len(model_from_common_item_indices) < 1:
            raise ValueError("At least one common item must be provided.")
        if method not in ["spline", "neuralnet"]:
            raise ValueError("Method must be either 'spline' or 'neuralnet'.")

        self._method = method
        self._model_from = model_from
        self._model_to = model_to
        self._model_from_common_item_indices = model_from_common_item_indices
        self._model_to_common_item_indices = model_to_common_item_indices
        self._common_item_categories = [
            model_from.item_categories[i] for i in model_from_common_item_indices
        ]
        self._transformation_multiplier = torch.ones(1) * -1 if inverted else torch.ones(1)

        if method == "spline":
            Scale.__init__(self, invertible=False)
            spline_params = {
                'lower_input_bound': -5.5,
                'upper_input_bound': 5.5,
                'lower_output_bound': -5.5,
                'upper_output_bound': 5.5,
                'num_bins': 10,
                'free_endpoints': True
            }
            spline_params.update(kwargs)
            self._transformation = RationalQuadraticSpline(1, **spline_params)
        elif method == "neuralnet":
            Scale.__init__(self, invertible=False)
            neurons = kwargs.get("neurons", 9)
            self._transformation = SoftplusLinear(1, neurons)
        else:
            raise NotImplementedError("Transformation method not implemented.")

    def fit(
        self,
        theta_from: torch.Tensor,
        batch_size: int = None,
        learning_rate: float = 0.01,
        learning_rate_updates_before_stopping: int = 1,
        evaluation_interval_size: int = 50,
        max_epochs: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Fits the normalizing flow to the data. Typically used from within an IRT model instance. Use batch_size if the data is too large to fit in memory.
        
        Parameters
        ----------
        theta_from : torch.Tensor
            A 2D tensor containing latent variable theta scores from the model which theta scale we are transforming (model_from).
            Usually the training data and respresents the population. Each column represents one latent variable.
        batch_size : int, optional
            The batch size for the data loader. Default is None and uses no batches.
        learning_rate : float, optional
            The learning rate for the optimizer. Default is 0.1.
        learning_rate_updates_before_stopping, optional
            The number of learning rate updates before stopping the training. Default is 1.
        evaluation_interval_size: int, optional
            The number of iterations between each model evaluation during training. (default is 50)
        max_epochs : int, optional
            The maximum number of epochs to train the flow. Default is 1000.
        device : str, optional
            The device to use for the computation. Default is "cuda" if available, otherwise "cpu".
        """
        if batch_size is None:
            batch_size = theta_from.shape[0]

        optimizer = torch.optim.Adam(self._transformation.parameters(), lr=learning_rate, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.6, patience=1
        )
        self._transformation.to(device)
        self._model_from.to(device)
        self._model_to.to(device)
        self._transformation_multiplier = self._transformation_multiplier.to(device)
        loader = DataLoader(TensorDataset(theta_from.to(device)), batch_size=batch_size, shuffle=True)

        lr_update_count = 0
        total_iterations = 0
        prev_losses = []
        for epoch in range(max_epochs):
            epoch_losses = []
            for batch in loader:
                optimizer.zero_grad()
                linked_thetas = self.transform(batch[0])
                loss = self._loss_function(batch[0], linked_thetas)
                if torch.isnan(loss):
                    logger.warning("Loss is NaN. Stopping training. Try increasing batch size or lowering the learning rate.")
                    break

                loss.backward()
                optimizer.step()
                
                total_iterations += 1
                prev_losses.append(loss.item())
                epoch_losses.append(loss.item())
                if len(prev_losses) > evaluation_interval_size:
                    prev_losses.pop(0)

                # Update the learning rate scheduler every self.evaluation_interval_size iterations if no improvement
                if (total_iterations) % evaluation_interval_size == 0 and total_iterations != 1:
                    mean_loss = np.mean(prev_losses)
                    scheduler.step(mean_loss)
                    logger.info("Iteration %s: Mean loss: %s", total_iterations, mean_loss.item())

                # Check if the learning rate has been updated
                if learning_rate != optimizer.param_groups[0]['lr']:
                    learning_rate = optimizer.param_groups[0]['lr']
                    lr_update_count += 1

                if lr_update_count >= learning_rate_updates_before_stopping:
                    logger.info("Stopping training after %s learning rate updates.", learning_rate_updates_before_stopping)
                    break
            
            if torch.isnan(loss):
                break
            if lr_update_count >= learning_rate_updates_before_stopping:
                break

            logger.info("Epoch %s: Loss: %s", epoch, np.mean(epoch_losses))

        self._transformation.to("cpu")
        self._transformation.eval()
        self._model_from.to("cpu")
        self._model_to.to("cpu")
        self._transformation_multiplier = self._transformation_multiplier.to("cpu")

    def transform(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Transforms the input theta scores into the new scale.

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing latent variable theta scores. Each column represents one latent variable.
        """
        if self._method == "neuralnet":
            transformed = self._split_activation(self._transformation(theta)).sum(dim=1).view(-1, 1)
        elif self._method == "spline":
            transformed, _ = self._transformation(theta)
        else:
            raise NotImplementedError("Transformation method not implemented.")
        return transformed * self._transformation_multiplier

    def inverse(self, transformed_theta: torch.Tensor) -> torch.Tensor:
        """
        Puts the scores back to the original theta scale.

        Parameters
        ----------
        transformed_theta : torch.Tensor
            A 2D tensor containing transformed theta scores. Each column represents one latent variable.

        Returns
        -------
        torch.Tensor
            A 2D tensor containing theta scores on the the original scale.
        """
        if self._method == "spline":
            theta, _ = self._transformation(transformed_theta, inverse=True)
            return theta * self._transformation_multiplier
        else:
            raise NotImplementedError("Inverse transformation is not yet implemented for monotonic neural networks.")

    def jacobian(
        self,
        theta: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Computes the Jacobian of scale scores for each :math:`j` with respect to the input theta scores.

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing latent variable theta scores. Each column represents one latent variable.

        Returns
        -------
        torch.Tensor
            A tensor with the Jacobian for each input row. Dimensions are (theta rows, latent variables, latent variables) where the last two are the jacobians.
        """
        theta_scores = theta.clone()
        theta_scores.detach_().requires_grad_(True)
        if self._method == "neuralnet":
            transformed_thetas = self._transformation(theta_scores)
        else:
            transformed_thetas, _ = self._transformation(theta_scores, inverse=False)
        transformed_thetas = transformed_thetas * self._transformation_multiplier
        transformed_thetas.sum().backward()
        jacobians = torch.diag_embed(theta_scores.grad) # Since each transformation is only dependent on one theta score
        return jacobians
    
    def _split_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs various activation functions on every third item in the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape as the input tensor.
        """
        x1 = F.elu(x[:, ::3])
        x2 = -F.elu(-x[:, 1::3])
        x3 = BoundedELU.apply(x[:, 2::3], 1.0)
        y = torch.stack((x1, x2, x3), dim=2).view(x.shape)
        return y
    
    def _loss_function(self, latent_variables, linked_latent_variables) -> torch.Tensor:
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        target_probs = self._model_from.item_probabilities(latent_variables)[:, self._model_from_common_item_indices]
        linked_probs = self._model_to.item_probabilities(linked_latent_variables)[:, self._model_to_common_item_indices]
        target_probs = target_probs.view(-1, target_probs.shape[2])
        linked_probs = linked_probs.view(-1, linked_probs.shape[2])
        # linked_probs need log scale for KL div
        # Adding a tiny constant to avoid issues with log(0).
        loss = kl_loss(torch.log(linked_probs + 1e-10), target_probs)
        return loss
