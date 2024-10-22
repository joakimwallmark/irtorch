import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Distribution, Normal
from irtorch.rescale import Scale
from irtorch.torch_modules import NeuralSplineFlow
from irtorch.torch_modules import RationalQuadraticSpline

logger = logging.getLogger("irtorch")

class Flow(Scale):
    """
    Normalizing flow transformation of IRT theta scales using rational quadratic splines as per :cite:t:`Durkan2019`.
    Supports gradient computation and the transformation is invertible.
    
    Parameters
    ----------
    latent_variables : int
        The number of latent variables.
    
    Examples
    --------
    >>> import irtorch
    >>> from irtorch.models import GradedResponse
    >>> from irtorch.estimation_algorithms import AE
    >>> from irtorch.rescale import Flow
    >>> data = irtorch.load_dataset.swedish_national_mathematics_1()
    >>> model = GradedResponse(data)
    >>> model.fit(train_data=data, algorithm=AE())
    >>> thetas = model.latent_scores(data)
    >>> # Initalize and fit the flow scale transformation. Supply it to the model.
    >>> flow = Flow(1)
    >>> flow.fit(thetas)
    >>> model.add_scale_tranformation(flow)
    >>> # Estimate thetas on the transformed scale
    >>> rescaled_thetas = model.latent_scores(data)
    >>> # Or alternatively by directly converting the old ones
    >>> rescaled_thetas = model.transform_theta(thetas)
    >>> # Plot the differences
    >>> model.plot.plot_latent_score_distribution(thetas).show()
    >>> model.plot.plot_latent_score_distribution(rescaled_thetas).show()
    >>> # Put the thetas back to the original scale
    >>> original_thetas = model.inverse_transform_theta(rescaled_thetas)
    >>> # Plot an item on the flow transformed scale
    >>> model.plot.plot_item_probabilities(1).show()
    """
    def __init__(self, latent_variables: int):
        super().__init__(invertible=True)
        self.latent_variables = latent_variables
        self.theta_means = torch.zeros(latent_variables)
        self.theta_stds = torch.ones(latent_variables)
        self.flow = None

    def fit(
        self,
        theta:torch.Tensor,
        transformation: RationalQuadraticSpline = None,
        distribution: Distribution = None,
        batch_size: int = 512,
        learning_rate: float = 0.05,
        learning_rate_updates_before_stopping: int = 2,
        evaluation_interval_size: int = 30,
        max_epochs: int = 200,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        Fits the normalizing flow to the data. Typically used from within an IRT model instance.
        
        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing latent variable theta scores of the population. Usually the training data. Each column represents one latent variable.
        transformation : RationalQuadraticSpline, optional
            The transformation to apply to the data.
        distribution : Distribution, optional
            The distribution to apply to the latent variables. If None, a standard normal distribution is used.
        batch_size : int, optional
            The batch size for the data loader. Default is 256.
        learning_rate : float, optional
            The learning rate for the optimizer. Default is 0.01.
        learning_rate_updates_before_stopping: int = 5,
            The number of learning rate updates before stopping the training. Default is 2.
        evaluation_interval_size: int, optional
            The number of iterations between each model evaluation during training. (default is 60)
        max_epochs : int, optional
            The maximum number of epochs to train the flow. Default is 20.
        device : str, optional
            The device to use for the computation. Default is "cuda" if available, otherwise "cpu".
        **kwargs
            Additional keyword arguments for :class:`irtorch.torch_modules.RationalQuadraticSpline` constructor.
        """
        if transformation is None:
            transformation = RationalQuadraticSpline(self.latent_variables, **kwargs)

        if distribution is None:
            distribution = Normal(
                torch.zeros(1).to(device),
                torch.ones(1).to(device)
            )
        if len(distribution.batch_shape) > 0 and distribution.batch_shape[0] not in [1, self.latent_variables]:
            raise ValueError("The distribution batch shape should match the number of latent variables or be 0.")

        # standardize the data and store the mean and std for later use
        self.theta_means = theta.mean(dim=0)
        self.theta_stds = theta.std(dim=0)
        theta = (theta - self.theta_means) / self.theta_stds

        self.flow = NeuralSplineFlow(transformation, distribution).to(device)
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=learning_rate, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.6, patience=1
        )
        loader = DataLoader(TensorDataset(theta.to(device)), batch_size=batch_size, shuffle=True)

        lr_update_count = 0
        total_iterations = 0
        prev_losses = []
        for epoch in range(max_epochs):
            epoch_losses = []
            for batch in loader:
                loss = -self.flow.log_prob(batch[0]).mean()
                if torch.isnan(loss):
                    logger.warning("Loss is NaN. Stopping training. Try increasing batch size or lowering the learning rate.")
                    break

                optimizer.zero_grad()
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

        self.flow.to("cpu")
        self.flow.eval()

    def _flow_exists(self):
        if self.flow is None:
            raise ValueError("The flow has not been initialized. Please fit the flow first using the fit method.")

    def transform(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Transforms the input theta scores into the new scale.

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing latent variable theta scores. Each column represents one latent variable.
        """
        self._flow_exists()
        theta = (theta - self.theta_means) / self.theta_stds
        return self.flow(theta)

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
        self._flow_exists()
        theta = self.flow.inverse(transformed_theta)
        # destandardize
        return theta * self.theta_stds + self.theta_means

    def jacobian(
        self,
        theta: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Computes the gradients of scale scores for each :math:`j` with respect to the input theta scores.

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor containing latent variable theta scores. Each column represents one latent variable.

        Returns
        -------
        torch.Tensor
            A torch tensor with the gradients for each theta score. Dimensions are (theta rows, latent variables, latent variables) where the last two are the jacobians.
        """
        self._flow_exists()
        theta_scores = theta.clone()
        theta_scores.requires_grad_(True)
        standardized_theta_scores = (theta_scores - self.theta_means) / self.theta_stds
        transformed_thetas = self.flow(standardized_theta_scores)
        transformed_thetas.sum().backward()
        jacobians = torch.diag_embed(theta_scores.grad) # Since each transformation is only dependent on one theta score
        return jacobians
