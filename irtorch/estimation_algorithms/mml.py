import logging
import copy
import torch
from torch.distributions import MultivariateNormal
from irtorch.models import BaseIRTModel, MonotoneBSpline, NestedLogit, MonotoneBSpline
from irtorch.utils import gauss_hermite
from irtorch._internal_utils import dynamic_print
from irtorch.irt_dataset import PytorchIRTDataset
from irtorch.estimation_algorithms import BaseIRTAlgorithm

logger = logging.getLogger("irtorch")

class MML(BaseIRTAlgorithm):
    r"""
    Marginal Maximum Likelihood (MML) for fitting IRT models :cite:p:`Bock1981`. 
    Uses a multivariate normal distribution for the latent variables and Gradient Descent to optimize the model parameters.
    This method is generally effecive for models with a small number of latent variables. More than 3 is not supported.
    Note that this typically method runs much faster on a GPU.

    The marginal log-likelihood is calculated by integrating over an assumed normal distribution for the latent variables with density :math:`f(\mathbf{\theta})`.

    .. math::

        \log L(\phi) = \sum_{i=1}^N \log \left( \int P(\mathbf{X}_i = \mathbf{x}_i | \mathbf{\theta}, \phi)f(\mathbf{\theta})d\mathbf{\theta} \right)

    where

    - :math:`N` are the number of respondents,
    - :math:`\mathbf{X}_i` is the response vector of the :math:`i`-th respondent,
    - :math:`\mathbf{x}_i` is the observed response vector of the :math:`i`-th respondent,
    - :math:`\phi` are the model parameters,
    - :math:`\mathbf{\theta}` is the latent variable vector,

    The integral is approximated using Gauss-Hermite quadratures or a Quasi-Monte Carlo method. 
    :math:`\log L(\phi)` is then maximized using stochastic gradient descent. These steps are repeated until convergence.
    """
    def __init__(
        self,
    ):
        super().__init__()
        self.covariance_matrix = None
        self.optimizer = None
        self.training_history = {
            "train_loss": [],
        }

    def fit(
        self,
        model: BaseIRTModel,
        train_data: torch.Tensor,
        max_epochs: int = 1000,
        integration_method: str = "quasi_mc",
        quadrature_points: int = None,
        covariance_matrix: torch.Tensor = None,
        learning_rate: float = 0.20,
        learning_rate_update_patience: int = 7,
        learning_rate_updates_before_stopping: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Train the model.

        Parameters
        ----------
        model : BaseIRTModel
            The model to fit. Needs to inherit :class:`irtorch.models.BaseIRTModel`.
        train_data : torch.Tensor
            The training data. Item responses should be coded 0, 1, ... and missing responses coded as nan or -1.
        max_epochs : int, optional
            The maximum number of epochs to train for. (default is 1000)
        integration_method : str, optional
            The method to use for approximating integrals over the latent variables. Can be either "gauss_hermite" for Gauss-Hermite quadrature
            or "quasi_mc" for quasi-Monte Carlo. (default is "quasi_mc").
        quadrature_points : int, optional
            The number of quadrature points to use for latent variable integration. Note that large datasets may lead to memory issues if quadratures points are too high. (default is 'None' and uses a function of the number of latent variables)
        covariance_matrix : torch.Tensor, optional
            The covariance matrix for the multivariate normal distribution for the latent variables. (default is None and uses uncorrelated variables)
        learning_rate : float, optional
            The initial learning rate for the optimizer. (default is 0.25)
        learning_rate_update_patience : int, optional
            The number of epochs to wait before reducing the learning rate. (default is 7)
        learning_rate_updates_before_stopping : int, optional
            The number of times the learning rate can be reduced before stopping training. (default is 2)
        device : str, optional
            The device to run the model on. (default is "cuda" if available else "cpu".)
        """
        super().fit(model = model, train_data = train_data)

        self.training_history = {
            "train_loss": [],
        }

        if quadrature_points is None:
            if model.latent_variables > 3:
                raise ValueError("MML is not implemented for models with more than 3 latent variables because of large integration grid.")
            quadrature_points = {
                1: 15,
                2: 7,
                3: 5
            }.get(model.latent_variables, 20)

        if covariance_matrix is not None:
            if covariance_matrix.shape[0] != model.latent_variables or covariance_matrix.shape[1] != model.latent_variables:
                raise ValueError("Covariance matrix must have the same dimensions as the latent variables.")
            self.covariance_matrix = covariance_matrix
        else:
            self.covariance_matrix = torch.eye(model.latent_variables)

        self.optimizer = torch.optim.Adam(
            list(model.parameters()), lr=learning_rate, amsgrad=True
        )

        if integration_method == "gauss_hermite":
            points, weights = gauss_hermite(
                n=quadrature_points,
                mean=torch.zeros(model.latent_variables),
                covariance=self.covariance_matrix
            )
            log_weights = weights.log()
        else:
            points, log_weights = self._quasi_mc(n_points=quadrature_points, latent_variables=model.latent_variables)

        # Reduce learning rate when loss stops decreasing ("min")
        # we multiply the learning rate by the factor
        # patience: We need no improvement after x epochs for it to trigger
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.6,
            patience=learning_rate_update_patience,  # More epochs before reducing
            threshold=5e-5,  # Consider improvements of this or more as significant
            threshold_mode='rel'  # Make threshold relative to the loss
        )

        model.to(device)
        self._training_loop(
            model,
            train_data.to(device),
            max_epochs,
            points.to(device, dtype = torch.float32),
            log_weights.to(device, dtype = torch.float32),
            scheduler,
            learning_rate_updates_before_stopping
        )
        
        model.to("cpu")
        model.eval()

    def _training_loop(
        self,
        model: BaseIRTModel,
        train_data: torch.Tensor,
        max_epochs: int,
        points: torch.Tensor,
        log_weights: torch.Tensor,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        learning_rate_updates_before_stopping: int,
    ):
        """
        The training loop for the model.

        Parameters
        ----------
        model : BaseIRTModel
            The model to train.
        train_data : torch.Tensor
            The training data.
        max_epochs : int
            The maximum number of epochs to train for.
        points : torch.Tensor
            The latent variable points to evaluate the MML integral.
        log_weights : torch.distributions.MultivariateNormal
            The logarithm of the integral weights associated with the points.
        scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
            The learning rate scheduler.
        learning_rate_updates_before_stopping : int
            The number of times the learning rate can be reduced before stopping training.

        Returns
        -------
        None
        """
        lr_update_count = 0
        best_model_state = None
        latent_combos_rep = points.repeat_interleave(train_data.size(0), dim=0)
        train_data_rep = train_data.repeat(points.size(0), 1)
        log_weights_rep = log_weights.repeat_interleave(train_data.size(0), dim=0)
        irt_dataset_rep = PytorchIRTDataset(data=train_data_rep)
        # precompute basis functions for spline models
        if isinstance(model, MonotoneBSpline) or isinstance(model, MonotoneBSpline) or (isinstance(model, NestedLogit) and model.incorrect_response_model == "bspline"):
            model.precompute_basis(latent_combos_rep)

        best_epoch = 0
        best_loss = float("inf")
        prev_lr = [group["lr"] for group in self.optimizer.param_groups]
        for epoch in range(max_epochs):
            train_loss = self._train_step(
                model,
                irt_dataset_rep,
                latent_combos_rep,
                log_weights_rep,
                points.size(0)
            )

            current_loss = train_loss
            scheduler.step(train_loss)
            dynamic_print(f"Epoch: {epoch}. Loss: {train_loss:.4f}")

            if current_loss < best_loss:
                best_loss = current_loss
                best_epoch = epoch
                best_model_state = { "state_dict": copy.deepcopy(model.state_dict()),
                                    "optimizer": copy.deepcopy(self.optimizer.state_dict()) }
            

            current_lr = [group["lr"] for group in self.optimizer.param_groups]
            # Check if the learning rate has been updated
            if current_lr != prev_lr:
                lr_update_count += 1
                prev_lr = current_lr

            if lr_update_count >= learning_rate_updates_before_stopping:
                logger.info("Stopping training after %s learning rate updates.", learning_rate_updates_before_stopping)
                break

            logger.debug("Current learning rate: %s", self.optimizer.param_groups[0]["lr"])

        # remove basis functions for spline models after fitting
        if isinstance(model, MonotoneBSpline) or isinstance(model, MonotoneBSpline) or (isinstance(model, NestedLogit) and model.incorrect_response_model == "bspline"):
            model.basis = None

        # Load the best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state["state_dict"])
            self.optimizer.load_state_dict(best_model_state["optimizer"])
            logger.info("Best model found at iteration %s with loss %.4f.", best_epoch, best_loss)

    def _train_step(
        self,
        model: BaseIRTModel,
        train_data: PytorchIRTDataset,
        latent_grid: torch.Tensor,
        log_weights: torch.Tensor,
        number_of_weights: int,
    ):
        """
        Training step for an epoch.

        Parameters
        ----------
        model : BaseIRTModel
            The model to train.
        train_data : PytorchIRTDataset
            The training data.
        latent_grid : torch.Tensor
            The grid of latent variables.
        log_weights : torch.Tensor
            The log weights for the latent variables.
        number_of_weights : int
            The number of quadrature points. The number of different weights before expanding.

        Returns
        -------
        float
            The loss after the training step.
        """
        model.train()

        self.optimizer.zero_grad()
        logits = model(latent_grid)
        ll = model.log_likelihood(train_data.data, logits, missing_mask=train_data.mask, loss_reduction="none")
        ll = ll.view(-1, model.items).nansum(dim=1) # sum over items
        
        log_sums = (log_weights + ll).view(number_of_weights, -1)
        constant = log_sums.max(dim=0)[0] # for logexpsum trick (one constant per respondent)
        exp_log_sums = (log_sums-constant).exp()
        loss = -(exp_log_sums.sum(dim=0).log() + constant).sum()

        loss.backward()
        self.optimizer.step()
        
        self.training_history["train_loss"].append(loss.item())
        return loss.item()

    def _quasi_mc(self, n_points: int, latent_variables: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the points and weights for Quasi-Monte Carlo integral approximation.

        Parameters
        ----------
        n_points : int
            The number of points of evaluation.
        latent_variables : int
            The number of latent variables.

        Returns
        -------
        torch.Tensor
            The points of integration.
        torch.Tensor
            The logarithm of the weights associated with the points.
        """
        latent_grid = torch.linspace(-3, 3, n_points).view(-1, 1)
        latent_grid = latent_grid.expand(-1, latent_variables).contiguous()
        if latent_variables > 1:
            columns = [latent_grid[:, i] for i in range(latent_grid.size(1))]
            latent_combos = torch.cartesian_prod(*columns)
        else:
            latent_combos = latent_grid

        normal_dist = MultivariateNormal(
            loc=torch.zeros(latent_variables),
            covariance_matrix=self.covariance_matrix
        )
        weights = normal_dist.log_prob(latent_combos)
        return latent_combos, weights
