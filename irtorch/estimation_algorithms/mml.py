import logging
import copy
import torch
from torch import nn
from irtorch.models import BaseIRTModel
from irtorch.estimation_algorithms import BaseIRTAlgorithm
from torch.distributions import MultivariateNormal
from irtorch._internal_utils import dynamic_print, PytorchIRTDataset
from irtorch.utils import one_hot_encode_test_data, decode_one_hot_test_data

logger = logging.getLogger('irtorch')

class MMLIRT(BaseIRTAlgorithm, nn.Module):
    r"""
    Marginal Maximum Likelihood (MML) for fitting IRT models. 
    Uses a multivariate normal distribution for the latent variables and Gradient Descent to optimize the model parameters.
    This method is generally effecive for models with a small number of latent variables. More than 3 is not supported.
    Note that this method runs much faster on a GPU.

    Parameters
    ----------
    model : BaseIRTModel
        The model to fit. Needs to inherit irtorch.models.BaseIRTModel.
    one_hot_encoded : bool, optional
        Whether the model uses one-hot encoded data. (default is False)

    Notes
    -----
    Estimates the model parameters using the Marginal Maximum Likelihood (MML) method. 
    The marginal likelihood is calculated by integrating over an assumed latent variable distribution with density :math:`f(\mathbf{z})`.

    .. math::

        \log L(\phi) = \sum_{i=1}^N \log \left( \int P(\mathbf{X}_i = \mathbf{x}_i | \mathbf{z}, \phi)f(\mathbf{z})d\mathbf{z} \right)

    where

    - :math:`N` are the number of respondents,
    - :math:`\mathbf{X}_i` is the response vector of the :math:`i`-th respondent,
    - :math:`\mathbf{x}_i` is the observed response vector of the :math:`i`-th respondent,
    - :math:`\phi` are the model parameters,
    - :math:`\mathbf{z}` is the latent variable vector,

    Gaussian quadratures are used to approximate the integral.
    """
    def __init__(
        self,
        model: BaseIRTModel,
        one_hot_encoded: bool = False,
    ):
        super().__init__(model = model, one_hot_encoded=one_hot_encoded)
        self.imputation_method = "zero"
        self.covariance_matrix = torch.eye(self.model.latent_variables)
        self.training_z_scores = None
        self.training_history = {
            "train_loss": [],
        }

        # always set to eval mode by default
        self.model.eval()

    def forward(self, latent_variables):
        return self.model(latent_variables)

    def fit(
        self,
        train_data: torch.Tensor,
        max_epochs: int = 1000,
        quadrature_points: int = None,
        covariance_matrix: torch.Tensor = None,
        learning_rate: float = 0.2,
        learning_rate_update_patience: int = 4,
        learning_rate_updates_before_stopping: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        imputation_method: str = "zero",
    ):
        """
        Train the model.

        Parameters
        ----------
        train_data : torch.Tensor
            The training data. Item responses should be coded 0, 1, ... and missing responses coded as nan or -1.
        max_epochs : int, optional
            The maximum number of epochs to train for. (default is 1000)
        quadrature_points : int, optional
            The number of quadrature points to use for latent variable integration. Large datasets may lead to memory issues if quadratures points are too high. (default is 'None' and uses a function of the number of latent variables)
        covariance_matrix : torch.Tensor, optional
            The covariance matrix for the multivariate normal distribution for the latent variables. (default is None and uses uncorrelated variables)
        learning_rate : float, optional
            The initial learning rate for the optimizer. (default is 0.2)
        learning_rate_update_patience : int, optional
            The number of epochs to wait before reducing the learning rate. (default is 4)
        learning_rate_updates_before_stopping : int, optional
            The number of times the learning rate can be reduced before stopping training. (default is 2)
        device : str, optional
            The device to run the model on. (default is "cuda" if available else "cpu".)
        imputation_method : str, optional
            The method to use for imputing missing data. (default is "zero")
        """
        super().fit(train_data)
        self.imputation_method = imputation_method

        if self.one_hot_encoded:
            train_data = one_hot_encode_test_data(train_data, self.model.item_categories, encode_missing=self.model.model_missing)

        self.training_history = {
            "train_loss": [],
        }

        if quadrature_points is None:
            if self.model.latent_variables > 3:
                raise ValueError("MML is not implemented for models with more than 3 latent variables because of large integration grid.")
            quadrature_points = {
                1: 20,
                2: 8,
                3: 5
            }.get(self.model.latent_variables, 20)

        latent_grid = torch.linspace(-3, 3, quadrature_points).view(-1, 1)
        latent_grid = latent_grid.expand(-1, self.model.latent_variables).contiguous()
        if covariance_matrix is not None:
            if covariance_matrix.shape[0] != self.model.latent_variables or covariance_matrix.shape[1] != self.model.latent_variables:
                raise ValueError("Covariance matrix must have the same dimensions as the latent variables.")
            self.covariance_matrix = covariance_matrix

        self.optimizer = torch.optim.Adam(
            [{"params": self.parameters()}], lr=learning_rate, amsgrad=True
        )

        # Reduce learning rate when loss stops decreasing ("min")
        # we multiply the learning rate by the factor
        # patience: We need no improvement after x epochs for it to trigger
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.6, patience=learning_rate_update_patience
        )

        self.to(device)
        normal_dist = MultivariateNormal(
            loc=torch.zeros(self.model.latent_variables).to(device),
            covariance_matrix=self.covariance_matrix.to(device)
        )
        self._training_loop(
            train_data.to(device),
            max_epochs,
            latent_grid.to(device),
            normal_dist,
            scheduler,
            learning_rate_updates_before_stopping
        )
        self.to("cpu")
        self.eval()

        # store the latent z scores of the training data
        # used for more efficient computation when using other methods
        if not self.one_hot_encoded:
            train_data = self.fix_missing_values(train_data)

    def _training_loop(
        self,
        train_data: torch.Tensor,
        max_epochs: int,
        latent_grid: torch.Tensor,
        normal_dist: MultivariateNormal,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        learning_rate_updates_before_stopping: int,
    ):
        """
        The training loop for the model.

        Parameters
        ----------
        train_data : torch.Tensor
            The training data.
        max_epochs : int
            The maximum number of epochs to train for.
        latent_grid : torch.Tensor
            The grid of latent variables.
        normal_dist : torch.distributions.MultivariateNormal
            The multivariate normal distribution for the latent variables.
        scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
            The learning rate scheduler.
        learning_rate_updates_before_stopping : int, optional
            The number of times the learning rate can be reduced before stopping training.
        """
        lr_update_count = 0
        irt_dataset = PytorchIRTDataset(data=train_data)
        train_data = self._impute_missing(irt_dataset.data, irt_dataset.mask)
        train_data = decode_one_hot_test_data(train_data, self.model.modeled_item_responses) if self.one_hot_encoded else train_data
        if latent_grid.size(1) > 1:
            columns = [latent_grid[:, i] for i in range(latent_grid.size(1))]
            latent_combos = torch.cartesian_prod(*columns)
        else:
            latent_combos = latent_grid

        log_weights = normal_dist.log_prob(latent_combos)
        latent_combos_rep = latent_combos.repeat_interleave(train_data.size(0), dim=0)
        train_data_rep = train_data.repeat(latent_combos.size(0), 1)
        log_weights_rep = log_weights.repeat_interleave(train_data.size(0), dim=0)

        best_loss = float('inf')
        prev_lr = [group['lr'] for group in self.optimizer.param_groups]
        for epoch in range(max_epochs):
            train_loss = self._train_step(
                train_data_rep,
                latent_combos_rep,
                log_weights_rep,
                latent_combos.size(0)
            )

            current_loss = train_loss
            scheduler.step(train_loss)
            dynamic_print(f"Epoch: {epoch}. Average training batch loss function: {train_loss:.4f}")

            if current_loss < best_loss:
                best_loss = current_loss
                best_epoch = epoch
                best_model_state = { 'state_dict': copy.deepcopy(self.state_dict()),
                                    'optimizer': copy.deepcopy(self.optimizer.state_dict()) }
            

            current_lr = [group['lr'] for group in self.optimizer.param_groups]
            # Check if the learning rate has been updated
            if current_lr != prev_lr:
                lr_update_count += 1
                prev_lr = current_lr

            if lr_update_count >= learning_rate_updates_before_stopping:
                logger.info("Stopping training after %s learning rate updates.", learning_rate_updates_before_stopping)
                break

            logger.debug("Current learning rate: %s", self.optimizer.param_groups[0]['lr'])

        # Load the best model state
        if best_model_state is not None:
            self.load_state_dict(best_model_state['state_dict'])
            self.optimizer.load_state_dict(best_model_state['optimizer'])
            logger.info("Best model found at epoch %s with loss %.4f.", best_epoch, best_loss)

    def _train_step(
        self,
        train_data: torch.Tensor,
        latent_grid: torch.Tensor,
        log_weights: torch.Tensor,
        number_of_weights: int,
    ):
        """
        Training step for an epoch.

        Parameters
        ----------
        train_data : torch.Tensor
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
        self.train()

        self.optimizer.zero_grad()
        logits = self(latent_grid)
        ll = self.model.log_likelihood(train_data, logits, loss_reduction='none')
        ll = ll.view(-1, self.model.items).sum(dim=1) # sum over items
        
        log_sums = (log_weights + ll).view(number_of_weights, -1)
        constant = log_sums.max(dim=0)[0] # for logexpsum trick (one constant per respondent)
        exp_log_sums = (log_sums-constant).exp()
        loss = -(exp_log_sums.sum(dim=0).log() + constant).sum()

        # log_sums = (log_weights + ll).view(number_of_weights, -1)
        # constant = log_sums.max() # for logexpsum trick
        # exp_log_sums = (log_sums-constant).exp()
        # loss = -(exp_log_sums.sum(dim=0).log() + constant).sum()

        loss.backward()
        self.optimizer.step()
        
        self.training_history["train_loss"].append(loss.item())
        return loss.item()

    def _impute_missing(self, data, missing_mask):
        if torch.sum(missing_mask) > 0:
            if self.imputation_method == "zero":
                imputed_data = data
                imputed_data = imputed_data.masked_fill(missing_mask.bool(), 0)
            elif self.imputation_method == "prior":
                imputed_data = self._impute_missing_with_prior(data, missing_mask)
            elif self.imputation_method == "mean":
                raise NotImplementedError("Mean imputation not implemented")
            else:
                raise ValueError(
                    f"Imputation method {self.imputation_method} not implmented"
                )
            return imputed_data

        return data

    @torch.inference_mode()
    def _impute_missing_with_prior(self, batch, missing_mask):
        # get the decoder logits for the prior mean person
        prior_logits = self.model(
            torch.zeros(1, self.model.latent_variables).to(next(self.parameters()).device)
        )
        prior_mean_scores = self._mean_scores(prior_logits)
        batch[missing_mask.bool()] = prior_mean_scores.repeat(batch.shape[0], 1).to(
            next(self.parameters()).device
        )[missing_mask.bool()]

        return batch

    @torch.inference_mode()
    def _batch_fit_measures(self, batch: torch.Tensor):
        """
        Calculate the fit measures for a batch.

        Parameters
        ----------
        batch : torch.Tensor
            The batch of data.

        Returns
        -------
        tuple
            The loss, log likelihood, and accuracy for the batch.
        """
        output = self(batch)
        if self.one_hot_encoded:
            # for running with loss_function
            batch = decode_one_hot_test_data(batch, self.model.modeled_item_responses)
        # negative ce is log likelihood
        log_likelihood = self.model.log_likelihood(batch, output)
        loss = -log_likelihood / batch.shape[0]
        return loss, log_likelihood
