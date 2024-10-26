import logging
import copy
import torch
import numpy as np
from irtorch.models import BaseIRTModel
from irtorch.estimation_algorithms import BaseIRTAlgorithm
from irtorch.estimation_algorithms.encoders import StandardEncoder
from irtorch._internal_utils import dynamic_print
from irtorch.irt_dataset import PytorchIRTDataset

logger = logging.getLogger("irtorch")

class AE(BaseIRTAlgorithm):
    """
    Autoencoder neural network for fitting IRT models.
    """
    def __init__(self):
        super().__init__()
        self.one_hot_encoded = True
        self.batch_normalization = False
        self.encoder = None
        self.imputation_method = None
        self.training_theta_scores = None
        self.data_loader = None
        self.optimizer = None
        self.item_categories = None
        self.mc_correct = None
        self.training_history = {
            "train_loss": [],
        }
        self.best_model_state = None
        self.batch_mean_losses = []
        self.best_avg_loss = float("Inf"), 0
        self.total_iterations = 0
        self.lr_update_count = 0
        self.evaluation_interval_size = 60

    def fit(
        self,
        model: BaseIRTModel,
        train_data: torch.Tensor,
        one_hot_encoded: bool = True,
        imputation_method: str = None,
        learning_rate: float = 0.02,
        learning_rate_updates_before_stopping: int = 3,
        evaluation_interval_size: int = 80,
        max_epochs: int = 10000,
        batch_size: int = None,
        batch_normalization_encoder: bool = False,
        nonlinear_encoder = torch.nn.ELU(),
        hidden_layers_encoder: list[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Train an IRT model using the autoencoder. If the algorithm fails to converge, try lowering the learning rate. Use batch_size if the data is too large to fit in memory.

        Parameters
        ----------
        model : BaseIRTModel
            The model to fit. Needs to inherit :class:`irtorch.models.BaseIRTModel`.
        train_data : torch.Tensor
            The training data. Item responses should be coded 0, 1, ... and missing responses coded as nan or -1.
        one_hot_encoded : bool, optional
            Whether or not to one-hot encode the train data as encoder input inside this fit method. (default is True)
        imputation_method : str, optional
            The method to use for imputing missing data for the encoder. For options see :func:`irtorch.utils.impute_missing`.
            Only methods not relying on a fitted model can be used. 
            Note that missing values are removed from the loss calculation even after imputation.
            If you do not want this, do the imputation to your dataset before fitting. (default is None and only works for one-hot encoded inputs)
        learning_rate : float, optional
            The initial learning rate for the optimizer. (default is 0.005)
        learning_rate_updates_before_stopping : int, optional
            The number of times the learning rate can be reduced before stopping training. (default is 2)
        evaluation_interval_size: int, optional
            The number of iterations between each model evaluation during training. (default is 80)
        max_epochs : int, optional
            The maximum number of epochs to train for. (default is 1000)
        batch_size : int, optional
            The batch size for training. (default is None and uses the full dataset)
        batch_normalization_encoder : bool, optional
            Whether to use batch normalization for the encoder. (default is True)
        nonlinear_encoder : torch.nn.Module, optional
            The non-linear function to use after each hidden layer in the encoder. (default is torch.nn.ELU())
        hidden_layers_encoder : list[int], optional
            List of hidden layers for the encoder. Each element is a layer with the number of neurons represented as integers. If not provided, uses one hidden layer with 2 * sum(item_categories) neurons.
        device : str, optional
            The device to run the model on. (default is "cuda" if available else "cpu".)
        """
        super().fit(model = model, train_data = train_data)
        # Re-initialize the training history
        self.training_history = {
            "train_loss": [],
        }
        self.best_model_state = None
        self.batch_mean_losses = []
        self.best_avg_loss = float("Inf"), 0
        self.total_iterations = 0
        self.lr_update_count = 0

        self.one_hot_encoded = one_hot_encoded
        self.batch_normalization = batch_normalization_encoder
        self.imputation_method = imputation_method
        self.item_categories = model.item_categories
        self.mc_correct = model.mc_correct
        self.evaluation_interval_size = evaluation_interval_size

        if batch_size is None:
            batch_size = train_data.shape[0]
        if self.one_hot_encoded:
            input_dim = sum(model.item_categories)
        else:
            input_dim = len(model.item_categories)

        if hidden_layers_encoder is None:  # 1 layer with 2x number of categories as neurons is default
            hidden_layers_encoder = [2 * sum(model.item_categories)]

        self.encoder = StandardEncoder(
            input_dim=input_dim,
            latent_variables=model.latent_variables,
            hidden_dim=hidden_layers_encoder,
            batch_normalization=batch_normalization_encoder,
            nonlinear=nonlinear_encoder,
        )

        if not one_hot_encoded and imputation_method is None:
            raise ValueError("imputation_method must be supplied for non-one-hot encoded data.")

        train_data_irt = PytorchIRTDataset(
            data=train_data.to(device),
            one_hot_encoded=self.one_hot_encoded,
            item_categories=model.item_categories,
            imputation_method=imputation_method,
            mc_correct=model.mc_correct
        )
        self.data_loader = torch.utils.data.DataLoader(
            train_data_irt,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
        )

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(model.parameters()), lr=learning_rate, amsgrad=True
        )

        # Reduce learning rate when loss stops decreasing ("min")
        # we multiply the learning rate by the factor
        # patience: We need no improvement every step for it to trigger
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.6, patience=1
        )

        self.encoder.to(device)
        model.to(device)
        self._training_loop(model, max_epochs, learning_rate_updates_before_stopping)
        self.encoder.to("cpu")
        model.to("cpu")
        self.encoder.eval()
        model.eval()

        # store the latent theta scores of the training data
        # used for more efficient computation when using other methods
        self.training_theta_scores = self.theta_scores(
            train_data_irt.input_data.to("cpu")
        ).clone().detach()

    def _training_loop(
        self,
        model: BaseIRTModel,
        max_epochs: int,
        learning_rate_updates_before_stopping: int = 2,
    ):
        """
        The training loop for the model.

        Parameters
        ----------
        model : BaseIRTModel
            The model to train.
        max_epochs : int
            The maximum number of epochs to train for.
        scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
            The learning rate scheduler.
        learning_rate_updates_before_stopping : int, optional
            The number of times the learning rate can be reduced before stopping training. (default is 2)
        """
        for epoch in range(max_epochs):
            train_loss = self._train_step(model, epoch, learning_rate_updates_before_stopping)
            if np.isnan(train_loss):
                break
            if self.lr_update_count >= learning_rate_updates_before_stopping:
                break

        # Load the best model state
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state["state_dict_model"])
            self.encoder.load_state_dict(self.best_model_state["state_dict_encoder"])
            self.optimizer.load_state_dict(self.best_model_state["optimizer"])
            logger.info("Best model found after %d iterations (%d interval updates) with interval averaged loss %.4f.", self.best_avg_loss[1], self.best_avg_loss[1]/self.evaluation_interval_size, self.best_avg_loss[0])

    def _train_step(self, model: BaseIRTModel, epoch: int, learning_rate_updates_before_stopping: int = 5) -> torch.Tensor:
        """
        Perform a training step for the model.

        Parameters
        ----------
        model : BaseIRTModel
            The model to train.
        epoch : int
            The current epoch.
        learning_rate_updates_before_stopping : int, optional
            The number of times the learning rate can be reduced before stopping training. (default is 5)

        Returns
        -------
        float
            The loss after the training step.
        """
        self.encoder.train()
        model.train()
        epoch_mean_loss = 0
        learning_rate = self.optimizer.param_groups[0]['lr']
        for _, (batch, mask, input_batch, _) in enumerate(self.data_loader):
            # small batches leads to inaccurate batch variance, so we drop the last few observations
            if batch.shape[0] < 4 and self.batch_normalization:
                continue
            self.optimizer.zero_grad()
            batch_loss = self._train_batch(model, input_batch, batch, mask)
            if torch.isnan(batch_loss):
                logger.warning("Batch loss is nan. Try increasing batch size or lowering the learning rate.")
                return float("nan")

            batch_loss.backward()

            self.optimizer.step()
            self.total_iterations += 1
            epoch_mean_loss += batch_loss.item()

            avg_batch_loss = batch_loss.item()
            dynamic_print(f"Epoch: {epoch + 1}. Iteration: {self.total_iterations}. Average batch loss: {avg_batch_loss:.4f}. Current learning rate: {learning_rate:.4f}")

            self.batch_mean_losses.append(avg_batch_loss)
            if len(self.batch_mean_losses) > self.evaluation_interval_size:
                self.batch_mean_losses.pop(0)

            # Update the learning rate scheduler every self.evaluation_interval_size iterations if no improvement
            if (self.total_iterations) % self.evaluation_interval_size == 0 and self.total_iterations != 1:
                mean_loss = np.mean(self.batch_mean_losses)
                self.training_history["train_loss"].append(mean_loss)
                self.scheduler.step(mean_loss)

                # Check if the learning rate has been updated
                if learning_rate != self.optimizer.param_groups[0]['lr']:
                    self.lr_update_count += 1
                    learning_rate = self.optimizer.param_groups[0]['lr']
                
                if mean_loss < self.best_avg_loss[0]:
                    self.best_avg_loss = mean_loss, self.total_iterations
                    self.best_model_state = {
                        "state_dict_model": copy.deepcopy(model.state_dict()),
                        "state_dict_encoder": copy.deepcopy(self.encoder.state_dict()),
                        "optimizer": copy.deepcopy(self.optimizer.state_dict()) 
                    }
                
                if self.lr_update_count >= learning_rate_updates_before_stopping:
                    logger.info("Stopping training after %s learning rate updates.", learning_rate_updates_before_stopping)
                    break

        epoch_mean_loss /= len(self.data_loader)
        return epoch_mean_loss

    def _train_batch(self, model: BaseIRTModel, input_batch: torch.Tensor, batch: torch.Tensor, missing_mask: torch.Tensor):
        """
        Train the model on a batch of data.

        Parameters
        ----------
        model : BaseIRTModel
            The model to train.
        input_batch : torch.Tensor
            The input batch for the encoder.
        batch : torch.Tensor
            The batch of data for computing the loss.
        missing_mask : torch.Tensor
            The mask for missing data.
            
        Returns
        -------
        tuple
            The logits and loss after training on the batch.
        """
        outputs = model(self.encoder(input_batch))
        # check if all outputs are nan
        if torch.isnan(outputs).all():
            return torch.tensor(torch.nan)
        batch_loss = -model.log_likelihood(batch, outputs, missing_mask) / batch.shape[0]
        return batch_loss

    @torch.no_grad()
    def _batch_fit_measures(self, model: BaseIRTModel, input_batch: torch.Tensor, batch: torch.Tensor, missing_mask: torch.Tensor):
        """
        Calculate the fit measures for a batch.

        Parameters
        ----------
        model : BaseIRTModel
            The model to calculate the fit measures for.
        input_batch : torch.Tensor
            The input batch for the encoder.
        batch : torch.Tensor
            The batch of data.
        missing_mask : torch.Tensor
            The mask for missing data.

        Returns
        -------
        tuple
            The loss, log likelihood, and accuracy for the batch.
        """
        output = model(self.encoder(input_batch))
        log_likelihood = model.log_likelihood(batch, output, missing_mask=missing_mask)
        loss = -log_likelihood / batch.shape[0]
        return loss

    @torch.no_grad()
    def theta_scores(self, data: torch.Tensor):
        """
        Get the latent scores from an input dataset using the encoder.

        Parameters
        ----------
        data: torch.Tensor
            A 2D tensor with test data. Columns are items and rows are respondents.

        Returns
        -------
        torch.Tensor
            A 2D tensor of latent scores. Rows are respondents and latent variables are columns.
        """
        data = data.contiguous()
        return self.encoder(data)
