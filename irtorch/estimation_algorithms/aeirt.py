import logging
import copy
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from irtorch.models import BaseIRTModel
from irtorch.estimation_algorithms import BaseIRTAlgorithm
from irtorch.estimation_algorithms.encoders import BaseEncoder, StandardEncoder
from irtorch._internal_utils import dynamic_print, PytorchIRTDataset
from irtorch.utils import one_hot_encode_test_data, decode_one_hot_test_data

logger = logging.getLogger('irtorch')

class AEIRT(BaseIRTAlgorithm, nn.Module):
    """
    Autoencoder neural network for fitting IRT models.

    Parameters
    ----------
    model : BaseIRTModel
        The model to fit. Needs to inherit irtorch.models.BaseIRTModel.
    encoder : BaseEncoder, optional
        The encoder instance to use. Needs to inherit irtorch.models.BaseEncoder. Overrides hidden_layers_encoder, nonlinear_encoder and batch_normalization_encoder if provided. (default is None)
    one_hot_encoded : bool, optional
        Whether the model uses one-hot encoded data. (default is False)
    hidden_layers_encoder : list[int], optional
        List of hidden layers for the encoder. Each element is a layer with the number of neurons represented as integers. If not provided, uses one hidden layer with 2 * sum(item_categories) neurons.
    nonlinear_encoder : torch.nn.Module, optional
        The non-linear function to use after each hidden layer in the encoder. (default is torch.nn.ELU())
    batch_normalization_encoder : bool, optional
        Whether to use batch normalization for the encoder. (default is True)
    """
    def __init__(
        self,
        model: BaseIRTModel,
        encoder: BaseEncoder = None,
        one_hot_encoded: bool = False,
        hidden_layers_encoder: list[int] = None,
        nonlinear_encoder = torch.nn.ELU(),
        batch_normalization_encoder: bool = True,
        summary_writer: SummaryWriter = None,
    ):
        super().__init__(model = model, one_hot_encoded=one_hot_encoded)
        self.imputation_method = "zero"
        self.summary_writer = summary_writer
        self.batch_normalization = batch_normalization_encoder

        if encoder is not None and self.model is not None:
            if encoder.latent_variables != self.model.latent_variables:
                raise ValueError(
                    "Encoder and decoder must have the same number of latent variables"
                )
        
        if encoder is not None:
            self.model.latent_variables = encoder.latent_variables
            self.encoder = encoder
        else:
            if one_hot_encoded:
                input_dim = sum(self.model.modeled_item_responses)
            else:
                input_dim = len(self.model.modeled_item_responses)
            if hidden_layers_encoder is None:  # 1 layer with 2x number of categories as neurons is default
                hidden_layers_encoder = [2 * sum(self.model.modeled_item_responses)]
            self.encoder = StandardEncoder(
                input_dim=input_dim,
                latent_variables=self.model.latent_variables,
                hidden_dim=hidden_layers_encoder,
                batch_normalization=batch_normalization_encoder,
                nonlinear=nonlinear_encoder,
            )

        self.training_z_scores = None
        self.training_history = {
            "train_loss": [],
            "validation_loss": [],
        }

        # always set to eval mode by default
        self.model.eval()
        self.encoder.eval()

    def forward(self, data):
        return self.model(self.encoder(data))

    def fit(
        self,
        train_data: torch.Tensor,
        validation_data: torch.Tensor = None,
        batch_size: int = 64,
        max_epochs: int = 1000,
        learning_rate: float = 0.04,
        learning_rate_update_patience: int = 4,
        learning_rate_updates_before_stopping: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        imputation_method: str = "zero",
    ):
        """
        Train the autoencoder model.

        Parameters
        ----------
        train_data : torch.Tensor
            The training data. Item responses should be coded 0, 1, ... and missing responses coded as nan or -1.
        validation_data : torch.Tensor, optional
            The validation data. (default is None)
        batch_size : int, optional
            The batch size for training. (default is 64)
        max_epochs : int, optional
            The maximum number of epochs to train for. (default is 1000)
        learning_rate : float, optional
            The initial learning rate for the optimizer. (default is 0.08)
        learning_rate_update_patience : int, optional
            The number of epochs to wait before reducing the learning rate. (default is 4)
        learning_rate_updates_before_stopping : int, optional
            The number of times the learning rate can be reduced before stopping training. (default is 5)
        device : str, optional
            The device to run the model on. (default is "cuda" if available else "cpu".)
        imputation_method : str, optional
            The method to use for imputing missing data. (default is "zero")
        """
        super().fit(train_data)
        self.imputation_method = imputation_method

        # Initialize the training history
        self.training_history = {
            "train_loss": [],
            "validation_loss": [],
        }

        if self.one_hot_encoded:
            train_data = one_hot_encode_test_data(train_data, self.model.item_categories, encode_missing=self.model.model_missing)
            if validation_data is not None:
                validation_data = one_hot_encode_test_data(validation_data, self.model.item_categories, encode_missing=self.model.model_missing)

        self.data_loader = torch.utils.data.DataLoader(
            PytorchIRTDataset(data=train_data.to(device)),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
        )
        if validation_data is not None:
            validation_data = validation_data.to(device)
            self.validation_data_loader = torch.utils.data.DataLoader(
                PytorchIRTDataset(data=validation_data),
                batch_size=batch_size,
                shuffle=False,
            )

        self.optimizer = torch.optim.Adam(
            [{"params": self.parameters()}], lr=learning_rate, amsgrad=True
        )

        # Reduce learning rate when loss stops decreasing ("min")
        # we multiply the learning rate by the factor
        # patience: We need to improvement after 5 epochs for it to trigger
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.6, patience=learning_rate_update_patience
        )

        self.to(device)
        self._training_loop(max_epochs, scheduler, validation_data, learning_rate_updates_before_stopping)
        self.to("cpu")
        self.eval()

        # store the latent z scores of the training data
        # used for more efficient computation when using other methods
        if not self.one_hot_encoded:
            train_data = self.fix_missing_values(train_data)
        self.training_z_scores = self.z_scores(train_data).clone().detach()

    def _training_loop(
        self,
        max_epochs: int,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        validation_data: torch.Tensor = None,
        learning_rate_updates_before_stopping: int = 5,
    ):
        """
        The training loop for the model.

        Parameters
        ----------
        max_epochs : int
            The maximum number of epochs to train for.
        validation_data : torch.Tensor, optional
            The validation data.
        """
        lr_update_count = 0
        best_loss = float('inf')
        prev_lr = [group['lr'] for group in self.optimizer.param_groups]
        for epoch in range(max_epochs):
            # logger.info("-----------\nEpoch: %s\n-----------", epoch)
            
            if hasattr(self, "anneal"):
                if self.anneal:
                    self.annealing_factor = min(1.0, epoch / self.annealing_epochs)
                else:
                    self.annealing_factor = 1.0

            train_loss = self._train_step(epoch)

            if validation_data is not None:
                validation_loss = self._validation_step()
                current_loss = validation_loss
                scheduler.step(validation_loss)
                dynamic_print(f"Epoch: {epoch}. Average training batch loss: {train_loss:.4f}. Average validation batch loss: {validation_loss:.4f}")
            else:
                current_loss = train_loss
                scheduler.step(train_loss)
                dynamic_print(f"Epoch: {epoch}. Average training batch loss function: {train_loss:.4f}")

            if current_loss < best_loss:
                best_loss = current_loss
                best_epoch = epoch
                best_model_state = { 'state_dict': copy.deepcopy(self.state_dict()),
                                    'optimizer': copy.deepcopy(self.optimizer.state_dict()) }
            
            # Break if the learning rate has been updated 3 times
            if lr_update_count >= learning_rate_updates_before_stopping:
                logger.info("Stopping training after %s learning rate updates.", learning_rate_updates_before_stopping)
                break

            current_lr = [group['lr'] for group in self.optimizer.param_groups]
            # Check if the learning rate has been updated
            if current_lr != prev_lr:
                lr_update_count += 1
                prev_lr = current_lr

            logger.debug("Current learning rate: %s", self.optimizer.param_groups[0]['lr'])

        # Load the best model state
        if best_model_state is not None:
            self.load_state_dict(best_model_state['state_dict'])
            self.optimizer.load_state_dict(best_model_state['optimizer'])
            logger.info("Best model found at epoch %s with loss %.4f.", best_epoch, best_loss)

        if self.summary_writer is not None:
            self.summary_writer.close()

    def _train_step(self, epoch):
        """
        Perform a training step for the model.

        Returns
        -------
        float
            The loss after the training step.
        """
        self.train()
        loss = 0

        for _, (batch, mask) in enumerate(self.data_loader):
            # small batches leads to inaccurate batch variance, so we drop the last few observations
            if batch.shape[0] < 4 and self.batch_normalization:
                continue
            batch = self._impute_missing(batch, mask)
            self.optimizer.zero_grad()
            batch_loss = self._train_batch(batch)
            batch_loss.backward()

            # After the backward pass, log the gradients and the parameters
            if self.summary_writer is not None:
                for name, param in self.model.named_parameters():
                    self.summary_writer.add_histogram(f'{name}', param, epoch)
                    self.summary_writer.add_histogram(f'{name}.grad', param.grad, epoch)

            self.optimizer.step()
            loss += batch_loss.item()

        # Calculate averge per batch loss and accuracy per epoch and print out what's happening
        loss /= len(self.data_loader)
        self.training_history["train_loss"].append(loss)
        return loss

    def _impute_missing(self, batch, missing_mask):
        if torch.sum(missing_mask) > 0:
            if self.imputation_method == "zero":
                imputed_batch = batch
                imputed_batch = imputed_batch.masked_fill(missing_mask.bool(), 0)
            elif self.imputation_method == "prior":
                imputed_batch = self._impute_missing_with_prior(batch, missing_mask)
            elif self.imputation_method == "mean":
                raise NotImplementedError("Mean imputation not implemented")
            else:
                raise ValueError(
                    f"Imputation method {self.imputation_method} not implmented"
                )
            return imputed_batch

        return batch

    def _impute_missing_with_prior(self, batch, missing_mask):
        raise NotImplementedError(
            "There is no prior for non-variational autoencoder models"
        )

    def _train_batch(self, batch: torch.Tensor):
        """
        Train the model on a batch of data.

        Parameters
        ----------
        batch : torch.Tensor
            The batch of data.

        Returns
        -------
        tuple
            The logits and loss after training on the batch.
        """
        batch_logits = self(batch)
        if self.one_hot_encoded:
            # for running with loss_function
            batch = decode_one_hot_test_data(batch, self.model.modeled_item_responses)
        batch_loss = -self.model.log_likelihood(batch, batch_logits) / batch.shape[0]
        return batch_loss

    @torch.inference_mode()  # decorator applies with torch.inference_mode(): to entire function
    def _validation_step(self):
        """
        Perform a validation step.

        Returns
        -------
        torch.Tensor
            The loss after the validation step.
        """
        self.eval()
        loss = 0
        for _, (batch, mask) in enumerate(self.validation_data_loader):
            batch = self._impute_missing(batch, mask)
            batch_loss, _ = self._batch_fit_measures(batch)

            loss += batch_loss.item()
        loss /= len(self.validation_data_loader)
        self.training_history["validation_loss"].append(loss)
        return loss

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

    @torch.inference_mode()
    def z_scores(self, data: torch.Tensor):
        """
        Get the latent scores from an input

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
