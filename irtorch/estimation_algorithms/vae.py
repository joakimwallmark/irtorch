import logging
import torch
from torch.distributions import Normal
from irtorch.models import BaseIRTModel
from irtorch.estimation_algorithms.ae import AE
from irtorch.estimation_algorithms.encoders import VariationalEncoder
from irtorch._internal_utils import PytorchIRTDataset
from irtorch.utils import one_hot_encode_test_data, decode_one_hot_test_data

logger = logging.getLogger("irtorch")

class VAE(AE):
    """
    Variational autoencoder neural network for fitting IRT models.

    """
    def __init__(self):
        super().__init__()
        self.iw_samples = 1
        self.annealing_epochs = 5
        self.anneal = True
        self.annealing_factor = 1.0

    def fit(
        self,
        model: BaseIRTModel,
        train_data: torch.Tensor,
        validation_data: torch.Tensor = None,
        one_hot_encoded: bool = True,
        hidden_layers_encoder: list[int] = None,
        nonlinear_encoder = torch.nn.ELU(),
        batch_normalization_encoder: bool = True,
        batch_size: int = 32,
        max_epochs: int = 1000,
        learning_rate: float = 0.004,
        learning_rate_update_patience: int = 4,
        learning_rate_updates_before_stopping: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        imputation_method: str = "zero",
        anneal: int = True,
        annealing_epochs: int = 5,
        iw_samples: int = 1,
    ):
        """
        Train the variational autoencoder model.

        Parameters
        ----------
        model : BaseIRTModel
            The model to fit. Needs to inherit :class:`irtorch.models.BaseIRTModel`.
        train_data : torch.Tensor
            The training data. Item responses should be coded 0, 1, ... and missing responses coded as nan or -1.
        validation_data : torch.Tensor, optional
            The validation data. (default is None)
        one_hot_encoded : bool, optional
            Whether the model uses one-hot encoded data. (default is False)
        hidden_layers_encoder : list[int], optional
            List of hidden layers for the encoder. Each element is a layer with the number of neurons represented as integers. If not provided, uses one hidden layer with 2 * sum(item_categories) neurons.
        nonlinear_encoder : torch.nn.Module, optional
            The non-linear function to use after each hidden layer in the encoder. (default is torch.nn.ELU())
        batch_normalization_encoder : bool, optional
            Whether to use batch normalization for the encoder. (default is True)
        batch_size : int, optional
            The batch size for training. (default is 64)
        max_epochs : int, optional
            The maximum number of epochs to train for. (default is 1000)
        learning_rate : float, optional
            The initial learning rate for the optimizer. (default is 0.004)
        learning_rate_update_patience : int, optional
            The number of epochs to wait before reducing the learning rate. (default is 4)
        learning_rate_updates_before_stopping : int, optional
            The number of times the learning rate can be reduced before stopping training. (default is 5)
        device : str, optional
            The device to run the model on. (default is "cuda" if available else "cpu".)
        imputation_method : str, optional
            The method to use for imputing missing data. (default is "zero")
        anneal : bool, optional
            Whether to anneal the KL divergence. (default is True)
        annealing_epochs : int, optional
            The number of epochs to anneal the KL divergence. (default is 5)
        iw_samples : int, optional
            The number of importance weighted samples to use. (default is 1)
        """
        if self.train_data is not None:
            self.train_data = torch.cat((self.train_data, train_data), dim=0).contiguous()
        else:
            self.train_data = train_data.contiguous()

        self.iw_samples = iw_samples
        self.annealing_epochs = annealing_epochs
        self.anneal = anneal
        self.imputation_method = imputation_method
        self.one_hot_encoded = one_hot_encoded

        if one_hot_encoded:
            input_dim = sum(model.modeled_item_responses)
        else:
            input_dim = len(model.modeled_item_responses)
        if hidden_layers_encoder is None:  # 1 layer with 2x number of categories as neurons is default
            hidden_layers_encoder = [2 * sum(model.modeled_item_responses)]

        self.encoder = VariationalEncoder(
            input_dim,
            model.latent_variables,
            hidden_layers_encoder,
            batch_normalization=batch_normalization_encoder,
            nonlinear=nonlinear_encoder,
        )
        
        # Re-initialize the training history
        self.training_history = {
            "train_loss": [],
            "validation_loss": [],
        }

        if self.one_hot_encoded:
            train_data = one_hot_encode_test_data(train_data, model.item_categories, encode_missing=model.model_missing)
            if validation_data is not None:
                validation_data = one_hot_encode_test_data(validation_data, model.item_categories, encode_missing=model.model_missing)

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
            list(self.encoder.parameters()) + list(model.parameters()), lr=learning_rate, amsgrad=True
        )

        # Reduce learning rate when loss stops decreasing ("min")
        # we multiply the learning rate by the factor
        # patience: We need no improvement after x epochs for it to trigger
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.6, patience=learning_rate_update_patience
        )

        self.encoder.to(device)
        model.to(device)
        self._training_loop(model, max_epochs, scheduler, validation_data, learning_rate_updates_before_stopping)
        self.encoder.to("cpu")
        model.to("cpu")
        self.encoder.eval()
        model.eval()

        # store the latent z scores of the training data
        # used for more efficient computation when using other methods
        if not self.one_hot_encoded:
            train_data = self.fix_missing_values(train_data)
        self.training_z_scores = self.z_scores(train_data).clone().detach()

    def _train_batch(self, model: BaseIRTModel, batch):
        """
        Train the model on a batch of data.

        Parameters
        ----------
        model : BaseIRTModel
            The model to fit.
        batch : torch.Tensor
            The batch of data.

        Returns
        -------
        tuple
            The logits and loss after training on the batch.
        """
        mean, logvar = self.encoder(batch)

        # takes iw_samples from the latent space for each data point (for importance weighting)
        mean = mean.repeat(self.iw_samples, 1)
        logvar = logvar.repeat(self.iw_samples, 1)
        z_samples = self.reparameterize(mean, logvar)

        batch_logits = model(z_samples)

        if self.one_hot_encoded:
            # for running with loss_function
            batch = decode_one_hot_test_data(batch, model.modeled_item_responses)
        batch_loss = self._loss_function(model, batch, batch_logits, z_samples, mean, logvar)
        return batch_loss

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    # IWAE Loss = E log (1/K Σ_{k=1}^{K} w_k) (ELBO for 1 iw sample)
    def _loss_function(
        self,
        model: BaseIRTModel,
        data: torch.Tensor,
        logits: torch.Tensor,
        z_samples: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ):
        """
        The IWAE loss function for the model as introduced by Burda et. al. (2015).
        When self.iw_samples == 1 this generalizes to the standard VAE ELBO loss.

        Parameters
        ----------
        model : BaseIRTModel
            The model to fit.
        data : torch.Tensor
            The input data.
        logits : torch.Tensor
            The logits output by the model.
        z_samples : torch.Tensor
            Samples from the encoder distribution
        mean : torch.Tensor
            Means from the encoder
        logvar : torch.Tensor
            Log of the variance from the encoder

        Returns
        -------
        torch.Tensor
            The calculated loss.
        """
        log_p_x_z = model.log_likelihood(
            data.repeat(self.iw_samples, 1),
            logits,
            loss_reduction="none",
        )

        # Reshape tensor by each iw-sample
        # we need to keep a dimension for each respondent, but we can sum over items
        log_p_x_z = log_p_x_z.view(
            self.iw_samples,
            log_p_x_z.shape[0] // (self.iw_samples * data.shape[1]),
            data.shape[1],
        ).sum(2)

        if self.iw_samples == 1:
            # ELBO (1 sample) can be computed using the true means and variances
            # sum up kl div for each person (row)
            kl_div = 0.5 * torch.sum(mean.pow(2) + logvar.exp() - 1 - logvar, dim=1)
            iwae_bound = log_p_x_z - self.annealing_factor * kl_div
        else:
            log_p_z = Normal(0, 1).log_prob(z_samples)
            log_q_z_x = Normal(mean, torch.exp(0.5 * logvar)).log_prob(z_samples)
            kl_div = log_q_z_x - log_p_z
            kl_div = kl_div.view(
                self.iw_samples, kl_div.shape[0] // self.iw_samples, kl_div.shape[1]
            ).sum(2)

            iwae_bound = log_p_x_z - self.annealing_factor * kl_div
            if self.iw_samples > 1:
                iwae_bound = torch.logsumexp(iwae_bound, dim=0) - torch.log(
                    torch.tensor(self.iw_samples).float()
                )  # Importance weighting log (1/K Σ_{k=1}^{K} w_k)

        # Estimate expectation
        return -iwae_bound.mean()

    def _batch_fit_measures(self, model: BaseIRTModel, batch):
        """
        Calculate the fit measures for a batch.

        Parameters
        ----------
        model : BaseIRTModel
            The model to fit.
        batch : torch.Tensor
            The batch of data.

        Returns
        -------
        tuple
            The loss, log likelihood, and accuracy for the batch.
        """
        encoder_mean, encoder_logvar = self.encoder(batch)
        output = model(encoder_mean)
        z_sample = self.reparameterize(encoder_mean, encoder_logvar)
        output_stochastic = model(z_sample)

        if self.one_hot_encoded:
            # for running with loss_function
            batch = decode_one_hot_test_data(batch, model.item_categories)

        # negative ce is log lik
        log_likelihood = model.log_likelihood(
            batch, output,
        )
        log_lik_stochastic = model.log_likelihood(
            batch, output_stochastic,
        )
        kl_loss = 0.5 * torch.sum(
            encoder_mean.pow(2) + encoder_logvar.exp() - 1 - encoder_logvar
        )
        loss = (log_lik_stochastic + self.annealing_factor * kl_loss) / batch.shape[0]
        return loss, log_likelihood

    def _impute_missing(self, model: BaseIRTModel, batch, missing_mask):
        if torch.sum(missing_mask) > 0:
            if self.imputation_method == "zero":
                imputed_batch = batch
                imputed_batch = imputed_batch.masked_fill(missing_mask.bool(), 0)
            elif self.imputation_method == "prior":
                imputed_batch = self._impute_missing_with_prior(model, batch, missing_mask)
            elif self.imputation_method == "mean":
                raise NotImplementedError("Mean imputation not implemented")
            else:
                raise ValueError(
                    f"Imputation method {self.imputation_method} not implmented"
                )
            return imputed_batch

        return batch

    @torch.inference_mode()
    def _impute_missing_with_prior(self, model: BaseIRTModel, batch, missing_mask):
        # get the decoder logits for the prior mean person
        prior_logits = model(
            torch.zeros(1, model.latent_variables).to(next(model.parameters()).device)
        )
        prior_mean_scores = self._mean_scores(model, prior_logits)
        batch[missing_mask.bool()] = prior_mean_scores.repeat(batch.shape[0], 1).to(
            next(model.parameters()).device
        )[missing_mask.bool()]

        return batch

    @torch.inference_mode()
    def _mean_scores(self, model: BaseIRTModel, output_logits):
        mean_scores = torch.zeros(len(model.modeled_item_responses))
        start = 0
        for item, item_cat in enumerate(model.modeled_item_responses):
            end = start + item_cat
            probabilities = torch.softmax(output_logits[:, start:end], dim=1)
            item_scores = torch.arange(item_cat).to(next(model.parameters()).device)
            mean_scores[item] = torch.sum(probabilities * item_scores)
            start = end

        return mean_scores

    @torch.inference_mode()
    def z_scores(
        self,
        data: torch.Tensor,
    ):
        """
        Get the z scores from an input

        Parameters
        ----------
        data: torch.Tensor
            A 2D tensor with test data. Columns are items and rows are respondents.
        
        Returns
        -------
        torch.Tensor
            A 2D tensor of latent scores. Rows are respondents and columns are latent variables.
        """
        data = data.contiguous()
        return self.encoder(data)[0]

    @torch.inference_mode()
    def sample_latent_variables(
        self, model: BaseIRTModel, sample_size: int, input_data: torch.Tensor = None
    ):
        if input_data is None:
            input_data = self.train_data
        else:
            input_data = input_data.contiguous().to(next(model.parameters()).device)

        # Sample test scores until we have sample_size
        indices = torch.randint(low=0, high=input_data.size(0), size=(sample_size,)).to(
            next(model.parameters()).device
        )
        samples = torch.index_select(input_data, 0, indices)
        # run the sample through the encoder
        mean, logvar = self.encoder(samples)
        return self.reparameterize(mean, logvar)

    @torch.inference_mode()
    def latent_credible_interval(
        self, input_data: torch.Tensor, alpha=0.05
    ) -> torch.tensor:
        input_data = input_data.contiguous()
        mean, logvar = self.encoder(input_data)
        # Create a Gaussian distribution with the calculated mean and std
        dist = Normal(mean, torch.exp(0.5 * logvar))

        # Prepare alpha tensors of the same shape as mean and std
        lower_alpha = torch.full_like(mean, alpha / 2)
        upper_alpha = torch.full_like(mean, 1 - alpha / 2)

        # Calculate the lower and upper bounds of the credible interval
        lower = dist.icdf(
            lower_alpha.clone().detach()
        )
        upper = dist.icdf(
            upper_alpha.clone().detach()
        )

        return lower, mean, upper
