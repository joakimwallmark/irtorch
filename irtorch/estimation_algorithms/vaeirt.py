import logging
import torch
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from irtorch.models import BaseIRTModel
from irtorch.estimation_algorithms.aeirt import AEIRT
from irtorch.estimation_algorithms.encoders import BaseEncoder
from irtorch.estimation_algorithms.encoders import VariationalEncoder
from irtorch.helper_functions import decode_one_hot_test_data

logger = logging.getLogger('irtorch')

class VAEIRT(AEIRT):
    def __init__(
        self,
        model: BaseIRTModel,
        encoder: BaseEncoder = None,
        one_hot_encoded: bool = False,
        hidden_layers_encoder: list[int] = None,
        nonlinear_encoder = torch.nn.ELU(),
        batch_normalization_encoder: bool = True,
        iw_samples: int = 1,
        anneal: bool = True,
        annealing_epochs: int = 5,
        summary_writer: SummaryWriter = None,
    ):
        """
        Initialize the autoencoder IRT neural network.

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
        iw_samples : int, optional
            The number of importance weighted samples to use. (default is 1)
        anneal : bool, optional
            Whether to anneal the KL divergence. (default is True)
        annealing_epochs : int, optional
            The number of epochs to anneal the KL divergence. (default is 5)
        """
        self.iw_samples = iw_samples
        self.annealing_epochs = annealing_epochs
        self.anneal = anneal
        self.annealing_factor = 1.0

        if encoder is not None:
            self.encoder = encoder
        else:
            if one_hot_encoded:
                input_dim = sum(model.modeled_item_responses)
            else:
                input_dim = len(model.modeled_item_responses)
            if hidden_layers_encoder is None:  # 1 layer with 2x number of categories as neurons is default
                hidden_layers_encoder = [2 * sum(model.modeled_item_responses)]
            encoder = (
                VariationalEncoder(
                    input_dim,
                    model.latent_variables,
                    hidden_layers_encoder,
                    batch_normalization=batch_normalization_encoder,
                    nonlinear=nonlinear_encoder,
                )
                if encoder is None
                else encoder
            )
        super().__init__(
            model=model,
            encoder=encoder,
            one_hot_encoded=one_hot_encoded,
            summary_writer=summary_writer,
        )

    def fit(
        self,
        train_data: torch.Tensor,
        validation_data: torch.Tensor = None,
        batch_size: int = 32,
        max_epochs: int = 1000,
        learning_rate: float = 0.004,
        learning_rate_update_patience: int = 4,
        learning_rate_updates_before_stopping: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        imputation_method: str = "zero",
        verbose: bool = False,
        anneal: int = True,
        annealing_epochs: int = 5,
        iw_samples: int = 1,
    ):
        """
        Train the variational autoencoder model.

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
            The initial learning rate for the optimizer. (default is 0.004)
        learning_rate_update_patience : int, optional
            The number of epochs to wait before reducing the learning rate. (default is 4)
        learning_rate_updates_before_stopping : int, optional
            The number of times the learning rate can be reduced before stopping training. (default is 5)
        device : str, optional
            The device to run the model on. (default is "cuda" if available else "cpu".)
        imputation_method : str, optional
            The method to use for imputing missing data. (default is "zero")
        verbose : bool, optional
            Whether to print out verbose training logs. (default is False)
        """
        self.iw_samples = iw_samples
        self.annealing_epochs = annealing_epochs
        self.anneal = anneal
        super().fit(
            train_data=train_data,
            validation_data=validation_data,
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            learning_rate_update_patience=learning_rate_update_patience,
            learning_rate_updates_before_stopping=learning_rate_updates_before_stopping,
            device=device,
            imputation_method=imputation_method,
            verbose=verbose
        )

    def forward(self, data):
        """
        Samples self.iw_samples logit outputs for each respondent in the input.

        Parameters
        ----------
        data : torch.Tensor
            The input test data

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
            Four 2D tensors with rows corresponding respondent samples.
            -   Output logits tensor
            -   z samples tensor
            -   Encoder means for each sample
            -   Encoder log variances for each sample
            A sample for each respondent reoccurs every data.shape[0]'th row.
        """
        mean, logvar = self.encoder(data)

        # takes iw_samples from the latent space for each data point (for importance weighting)
        mean = mean.repeat(self.iw_samples, 1)
        logvar = logvar.repeat(self.iw_samples, 1)
        z_samples = self.reparameterize(mean, logvar)

        logits = self.model(z_samples)
        return logits, z_samples, mean, logvar

    def _train_batch(self, batch):
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
        batch_logits, z_samples, mean, logvar = self(batch)
        if self.one_hot_encoded:
            # for running with loss_function
            batch = decode_one_hot_test_data(batch, self.model.modeled_item_responses)
        batch_loss = self._loss_function(batch, batch_logits, z_samples, mean, logvar)
        return batch_loss

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    # IWAE Loss = E log (1/K Σ_{k=1}^{K} w_k) (ELBO for 1 iw sample)
    def _loss_function(
        self,
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
        log_p_x_z = self.model.log_likelihood(
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

    def _batch_fit_measures(self, batch):
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
        encoder_mean, encoder_logvar = self.encoder(batch)
        output = self.model(encoder_mean)
        z_sample = self.reparameterize(encoder_mean, encoder_logvar)
        output_stochastic = self.model(z_sample)

        if self.one_hot_encoded:
            # for running with loss_function
            batch = decode_one_hot_test_data(batch, self.model.item_categories)

        # negative ce is log lik
        log_likelihood = self.model.log_likelihood(
            batch, output,
        )
        log_lik_stochastic = self.model.log_likelihood(
            batch, output_stochastic,
        )
        kl_loss = 0.5 * torch.sum(
            encoder_mean.pow(2) + encoder_logvar.exp() - 1 - encoder_logvar
        )
        loss = (log_lik_stochastic + self.annealing_factor * kl_loss) / batch.shape[0]
        return loss, log_likelihood

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
    def _mean_scores(self, output_logits):
        mean_scores = torch.zeros(len(self.model.modeled_item_responses))
        start = 0
        for item, item_cat in enumerate(self.model.modeled_item_responses):
            end = start + item_cat
            probabilities = torch.softmax(output_logits[:, start:end], dim=1)
            item_scores = torch.arange(item_cat).to(next(self.parameters()).device)
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
        self, sample_size: int, input_data: torch.Tensor = None
    ):
        if input_data is None:
            input_data = self.train_data
        else:
            input_data = input_data.contiguous().to(next(self.parameters()).device)

        # Sample test scores until we have sample_size
        indices = torch.randint(low=0, high=input_data.size(0), size=(sample_size,)).to(
            next(self.parameters()).device
        )
        samples = torch.index_select(input_data, 0, indices)
        # run the sample through the encoder
        mean, logvar = self.encoder(samples)
        return self.reparameterize(mean, logvar)

    @torch.inference_mode()
    def latent_credible_interval(
        self, input_data: torch.Tensor, alpha=0.05
    ) -> torch.tensor:
        input_data = input_data.contiguous().to(self.device)
        mean, logvar = self.encoder(input_data)
        # Create a Gaussian distribution with the calculated mean and std
        dist = Normal(mean, torch.exp(0.5 * logvar))

        # Prepare alpha tensors of the same shape as mean and std
        lower_alpha = torch.full_like(mean, alpha / 2)
        upper_alpha = torch.full_like(mean, 1 - alpha / 2)

        # Calculate the lower and upper bounds of the credible interval
        lower = dist.icdf(
            lower_alpha.clone().detach().to(next(self.parameters()).device)
        )
        upper = dist.icdf(
            upper_alpha.clone().detach().to(next(self.parameters()).device)
        )

        return lower, mean, upper
