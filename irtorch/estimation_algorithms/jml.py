import logging
import copy
import torch
from torch.distributions import MultivariateNormal
from irtorch.models import BaseIRTModel
from irtorch._internal_utils import dynamic_print, sum_score
from irtorch.irt_dataset import PytorchIRTDataset
from irtorch.estimation_algorithms import BaseIRTAlgorithm

logger = logging.getLogger("irtorch")

class JML(BaseIRTAlgorithm):
    r"""
    Joint Maximum Likelihood (JML) for fitting IRT models.
    """
    def __init__(
        self,
    ):
        super().__init__()
        self.training_theta_scores = None
        self.covariance_matrix = None
        self.optimizer = None
        self.training_history = {
            "train_loss": [],
        }

    def fit(
        self,
        model: BaseIRTModel,
        train_data: torch.Tensor,
        learning_rate: float = 0.15,
        learning_rate_update_patience: int = 4,
        learning_rate_updates_before_stopping: int = 1,
        max_epochs: int = 10000,
        batch_size: int = None,
        start_thetas: torch.Tensor = None,
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
        learning_rate : float, optional
            The initial learning rate for the optimizer. (default is 0.15)
        learning_rate_update_patience : int, optional
            The number of epochs to wait before reducing the learning rate. (default is 4)
        learning_rate_updates_before_stopping : int, optional
            The number of times the learning rate can be reduced before stopping training. (default is 2)
        max_epochs : int, optional
            The maximum number of epochs to train for. (default is 1000)
        batch_size : int, optional
            The batch size for training. (default is None and uses the full dataset)
        start_thetas : torch.Tensor, optional
            The starting thetas for the training. (default is None and uses the standardized sum scores)
        device : str, optional
            The device to run the model on. (default is "cuda" if available else "cpu".)
        """
        super().fit(model = model, train_data = train_data)

        self.training_history = {
            "train_loss": [],
        }
        if batch_size is None:
            batch_size = train_data.shape[0]

        if start_thetas is None:
            start_thetas = sum_score(train_data, model.mc_correct)
            start_thetas = (start_thetas - start_thetas.mean()) / start_thetas.std()
            start_thetas = start_thetas.detach().unsqueeze(1).repeat(1, model.latent_variables)
            if model.latent_variables > 1:
                # add small noise to avoid all latent variables updating the same way
                start_thetas = start_thetas + torch.randn(start_thetas.shape) * 0.1
        elif start_thetas.shape[0] != train_data.shape[0]:
            raise ValueError("The number of starting thetas must be the same as the number of respondents.")
        elif start_thetas.shape[1] != model.latent_variables:
            raise ValueError("start_thetas must contain the same number of latent variables as the model.")

        self.training_theta_scores = torch.nn.Parameter(
            start_thetas.to(device)
        )

        self.optimizer = torch.optim.Adam(
            list([self.training_theta_scores]) + list(model.parameters()), lr=learning_rate, amsgrad=True
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.6, patience=learning_rate_update_patience
        )

        train_data_irt = PytorchIRTDataset(
            data=train_data.to(device),
            item_categories=model.item_categories,
            mc_correct=model.mc_correct
        )
        data_loader = torch.utils.data.DataLoader(
            train_data_irt,
            batch_size=batch_size,
            # shuffle=True,
            shuffle=False,
            pin_memory=False,
        )

        model.to(device)
        self._training_loop(
            model,
            data_loader,
            scheduler,
            max_epochs,
            learning_rate_updates_before_stopping
        )
        model.to("cpu")
        self.training_theta_scores = self.training_theta_scores.to("cpu")
        model.eval()

    def _training_loop(
        self,
        model: BaseIRTModel,
        data_loader: torch.utils.data.DataLoader,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        max_epochs: int,
        learning_rate_updates_before_stopping: int,
    ):
        """
        The training loop for the model.

        Parameters
        ----------
        model : BaseIRTModel
            The model to train.
        data_loader : torch.utils.data.DataLoader
            The data loader for the training data.
        scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
            The learning rate scheduler.
        max_epochs : int
            The maximum number of epochs to train for.
        learning_rate_updates_before_stopping : int, optional
            The number of times the learning rate can be reduced before stopping training.
        """
        lr_update_count = 0

        best_loss = float("inf")
        best_model_state = None
        best_epoch = 0
        prev_lr = [group["lr"] for group in self.optimizer.param_groups]
        model.train()

        for epoch in range(max_epochs):
            for _, (batch, mask, _, idx) in enumerate(data_loader):
                self.optimizer.zero_grad()
                model_out = model(self.training_theta_scores[idx])
                loss = -model.log_likelihood(batch, model_out, missing_mask=mask, loss_reduction="sum")
                loss.backward()
                self.optimizer.step()
                
                train_loss = loss.item()
                self.training_history["train_loss"].append(loss.item())
                
                current_loss = train_loss
                scheduler.step(train_loss)
                dynamic_print(f"Epoch: {epoch}. Average training batch loss function: {train_loss:.4f}")

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_epoch = epoch
                    best_model_state = {
                        "theta": copy.deepcopy(self.training_theta_scores.data),
                        "state_dict": copy.deepcopy(model.state_dict()),
                        "optimizer": copy.deepcopy(self.optimizer.state_dict()) 
                    }

                current_lr = [group["lr"] for group in self.optimizer.param_groups]
                if current_lr != prev_lr:
                    lr_update_count += 1
                    prev_lr = current_lr
                    logger.info("Current learning rate: %s", current_lr)

                if lr_update_count >= learning_rate_updates_before_stopping:
                    break


            if lr_update_count >= learning_rate_updates_before_stopping:
                break

        # Load the best model state
        if best_model_state is not None:
            self.training_theta_scores = best_model_state["theta"]
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
