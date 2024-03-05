import sys
import logging
import torch
from torch.distributions import MultivariateNormal
from irtorch.models.base_irt_model import BaseIRTModel
from irtorch.estimation_algorithms.base_irt_algorithm import BaseIRTAlgorithm
from irtorch.estimation_algorithms.aeirt import AEIRT
from irtorch.quantile_mv_normal import QuantileMVNormal
from irtorch.gaussian_mixture_torch import GaussianMixtureTorch
from irtorch._internal_utils import output_to_item_entropy, random_guessing_data, linear_regression, dynamic_print
from irtorch.outlier_detector import OutlierDetector
from irtorch.utils import one_hot_encode_test_data

logger = logging.getLogger('irtorch')

class IRTScorer:
    def __init__(self, model: BaseIRTModel, algorithm: BaseIRTAlgorithm):
        """
        Initializes the IRTScorer class with a given model and fitting algorithm.

        Parameters
        ----------
        model : BaseIRTModel
            BaseIRTModel object.
        algorithm : BaseIRTAlgorithm
            BaseIRTAlgorithm object.
        """
        self.model = model
        self.algorithm = algorithm
        self.latent_density = None


    @torch.inference_mode()
    def approximate_latent_density(
        self,
        z_scores: torch.Tensor,
        approximation: str = "qmvn",
        cv_n_components: list[int] = None,
    ):
        """
        Approximate the latent space density.

        Parameters
        ----------
        z_scores : torch.Tensor
            A 2D tensor with z scores. Each row represents one respondent, and each column an item.
        approximation : str, optional
            The approximation method to use. (default is 'qmvn')
            - 'qmvn' for quantile multivariate normal approximation of a multivariate joint density function (QuantileMVNormal class).
            - 'gmm' for an sklearn gaussian mixture model.

        cv_n_components: int, optional
            The number of guassian components to use for 'gmm'. The best performing number . (default is 5)

        Returns
        -------
        torch.Tensor
            A 2D tensor of latent scores, with latent variables as columns.
        """
        if approximation == "gmm":
            cv_n_components = [2, 3, 4, 5, 10] if cv_n_components is None else cv_n_components
            self.latent_density = GaussianMixtureTorch()
            self.latent_density.fit(z_scores, cv_n_components)
        elif approximation == "qmvn":
            self.latent_density = QuantileMVNormal()
            self.latent_density.fit(z_scores)
        else:
            raise ValueError("Invalid approximation method. Choose either 'qmvn' or 'gmm'.")


    @torch.inference_mode()
    def min_max_z_for_integration(
        self,
        z: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the minimum and maximum z score for approximating integrals over the latent space. Uses one standard deviation below/above the min/max of each z score vector.

        Parameters
        ----------
        z : torch.Tensor, optional
            A 2D tensor. Columns are each latent variable, rows are respondents. Default is None and uses training data z scores.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple with 1D tensors, containing the min and max integration z scores of each latent variable.
        """
        if z is None:
            z = self.algorithm.training_z_scores

        z_min = z.min(dim=0)[0]
        z_max = z.max(dim=0)[0]
        z_stds = z.std(dim=0)

        return z_min - z_stds, z_max + z_stds


    @torch.inference_mode()
    def latent_scores(
        self,
        data: torch.Tensor,
        scale: str = "bit",
        z: torch.Tensor = None,
        z_estimation_method: str = "ML",
        ml_map_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lbfgs_learning_rate: float = 0.3,
        eap_z_integration_points: int = None,
        bit_score_one_dimensional: bool = False,
        bit_score_population_z: torch.Tensor = None,
        bit_score_grid_points: int = 300,
        bit_score_z_grid_method: str = None,
        bit_score_start_z: torch.Tensor = None,
        bit_score_start_z_guessing_probabilities: list[float] = None,
        bit_score_start_z_guessing_iterations: int = 10000,
        bit_score_items: list[int] = None
    ):
        """
        Returns the latent scores for given test data using encoder the neural network (NN), maximum likelihood (ML), expected a posteriori (EAP) or maximum a posteriori (MAP). 
        ML and MAP uses the LBFGS algorithm. EAP and MAP are not recommended for non-variational autoencoder models as there is nothing pulling the latent distribution towards a normal.        
        EAP for models with more than three factors is not recommended since the integration grid becomes huge.

        Parameters
        ----------
        data : torch.Tensor
            A 2D tensor with test data. Each row represents one respondent, each column an item.
        scale : str, optional
            The scoring method to use. Can be 'bit' or 'z'. (default is 'bit')
        z : torch.Tensor, optional
            For bit scores. A 2D tensor containing the pre-estimated z scores for each respondent in the data. If not provided, will be estimated using z_estimation_method. Each row corresponds to one respondent and each column represents a latent variable. (default is None)
        z_estimation_method : str, optional
            Method used to obtain the z scores. Also used for bit scores as they require the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'ML')
        ml_map_device: str, optional
            For ML and MAP. The device to use for computation. Can be 'cpu' or 'cuda'. (default is "cuda" if available else "cpu")
        lbfgs_learning_rate: float, optional
            For ML and MAP. The learning rate to use for the LBFGS optimizer. Try lowering this if your loss runs rampant. (default is 0.3)
        eap_z_integration_points: int, optional
            For EAP. The number of integration points for each latent variable. (default is 'None' and uses a function of the number of latent variables)
        bit_score_one_dimensional: bool, optional
            Whether to estimate one combined bit score for a multidimensional model. (default is False)
        bit_score_population_z: torch.Tensor, optional
            A 2D tensor with z scores of the population. Used to estimate relationships between each z and sum scores. Columns are latent variables and rows are respondents. (default is None and uses z_estimation_method with the model training data)
        bit_score_grid_points : int, optional
            The number of points to use for computing bit score. More steps lead to more accurate results. (default is 300)
        bit_score_z_grid_method : str, optional
            Method used to obtain the z score grid for bit score computation. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is None and uses z_estimation_method)
        bit_score_start_z : int, optional
            The z score used as the starting point for bit score computation. Computed automatically if not provided. (default is 'None')
        bit_score_start_z_guessing_probabilities: list[float], optional
            Custom guessing probabilities for each item. The same length as the number of items. Guessing is not supported for polytomously scored items and the probabilities for them will be ignored. (default is None and uses no guessing or, for multiple choice models, 1 over the number of item categories)
        bit_score_start_z_guessing_iterations: int, optional
            The number of iterations to use for approximating a minimum z when guessing is incorporated. (default is 10000)
        bit_score_items: list[int], optional
            The item indices for the items to use to compute the bit scores. (default is 'None' and uses all items)
        Returns
        -------
        torch.Tensor
            A 2D tensor of latent scores, with latent variables as columns.
        """
        if scale not in ["bit", "z"]:
            raise ValueError("Invalid scale. Choose either 'z' or 'bit'.")
        if z_estimation_method not in ["NN", "ML", "EAP", "MAP"]:
            raise ValueError("Invalid z_estimation_method. Choose either 'NN', 'ML', 'EAP' or 'MAP'.")

        data = data.contiguous()
        if data.dim() == 1:  # if we have only one observations
            data = data.view(1, -1)

        if z is None:
            if self.algorithm.one_hot_encoded and z_estimation_method in ["NN", "ML", "MAP"]:
                one_hot_data = one_hot_encode_test_data(data, self.model.item_categories, encode_missing=self.model.model_missing)
                if isinstance(self.algorithm, AEIRT):
                    z = self.algorithm.z_scores(one_hot_data).clone()
                else:
                    z = torch.zeros(one_hot_data.shape[0], self.model.latent_variables).float()
            
            data = self.algorithm.fix_missing_values(data)

            logger.info("%s estimation of z scores.", z_estimation_method)
            if not self.algorithm.one_hot_encoded and z_estimation_method in ["NN", "ML", "MAP"]:
                z = self.algorithm.z_scores(data).clone()
            if z_estimation_method in ["ML", "MAP"]:
                z = self._ml_map_z_scores(data, z, z_estimation_method, learning_rate=lbfgs_learning_rate, device=ml_map_device)
            elif z_estimation_method == "EAP":
                z = self._eap_z_scores(data, eap_z_integration_points)

        if scale == "z":
            return z
        elif scale == "bit":
            if bit_score_z_grid_method is None:
                bit_score_z_grid_method = z_estimation_method
            return self.bit_scores_from_z(
                z=z,
                start_z=bit_score_start_z,
                population_z=bit_score_population_z,
                one_dimensional=bit_score_one_dimensional,
                z_estimation_method=bit_score_z_grid_method,
                grid_points=bit_score_grid_points,
                items=bit_score_items,
                start_z_guessing_probabilities=bit_score_start_z_guessing_probabilities,
                start_z_guessing_iterations=bit_score_start_z_guessing_iterations
            )[0]

    @torch.inference_mode(False)
    def _ml_map_z_scores(
        self, 
        data: torch.Tensor, 
        encoder_z_scores:torch.Tensor = None,
        z_estimation_method: str = "ML",
        learning_rate: float = 0.3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> torch.Tensor:
        """
        Get the latent scores from test data using an already fitted model.

        Parameters
        ----------
        data: torch.Tensor
            A 2D tensor with test data. Columns are items and rows are respondents
        encoder_z_scores: torch.Tensor
            A 2D tensor with the z scores of the training data. Columns are latent variables and rows are respondents.
        z_estimation_method: str, optional
            Method used to obtain the z scores. Can be 'ML', 'MAP' for maximum likelihood or maximum a posteriori respectively. (default is 'ML')
        learning_rate: float, optional
            The learning rate to use for the LBFGS optimizer. (default is 0.3)
        device: str, optional
            The device to use for computation. Can be 'cpu' or 'cuda'. (default is "cuda" if available else "cpu")
        
        Returns
        -------
        torch.Tensor
            A torch.Tensor with the z scores. The columns are latent variables and rows are respondents.
        """
        try:
            if self.algorithm.training_z_scores is None:
                raise ValueError("Please fit the model before computing latent scores.")
                
            if z_estimation_method == "MAP": # Approximate prior
                train_z_scores = self.algorithm.training_z_scores
                # Center the data and compute the covariance matrix.
                mean_centered_z_scores = train_z_scores - train_z_scores.mean(dim=0)
                cov_matrix = mean_centered_z_scores.T @ mean_centered_z_scores / (train_z_scores.shape[0] - 1)
                # Create prior (multivariate normal distribution).
                prior_density = MultivariateNormal(torch.zeros(train_z_scores.shape[1]), cov_matrix)

            # Ensure decoder parameters gradients are not updated
            self.model.requires_grad_(False)

            if encoder_z_scores is None:
                encoder_z_scores = torch.zeros(data.shape[0], self.model.latent_variables).float()

            if device == "cuda":
                self.model = self.model.to(device)
                encoder_z_scores = encoder_z_scores.to(device)
                data = data.to(device)
                max_iter = 30
            else:
                max_iter = 20

            # Initial guess for the z_scores are the outputs from the encoder
            optimized_z_scores = encoder_z_scores.clone().detach().requires_grad_(True)

            optimizer = torch.optim.LBFGS([optimized_z_scores], lr = learning_rate)
            loss_history = []
            tolerance = 1e-8

            def closure():
                optimizer.zero_grad()
                logits = self.model(optimized_z_scores)
                if z_estimation_method == "MAP": # maximize -log likelihood - log prior
                    loss = -self.model.log_likelihood(data, logits, loss_reduction = "sum") - prior_density.log_prob(optimized_z_scores).sum()
                else: # maximize -log likelihood for ML
                    loss = -self.model.log_likelihood(data, logits, loss_reduction = "sum")
                loss.backward()
                return loss

            for i in range(max_iter):
                optimizer.step(closure)
                with torch.no_grad():
                    logits = self.model(optimized_z_scores)
                    if z_estimation_method == "MAP": # maximize -log likelihood - log prior
                        loss = -self.model.log_likelihood(data, logits, loss_reduction = "sum") - prior_density.log_prob(optimized_z_scores).sum()
                    else: # maximize -log likelihood for ML
                        loss = -self.model.log_likelihood(data, logits, loss_reduction = "sum")
                    loss = loss.item()

                denominator = data.numel()
                dynamic_print(f'Iteration {i+1}: Current Loss = {loss}')
                if len(loss_history) > 0 and abs(loss - loss_history[-1]) / denominator < tolerance:
                    logger.info("Converged at iteration %s.", i+1)
                    break

                loss_history.append(loss)
        except Exception as e:
            logger.error("Error in %s iteration %s: %s", z_estimation_method, i+1, e)
            raise e
        finally:
            # Reset requires_grad for decoder parameters if we want to train decoder later
            self.model = self.model.to("cpu")
            optimized_z_scores = optimized_z_scores.detach().to("cpu")
            self.model.requires_grad_(True)
        return optimized_z_scores
    
    @torch.inference_mode()
    def _eap_z_scores(self, data: torch.Tensor, grid_points: int = None) -> torch.Tensor:
        """
        Get the latent z scores from test data using an already fitted model.

        Parameters
        ----------
        data: torch.Tensor
            A 2D tensor with test data. Columns are items and rows are respondents
        grid_points: int, optional
            The number of grid points for each latent variable. (default is 'None' and uses a function of the number of latent variables)
        
        Returns
        -------
        torch.Tensor
            A torch.Tensor with the z scores. The columns are latent variables and rows are respondents.
        """
        if self.algorithm.training_z_scores is None:
            raise ValueError("Please fit the model before computing latent scores.")

        # Get grid for integration.
        train_z_scores = self.algorithm.training_z_scores
        if grid_points is None:
            if train_z_scores.shape[1] > 4:
                raise ValueError("EAP is not implemented for more than 4 latent variables because of large integration grid.")
            grid_points = {
                1: 200,
                2: 15,
                3: 7,
                4: 5
            }.get(train_z_scores.shape[1], 100)
            
        z_grid = self._z_grid(train_z_scores, grid_size=grid_points)
        
        # Center the data and compute the covariance matrix.
        mean_centered_z_scores = train_z_scores - train_z_scores.mean(dim=0)
        cov_matrix = mean_centered_z_scores.T @ mean_centered_z_scores / (train_z_scores.shape[0] - 1)
        # Create prior (multivariate normal distribution).
        prior_density = MultivariateNormal(torch.zeros(train_z_scores.shape[1]), cov_matrix)
        # Compute log of the prior.
        log_prior = prior_density.log_prob(z_grid)

        # Compute the log likelihood.
        logits = self.model(z_grid)
        replicated_data = data.repeat_interleave(z_grid.shape[0], dim=0)
        replicated_logits = torch.cat([logits] * data.shape[0], dim=0)
        replicated_z_grid = torch.cat([z_grid] * data.shape[0], dim=0)
        log_prior = torch.cat([log_prior] * data.shape[0], dim=0)
        grid_log_likelihoods = self.model.log_likelihood(replicated_data, replicated_logits, loss_reduction = "none")
        grid_log_likelihoods = grid_log_likelihoods.view(-1, data.shape[1]).sum(dim=1) # sum likelihood over items

        # Approximate integration integral(p(x|z)*p(z)dz)
        # p(x|z)p(z) / sum(p(x|z)p(z)) needs to sum to 1 for each respondent response pattern.
        log_posterior = (log_prior + grid_log_likelihoods).view(-1, z_grid.shape[0]) # each row is one respondent
        # convert to float 64 to prevent 0 probabilities
        exp_log_posterior = log_posterior.to(dtype=torch.float64).exp()
        posterior = (exp_log_posterior.T / exp_log_posterior.sum(dim=1)).T.view(-1, 1) # transform to one column

        # Get expected z
        posterior_times_z = replicated_z_grid * posterior
        expected_z = posterior_times_z.reshape(-1, z_grid.shape[0], posterior_times_z.shape[1])
        return expected_z.sum(dim=1).to(dtype=torch.float32)

    @torch.inference_mode()
    def _z_grid(self, z_scores: torch.Tensor, grid_size: int = None):
        """
        Returns a new z score tensor covering a large range of latent variable values in a grid.
        
        Parameters
        ----------
        z_scores: torch.Tensor
            The input test scores. Typically obtained from the training data.
        grid_size: int
            The number of grid points for each latent variable.
        
        Returns
        -------
        torch.Tensor
            A tensor with all combinations of latent variable values. Latent variables as columns.
        """
        if grid_size is None:
            grid_size = int(1e7 / (100 ** z_scores.shape[1]))
        min_vals, _ = z_scores.min(dim=0)
        max_vals, _ = z_scores.max(dim=0)
        # add / remove 0.25 times the diff minus max and min in the training data
        plus_minus = (max_vals-min_vals) * 0.25
        min_vals = min_vals - plus_minus
        max_vals = max_vals + plus_minus

        # Using linspace to generate a range between min and max for each latent variable
        grids = [torch.linspace(min_val, max_val, grid_size) for min_val, max_val in zip(min_vals, max_vals)]

        # Use torch.cartesian_prod to generate all combinations of the tensors in grid
        result = torch.cartesian_prod(*grids)
        # Ensure result is always a 2D tensor even with 1 latent variable
        return result.view(-1, z_scores.shape[1])

    def get_bit_score_starting_z(
        self,
        z_estimation_method: str = "ML",
        ml_map_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lbfgs_learning_rate: float = 0.3,
        items: list[int] = None,
        start_all_incorrect: bool = False,
        train_z: torch.Tensor = None,
        guessing_probabilities: list[float] = None,
        guessing_iterations: int = 10000,
    ):
        """
        Returns the starting z score from which to compute bit scores.
        
        Parameters
        ----------
        z_estimation_method : str, optional
            Method used to obtain the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'ML')
        ml_map_device: str, optional
            For ML and MAP. The device to use for computation. Can be 'cpu' or 'cuda'. (default is "cuda" if available else "cpu")
        lbfgs_learning_rate: float, optional
            For ML and MAP. The learning rate to use for the LBFGS optimizer. (default is 0.3)
        items: list[int], optional
            The item indices for the items to use to compute the bit scores. (default is None and uses all items)
        start_all_incorrect: bool, optional
            Whether to compute the starting z scores based incorrect responses. If false, starting z is computed based on relationships between each latent variable and the item responses. (default is False)
        train_z : torch.Tensor, optional
            A 2D tensor with the training data z scores. Used to estimate relationships between z and getting the items correct when start_all_incorrect is False. Columns are latent variables and rows are respondents. (default is None and uses encoder z scores from the model training data)
        guessing_probabilities: list[float], optional
            The guessing probability for each item. The same length as the number of items. Guessing is not supported for polytomously scored items and the probabilities for them will be ignored. (default is None and uses no guessing or, for multiple choice models, 1 over the number of item categories)
        guessing_iterations: int, optional
            The number of iterations to use for approximating a minimum z when guessing is incorporated. (default is 200)
        
        
        Returns
        -------
        torch.Tensor
            A tensor with all the starting z values.
        """
        items = items or list(range(len(self.model.modeled_item_responses)))
        mc_correct = torch.tensor(self.model.mc_correct) if self.model.mc_correct else None
        if not all(isinstance(item, int) for item in items):
            raise ValueError("items must be a list of integers.")
        
        if guessing_probabilities:
            if len(guessing_probabilities) != len(items) or not all(0 <= num < 1 for num in guessing_probabilities):
                raise ValueError("guessing_probabilities must be list of floats with the same length as the number of items and values between 0 and 1.")

        selected_item_categories = [self.model.item_categories[i] for i in items]

        if guessing_probabilities is None and mc_correct is not None:
            guessing_probabilities = [1 / categories for categories in selected_item_categories]

        if not start_all_incorrect:
            if train_z is None:
                train_z = self.algorithm.training_z_scores
            # Which latent variables are inversely related to the test scores?
            item_sum_scores = self.model.expected_item_sum_score(train_z)
            test_weights = linear_regression(train_z, item_sum_scores.sum(dim=1).reshape(-1, 1))[1:]
            inverted_scale = torch.where(test_weights < 0, torch.tensor(-1), torch.tensor(1)).reshape(-1)
            
            # Which latent variables are positively related to the item scores?
            directions = self.model.item_z_relationship_directions(train_z)
            item_z_postive = (inverted_scale * directions) >= 0 # Invert item relationship if overall test relationship is inverted

        if guessing_probabilities is None:
            if start_all_incorrect:
                logger.info("Computing z for all incorrect responses to get minimum bit score.")
                starting_z = self.latent_scores(torch.zeros((1, len(items))).float(), scale="z", z_estimation_method=z_estimation_method, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate)
            else:
                # Get minimum score in relation to each latent variable
                min_sum_score = torch.zeros((len(items), self.model.latent_variables))
                if self.model.model_missing:
                    min_sum_score[~item_z_postive] = (torch.tensor(selected_item_categories) - 2).view(-1, 1).float().repeat(1, self.model.latent_variables)[~item_z_postive]
                else:
                    min_sum_score[~item_z_postive] = (torch.tensor(selected_item_categories) - 1).view(-1, 1).float().repeat(1, self.model.latent_variables)[~item_z_postive]

                # get the minimum z scores based on the sum scores
                starting_z = torch.zeros((1, self.model.latent_variables)).float()
                for z in range(self.model.latent_variables):
                    logger.info("Computing minimum bit score z for latent variable %s.", z+1)
                    starting_z[:, z] = self.latent_scores(min_sum_score.float()[:, z], scale="z", z_estimation_method=z_estimation_method, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate)[:, z]
        else:
            if mc_correct is None:
                selected_correct = torch.ones(len(items))
                random_data = random_guessing_data(selected_item_categories, guessing_iterations, guessing_probabilities)
            else:
                selected_correct = mc_correct.index_select(0, torch.tensor(items))
                random_data = random_guessing_data(
                    selected_item_categories,
                    guessing_iterations,
                    guessing_probabilities,
                    selected_correct
                )

            if start_all_incorrect:
                # With one-dimensional bit scores, guessing for all items makes sense
                logger.info("Approximating z from random guessing data to get minimum bit score.")
                guessing_z = self.latent_scores(random_data, scale="z", z_estimation_method=z_estimation_method, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate)
            else:
                # Random for positively related and set to correct for others
                guessing_z = torch.zeros(random_data.shape[0], self.model.latent_variables)
                for z in range(self.model.latent_variables):
                    random_data_z = random_data.clone()
                    random_data_z[:, ~item_z_postive[:, z]] = selected_correct[~item_z_postive[:, z]].float() - 1
                    logger.info("Approximating minimum bit score z from random guessing data for latent variable %s.", z+1)
                    guessing_z[:, z] = self.latent_scores(random_data_z, scale="z", z_estimation_method=z_estimation_method, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate)[:, z]

            starting_z = guessing_z.detach().median(dim=0).values.reshape(1, self.model.latent_variables)

        return starting_z


    @torch.inference_mode()
    def bit_scores_from_z(
        self,
        z: torch.Tensor,
        start_z: torch.Tensor = None,
        population_z: torch.Tensor = None,
        one_dimensional: bool = False,
        z_estimation_method: str = "ML",
        ml_map_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lbfgs_learning_rate: float = 0.3,
        grid_points: int = 300,
        items: list[int] = None,
        start_z_guessing_probabilities: list[float] = None,
        start_z_guessing_iterations: int = 10000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the bit scores from z scores.

        Parameters
        ----------
        z : torch.Tensor
            A 2D tensor. Columns are latent variables and rows are respondents.
        start_z : torch.Tensor, optional
            A one row 2D tensor with bit score starting values of each latent variable. Estimated automatically if not provided. (default is None)
        population_z : torch.Tensor, optional
            A 2D tensor with z scores of the population. Used to estimate relationships between each z and sum scores. Columns are latent variables and rows are respondents. (default is None and uses z_estimation_method with the model training data)
        one_dimensional: bool, optional
            Whether to estimate one combined bit score for a multidimensional model. (default is True)
        z_estimation_method : str, optional
            Method used to obtain the z score grid for bit score computation. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'ML')
        ml_map_device: str, optional
            For ML and MAP. The device to use for computation. Can be 'cpu' or 'cuda'. (default is "cuda" if available else "cpu")
        lbfgs_learning_rate: float, optional
            For ML and MAP. The learning rate to use for the LBFGS optimizer. (default is 0.3)
        grid_points : int, optional
            The number of points to use for computing bit score. More steps lead to more accurate results. (default is 300)
        items: list[int], optional
            The item indices for the items to use to compute the bit scores. (default is None and uses all items)
        start_z_guessing_probabilities: list[float], optional
            The guessing probability for each item if start_z is None. The same length as the number of items. Guessing is not supported for polytomously scored items and the probabilities for them will be ignored. (default is None and uses no guessing or, for multiple choice models, 1 over the number of item categories)
        start_z_guessing_iterations: int, optional
            The number of iterations to use for approximating a minimum z when guessing is incorporated. (default is 10000)
        
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A 2D tensor with bit score scale scores for each respondent across the rows together with another tensor with start_z.
        """
        if grid_points <= 0:
            raise ValueError("steps must be a positive integer")
        if start_z is not None and start_z.shape != (1, self.model.latent_variables):
            raise ValueError(f"start_z must be a one row tensor with shape (1, {self.model.latent_variables}).")
        if z_estimation_method not in ["NN", "ML", "EAP", "MAP"]:
            raise ValueError("Invalid bit_score_z_grid_method. Choose either 'NN', 'ML', 'EAP' or 'MAP'.")
        if items is None:
            items = list(range(len(self.model.modeled_item_responses)))
        elif not isinstance(items, list) or not all(isinstance(item, int) for item in items):
            raise ValueError("items must be a list of integers.")
        
        if population_z is None:
            if z_estimation_method != "NN":
                logger.info("Estimating population z scores needed for bit score computation.")
                population_z = self.latent_scores(self.algorithm.train_data, scale="z", z_estimation_method=z_estimation_method, ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate)
            else:
                logger.info("Using traning data z scores as population z scores for bit score computation.")
                population_z = self.algorithm.training_z_scores

        if start_z is None:
            start_z = self.get_bit_score_starting_z(
                z_estimation_method=z_estimation_method,
                items=items,
                start_all_incorrect=one_dimensional,
                train_z=population_z,
                guessing_probabilities=start_z_guessing_probabilities,
                guessing_iterations=start_z_guessing_iterations,
            )

        inverted_scales = self._inverted_scales(population_z)
        z_adjusted, train_z_adjusted, start_z_adjusted = self._anti_invert_and_adjust_z_scores(z, population_z, start_z, inverted_scales)
        
        grid_start, grid_end, _ = self._get_grid_boundaries(train_z_adjusted, start_z_adjusted)
        
        if one_dimensional:
            bit_scores = self._compute_1d_bit_scores(
                z_adjusted,
                start_z_adjusted,
                grid_start,
                grid_end,
                inverted_scales,
                grid_points
            )
        else:
            bit_scores = self._compute_multi_dimensional_bit_scores(
                z_adjusted,
                start_z_adjusted,
                train_z_adjusted,
                grid_start,
                grid_end,
                inverted_scales,
                grid_points
            )
        
        return bit_scores, start_z
    
    @torch.inference_mode(False)
    def bit_score_gradients(
        self,
        z: torch.Tensor,
        h: float = None,
        independent_z: int = None,
        start_z: torch.Tensor = None,
        population_z: torch.Tensor = None,
        one_dimensional: bool = False,
        z_estimation_method: str = "ML",
        ml_map_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lbfgs_learning_rate: float = 0.3,
        grid_points: int = 300,
        items: list[int] = None,
        start_z_guessing_probabilities: list[float] = None,
        start_z_guessing_iterations: int = 10000,
    ) -> torch.Tensor:
        """
        Computes the gradients of the bit scores with respect to the input z scores using the central difference method: 
        .. math ::

            f^{\\prime}(z) \\approx \\frac{f(z+h)-f(z-h)}{2 h}

        Parameters
        ----------
        z : torch.Tensor
            A 2D tensor containing latent variable z scores. Each column represents one latent variable.
        h : float, optional
            The step size for the central difference method. (default is uses the difference between the smaller and upper outlier limits (computed using the interquantile range rule) of the training z scores divided by 1000)
        independent_z : int, optional
            The latent variable to differentiate with respect to. (default is None and computes gradients with respect to z)
        start_z : torch.Tensor, optional
            A one row 2D tensor with bit score starting values of each latent variable. Estimated automatically if not provided. (default is None)
        population_z : torch.Tensor, optional
            A 2D tensor with z scores of the population. Used to estimate relationships between each z and sum scores. Columns are latent variables and rows are respondents. (default is None and uses z_estimation_method with the model training data)
        one_dimensional: bool, optional
            Whether to estimate one combined bit score for a multidimensional model. (default is True)
        z_estimation_method : str, optional
            Method used to obtain the z score grid for bit score computation. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'ML')
        ml_map_device: str, optional
            For ML and MAP. The device to use for computation. Can be 'cpu' or 'cuda'. (default is "cuda" if available else "cpu")
        lbfgs_learning_rate: float, optional
            For ML and MAP. The learning rate to use for the LBFGS optimizer. (default is 0.3)
        grid_points : int, optional
            The number of points to use for computing bit score. More steps lead to more accurate results. (default is 300)
        items: list[int], optional
            The item indices for the items to use to compute the bit scores. (default is None and uses all items)
        start_z_guessing_probabilities: list[float], optional
            The guessing probability for each item if start_z is None. The same length as the number of items. Guessing is not supported for polytomously scored items and the probabilities for them will be ignored. (default is None and uses no guessing or, for multiple choice models, 1 over the number of item categories)
        start_z_guessing_iterations: int, optional
            The number of iterations to use for approximating a minimum z when guessing is incorporated. (default is 10000)

        Returns
        -------
        torch.Tensor
            A torch tensor with the gradients for each z score. Dimensions are (z rows, bit scores, z scores) where the last two dimensions represent the jacobian.
            If independent_z is provided, the tensor has dimensions (z rows, bit scores).
        """
        if start_z is None:
            start_z = self.get_bit_score_starting_z(
                z_estimation_method=z_estimation_method,
                items=items,
                start_all_incorrect=one_dimensional,
                train_z=population_z,
                guessing_probabilities=start_z_guessing_probabilities,
                guessing_iterations=start_z_guessing_iterations,
            )

        q1_q3 = torch.quantile(self.algorithm.training_z_scores, torch.tensor([0.25, 0.75]), dim=0)
        iqr = q1_q3[1] - q1_q3[0]
        lower_bound = q1_q3[0] - 1.5 * iqr
        upper_bound = q1_q3[1] + 1.5 * iqr
        h = (upper_bound - lower_bound) / 1000
        z_low = z-h
        z_high = z+h
        if independent_z is None:
            gradients = torch.zeros(z.shape[0], z.shape[1], z.shape[1])
            for latent_variable in range(z.shape[1]):
                z_low_var = torch.cat((z[:, :latent_variable], z_low[:, latent_variable].view(-1, 1), z[:, latent_variable+1:]), dim=1)
                z_high_var = torch.cat((z[:, :latent_variable], z_high[:, latent_variable].view(-1, 1), z[:, latent_variable+1:]), dim=1)
                bit_scores_low = self.bit_scores_from_z(
                    z_low_var, start_z = start_z, population_z=population_z, one_dimensional=one_dimensional, z_estimation_method=z_estimation_method,
                    ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate, grid_points=grid_points, items=items
                )[0]
                bit_scores_high = self.bit_scores_from_z(
                    z_high_var, start_z = start_z, population_z=population_z, one_dimensional=one_dimensional, z_estimation_method=z_estimation_method,
                    ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate, grid_points=grid_points, items=items
                )[0]
                gradients[:, latent_variable, :] = (bit_scores_high - bit_scores_low) / (2 * h[latent_variable])
        else:
            z_low_var = torch.cat((z[:, :independent_z-1], z_low[:, independent_z-1].view(-1, 1), z[:, independent_z:]), dim=1)
            z_high_var = torch.cat((z[:, :independent_z-1], z_high[:, independent_z-1].view(-1, 1), z[:, independent_z:]), dim=1)
            bit_scores_low = self.bit_scores_from_z(
                z_low_var, start_z = start_z, population_z=population_z, one_dimensional=one_dimensional, z_estimation_method=z_estimation_method,
                ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate, grid_points=grid_points, items=items
            )[0]
            bit_scores_high = self.bit_scores_from_z(
                z_high_var, start_z = start_z, population_z=population_z, one_dimensional=one_dimensional, z_estimation_method=z_estimation_method,
                ml_map_device=ml_map_device, lbfgs_learning_rate=lbfgs_learning_rate, grid_points=grid_points, items=items
            )[0]
            gradients = (bit_scores_high - bit_scores_low) / (2 * h[independent_z-1])

        return gradients

        
    @torch.inference_mode(False)
    def expected_item_score_slopes(
        self,
        z: torch.Tensor,
        scale: str = 'z',
        bit_scores: torch.Tensor = None,
        rescale_by_item_score: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Computes the slope of the expected item scores, averaged over the sample in z. Similar to loadings in traditional factor analysis. For each separate latent variable, the slope is computed as the average of the slopes of the expected item scores for each item, using the median z scores for the other latent variables.

        Parameters
        ----------
        z : torch.Tensor, optional
            A 2D tensor with latent z scores from the population of interest. Each row represents one respondent, and each column represents a latent variable. If not provided, uses the training z scores. (default is None)
        scale : str, optional
            The latent trait scale to differentiate with respect to. Can be 'bit' or 'z'. 
            'bit' is only a linear approximation for multidimensional models since multiple z scores can lead to the same bit scores, 
            and thus there are no unique derivatives of the item scores with respect to the bit scores for multidimensional models. (default is 'z')
        bit_scores: torch.Tensor, optional
            A 2D tensor with bit scores corresponding to the z scores. If not provided, computes the bit scores from the z scores. (default is None)
        rescale_by_item_score : bool, optional
            Whether to rescale the expected items scores to have a max of one by dividing by the max item score. (default is True)
        **kwargs
            Additional keyword arguments for the bit_score_gradients method.

        Returns
        -------
        torch.Tensor
            A tensor with the expected item score slopes.
        """
        if z.shape[0] < 2:
            raise ValueError("z must have at least 2 rows.")
        if z.requires_grad:
            z.requires_grad_(False)

        if scale == 'bit' and self.model.latent_variables > 1:
            expected_item_sum_scores = self.model.expected_item_sum_score(z, return_item_scores=True).detach()
            if not self.model.mc_correct and rescale_by_item_score:
                expected_item_sum_scores = expected_item_sum_scores / (torch.tensor(self.model.modeled_item_responses) - 1)
            if bit_scores is None:
                bit_scores = self.bit_scores_from_z(z)[0]
            # item score slopes for each item
            mean_slopes = linear_regression(bit_scores, expected_item_sum_scores).t()[:, 1:]
        else:
            median, _ = torch.median(z, dim=0)
            mean_slopes = torch.zeros(z.shape[0], len(self.model.modeled_item_responses), z.shape[1])
            for latent_variable in range(z.shape[1]):
                z_scores = median.repeat(z.shape[0], 1)
                z_scores[:, latent_variable], _ = z[:, latent_variable].sort()
                z_scores.requires_grad_(True)
                expected_item_sum_scores = self.model.expected_item_sum_score(z_scores, return_item_scores=True)
                if not self.model.mc_correct and rescale_by_item_score:
                    expected_item_sum_scores = expected_item_sum_scores / (torch.tensor(self.model.modeled_item_responses) - 1)

                # item score slopes for each item
                for item in range(expected_item_sum_scores.shape[1]):
                    if z_scores.grad is not None:
                        z_scores.grad.zero_()
                    expected_item_sum_scores[:, item].sum().backward(retain_graph=True)
                    mean_slopes[:, item, latent_variable] = z_scores.grad[:, latent_variable]

            if scale == 'bit' and self.model.latent_variables == 1:
                dbit_dz = self.bit_score_gradients(z, independent_z=1, **kwargs)
                # divide by the derivative of the bit scores with respect to the z scores
                # to get the expected item score slopes with respect to the bit scores
                mean_slopes = torch.einsum('ab...,a...->ab...', mean_slopes, 1/dbit_dz)
            else:
                mean_slopes = mean_slopes.mean(dim=0)

        return mean_slopes

    def _inverted_scales(self, train_z):
        """
        Compute a tensor with information about whether each latent variable is positively or negatively related to the test scores.

        Parameters
        ----------
        train_z : torch.Tensor
            The training data in the latent space.

        Returns
        -------
        torch.Tensor
            A one row tensor with elements corresponding to latent variables. 1's for positive and -1's for negative relationships.

        Notes
        -----
        If the neural network is set to handle multiple choice correct answers, 
        the method computes the scores based on the correct responses. 
        Otherwise, it simply sums up the training data. 
        The method then performs a linear regression between the latent variables 
        and the scores, and inverts the scales based on the linear weights.
        """
        scores = self.model.expected_item_sum_score(train_z).sum(dim=1).reshape(-1, 1)
        linear_weights = linear_regression(train_z, scores)[1:]
        inverted_scale = torch.where(linear_weights < 0, torch.tensor(-1), torch.tensor(1)).reshape(1, -1)
        return inverted_scale

    def _anti_invert_and_adjust_z_scores(self, z, train_z, start_z, inverted_scale):
        """
        Adjust the input and training z scores based on the starting z scores and the inverted scale.

        Parameters
        ----------
        z : torch.Tensor
            The input z scores.
        train_z : torch.Tensor
            The training z scores.
        start_z : torch.Tensor
            The starting z scores.
        inverted_scale : torch.Tensor
            The inverted scale.

        Returns
        -------
        tuple of torch.Tensor
            The adjusted input z scores, the adjusted training z scores and the adjusted starting z scores.

        Notes
        -----
        The method first anti-inverts the z scores by multiplying them with the inverted scale. 
        Then, If the anti-inverted z scores are smaller than the starting z scores, we set them to the starting z scores.
        """
        start_z_adjusted = start_z * inverted_scale
        z_adjusted = torch.max(z * inverted_scale, start_z_adjusted)
        train_z_adjusted = torch.max(train_z * inverted_scale, start_z_adjusted)

        return z_adjusted, train_z_adjusted, start_z_adjusted

    def _get_grid_boundaries(self, train_z_adjusted: torch.tensor, start_z_adjusted):
        """
        Determines the start and end points of the grid used for computing bit scores.

        Parameters
        ----------
        train_z_adjusted : torch.Tensor
            A 2D array containing the adjusted z scores of the training data. Each row represents one respondent, and each column represents a latent variable.
        start_z_adjusted : torch.Tensor
            A 1 row tensor containing starting z scores for the bit scores.

        Returns
        -------
        grid_start : torch.Tensor
            A 1D tensor containing the start points of the grid for each latent variable.
        grid_end : torch.Tensor
            A 1D tensor containing the end points of the grid for each latent variable.
        start_is_outlier : torch.Tensor
            A 1 row boolean tensor where each column corresponds to one latent variable. True if the starting z score is an outlier.
        """        
        outlier_detector = OutlierDetector(factor=4)
        start_is_outlier = outlier_detector.is_outlier(start_z_adjusted, data=train_z_adjusted, lower=True)[0, :]
        if any(start_is_outlier):
            smallest_non_outlier = outlier_detector.smallest_largest_non_outlier(train_z_adjusted, smallest=True)
            grid_start = torch.max(start_z_adjusted, smallest_non_outlier)
        else:
            grid_start = start_z_adjusted
        grid_end = outlier_detector.smallest_largest_non_outlier(train_z_adjusted, smallest=False)
        return grid_start, grid_end, start_is_outlier

    def _compute_1d_bit_scores(self, z_adjusted, start_z_adjusted, grid_start, grid_end, inverted_scale, grid_points):
        """
        Computes the 1D bit scores for the given input z scores.

        Parameters
        ----------
        z_adjusted : torch.Tensor
            The adjusted z-scores. A 2D tensor.
        start_z_adjusted : torch.Tensor
            The adjusted z-scores for the starting point. A one row tensor.
        grid_start : torch.Tensor
            The start of the grid. A one row tensor.
        grid_end : torch.Tensor
            The end of the grid. A one row tensor.
        inverted_scale : torch.Tensor
            The inverted scale. A one row tensor with -1 for latent variables inversely related to the test scores and 1 for others.
        grid_points : int
            The number of grid points.

        Returns
        -------
        torch.Tensor
            The computed 1D bit scores.
        """
        # Limit all z_scores to be within the grid range
        z_adjusted_capped = torch.clamp(z_adjusted, grid_start, grid_end)
        
        grid_linespace = torch.linspace(0, 1, steps=grid_points).unsqueeze(-1)
        # Use broadcasting to compute the grid
        grid = grid_start + (grid_linespace * (z_adjusted_capped - grid_start).unsqueeze(1))

        # set the entire grid for those below the grid to their z scores
        below_grid_start_mask = z_adjusted < grid_start
        grid = torch.where(below_grid_start_mask.unsqueeze(1), z_adjusted.unsqueeze(1), grid)

        # Ensure all values are larger than start_z_adjusted
        grid = torch.maximum(grid, start_z_adjusted)

        # set the last slot in each grid to the outliers z scores
        grid[:, -1, :] = torch.where(z_adjusted > grid_end, z_adjusted, grid[:, -1])
        # set the first slot in each grid to the start_z_adjusted (required when z_adjusted is an outlier)
        grid[:, 0, :] = start_z_adjusted.squeeze()

        # convert back to non-inverted z scale and compute grid entropies
        grid = grid.view(-1, grid.shape[2]) * inverted_scale
        output = self.model(grid)
        entropies = output_to_item_entropy(output, self.model.modeled_item_responses)
        # the goal is to get the entropies to the dimensions required for bit_score_distance
        # we need to transpose for view to order them correctly
        # and change positioning of each dimension in the end using permute
        entropies = (
            entropies.t()
            .view(entropies.shape[1], int(entropies.shape[0] / grid_points), grid_points)
            .permute(1, 0, 2)
        )

        # Compute the absolute difference
        diff = torch.abs(entropies - torch.roll(entropies, shifts=1, dims=2))
        # Note that for each sub-tensor in the third dimension, the first element will
        # be the difference with the last element of the previous sub-tensor because
        # of the roll function. We set this to 0
        diff[:, :, 0] = 0
        return diff.sum(dim=(1, 2)).unsqueeze(1)

    def _compute_multi_dimensional_bit_scores(self, z_adjusted, start_z_adjusted, train_z_adjusted, grid_start, grid_end, inverted_scale, grid_points):
        """
        Computes the multi-dimensional bit scores for the given input z scores.

        Parameters
        -----------
        z_adjusted : torch.Tensor
            The input z scores.
        start_z_adjusted : torch.Tensor
            The minimum z score to be used in the grid. A one row tensor.
        train_z_adjusted : torch.Tensor
            The z scores of the training data. Used for computing the median of each latent variable.
        grid_start : float
            The minimum value of the grid. A one row tensor.
        grid_end : float
            The maximum value of the grid. A one row tensor.
        inverted_scale : torch.Tensor
            The inverted scale. A one row tensor with -1 for latent variables inversely related to the test scores and 1 for others.
        grid_points : int
            The number of points in the grid.

        Returns
        --------
        bit_scores : torch.Tensor
            The multi-dimensional bit scores for the input z scores.
        """
        # Construct grid
        ratio = torch.linspace(0, 1, steps=grid_points).view(-1, 1)
        grid = (1 - ratio) * grid_start + ratio * grid_end
        # Add z scores to the grid and sort the columns
        grid = torch.cat([grid, z_adjusted], dim=0)
        # set all z scores smaller than start_z_adjusted to start_z_adjusted
        grid = torch.max(grid, start_z_adjusted)
        # set the first slot in each grid to the start_z_adjusted
        # required when any value in z_adjusted is an outlier
        grid[0, :] = start_z_adjusted

        grid, sorted_indices = torch.sort(grid, dim=0)
        # Fill each column in grid we are not not computing bit scores for with the median
        median, _ = torch.median(train_z_adjusted, dim=0)
        bit_scores = torch.zeros_like(z_adjusted)
        for z_var in range(grid.shape[1]):
            # Only compute once per unique value
            unique_grid, inverse_indices = grid[:, z_var].unique(return_inverse=True)
            latent_variable_grid = median.repeat(unique_grid.shape[0], 1)
            latent_variable_grid[:, z_var] = unique_grid

            # Convert back to non-inverted z scale and compute grid entropies
            output = self.model(latent_variable_grid * inverted_scale)
            entropies = output_to_item_entropy(output, self.model.modeled_item_responses)

            # Compute the absolute difference between each grid point entropy and the previous one
            diff = torch.zeros_like(entropies)
            diff[1:,:] = torch.abs(entropies[:-1, :] - entropies[1:, :])

            # cummulative sum over grid points and then sum the item scores
            bit_score_grid = diff.sum(dim=1).cumsum(dim=0)

            # add duplicates, unsort and take only the bit scores for the input z scores
            bit_scores[:, z_var] = bit_score_grid[inverse_indices][torch.sort(sorted_indices[:, z_var])[1]][-z_adjusted.shape[0]:]
            
        return bit_scores
