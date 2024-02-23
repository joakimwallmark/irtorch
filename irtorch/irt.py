import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from plotly import graph_objects as go
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from irtorch.models import BaseIRTModel
from irtorch.models import Parametric
from irtorch.models import NonparametricMonotoneNN
from irtorch.estimation_algorithms import AEIRT, VAEIRT
from irtorch.irt_scorer import IRTScorer
from irtorch.irt_plotter import IRTPlotter
from irtorch.irt_evaluator import IRTEvaluator
from irtorch.estimation_algorithms.encoders import BaseEncoder

logger = logging.getLogger('irtorch')

class IRT:
    """
    Main item response theory (IRT) class. 
    Consists of an IRT model instance (inheriting BaseIRTModel) which is fitted using the specified estimation algorithm.

    Parameters
    ----------
    model : str | BaseIRTModel, optional
        The IRT model to use. Available models are:

        - "1PL": One-parameter logistic model.
        - "2PL": Two-parameter logistic model.
        - "GPC": Generalized partial credit model.
        - "nominal": Nominal response model.
        - "MNN": Monotone neural network model.
        - "MMCNN": Monotone multiple choice neural network model.
        
        Default is None and uses either MNN or MMCNN depending on whether mc_correct is provided or not. 
        An instantiated model can also be provided.
    estimation_algorithm : str, optional
        The estimation algorithm to use. Available options are

        - "AE" for autoencoder. This is the default.
        - "VAE" for variational autoencoder.

    latent_variables : int, optional
        The number of latent variables to use for the model. (default is 1)
    data: torch.Tensor, optional
        A 2D torch tensor with test data. Used to automatically compute item_categories. Columns are items and rows are respondents. (default is None)
    item_categories : list[int], optional
        A list of integers where each integer is the number of possible responses for the corresponding item, exluding missing values. Overrides the data argument. (default is None)
    item_z_relationships: torch.Tensor, optional
        A tensor of shape (latent_variables, items) that defines the relationship between latent variables and item categories. If not provided, assumes relationships between all items and latent variables. (default is None)
    model_missing : bool, optional
        Whether missing values should be modeled as their own category. Ignored if an instantiated model is supplied. (default is False)
    mc_correct : list[int], optional
        List of correct answers for multiple choice questions. If provided also sets one_hot_encoded to True. (default is None)
    nominal_reference_category : bool, optional
        Whether to use a reference category for nominal models. If True, removes the model parameters for one response category per item. (default is False)

    encoder : BaseEncoder, optional
        The encoder to use for the AE or VAE. Overrides the one_hot_encoded, hidden_layers_encoder, nonlinear_encoder and batch_normalization_encoder arguments.
        If not provided, creates an instance of class StandardEncoder or VariationalEncoder for AE and VAE respectively. (default is None)
    one_hot_encoded : bool, optional
        Whether the model fitting algorithm uses one-hot encoded data. (default is False for all models except for MMC)
    hidden_layers_encoder : list[int], optional
        List of hidden layers for the encoder. Each element is a layer with the number of neurons represented as integers. If not provided, uses one hidden layer with 2 * sum(item_categories) neurons.
    nonlinear_encoder : torch.nn.Module, optional
        The non-linear function to use after each hidden layer in the encoder. (default is torch.nn.ELU())
    batch_normalization_encoder : bool, optional
        Whether to use batch normalization for the encoder. (default is True)
    """
    def __init__(
        self,
        model: str | BaseIRTModel = None,
        estimation_algorithm: str = "AE",
        latent_variables: int = 1,
        data: torch.Tensor = None,
        item_categories: list[int] = None,
        item_z_relationships: torch.Tensor = None,
        model_missing: bool = False,
        mc_correct: list[int] = None,
        nominal_reference_category: bool = False,
        encoder: BaseEncoder = None,
        one_hot_encoded: bool = False,
        hidden_layers_decoder: list[int] = None,
        hidden_layers_encoder: list[int] = None,
        nonlinear_encoder = torch.nn.ELU(),
        batch_normalization_encoder: bool = True,
        summary_writer: SummaryWriter = None,
    ):
        if isinstance(model, BaseIRTModel):
            self.latent_variables = model.latent_variables
            self.model = model
        else:
            if item_categories is None:
                if data is None:
                    raise ValueError("Either an instantiated model, item_categories or data must be provided to initialize the model.")
                else:
                    # replace nan with -inf to get max
                    item_categories = (torch.where(~data.isnan(), data, torch.tensor(float('-inf'))).max(dim=0).values + 1).int().tolist()

            if model in ["1PL", "2PL", "3PL", "GPC", "nominal"]:
                self.model = Parametric(
                    latent_variables=latent_variables,
                    item_categories=item_categories,
                    model=model,
                    model_missing=model_missing,
                    mc_correct=mc_correct,
                    item_z_relationships=item_z_relationships,
                    reference_category=nominal_reference_category,
                )
            elif model in ["MMCNN", "MNN"] or model is None:
                if hidden_layers_decoder is None:  # 1 layer with 2x number of categories as neurons is default
                    hidden_layers_decoder = [3]

                self.model = NonparametricMonotoneNN(
                    latent_variables=latent_variables,
                    item_categories=item_categories,
                    hidden_dim=hidden_layers_decoder,
                    model_missing=model_missing,
                    mc_correct = mc_correct,
                    item_z_relationships=item_z_relationships,
                    use_bounded_activation=True,
                )

        if estimation_algorithm == "AE":
            self.algorithm = AEIRT(
                model=self.model,
                encoder=encoder,
                one_hot_encoded=one_hot_encoded,
                hidden_layers_encoder=hidden_layers_encoder,
                nonlinear_encoder=nonlinear_encoder,
                batch_normalization_encoder=batch_normalization_encoder,
                summary_writer = summary_writer
            )
        elif estimation_algorithm == "VAE":
            self.algorithm = VAEIRT(
                model=self.model,
                encoder=encoder,
                one_hot_encoded=one_hot_encoded,
                hidden_layers_encoder=hidden_layers_encoder,
                nonlinear_encoder=nonlinear_encoder,
                batch_normalization_encoder=batch_normalization_encoder,
                summary_writer = summary_writer
            )

        self.scorer = IRTScorer(self.model, self.algorithm)
        self.evaluator = IRTEvaluator(self.model, self.algorithm, self.scorer)
        self.plotter = IRTPlotter(self.model, self.algorithm, self.scorer, self.evaluator)

    def fit(
        self,
        train_data: torch.Tensor,
        **kwargs
    ) -> None:
        """
        Train the autoencoder model.

        Parameters
        ----------
        train_data : torch.Tensor
            The training data. Item responses should be coded 0, 1, ... and missing responses coded as nan or -1.
        **kwargs
            Additional keyword arguments to pass to the fit method of the estimation algorithm. For details, see fit method documentation of the estimation algorithm. 
            Currently supported algorithms are AEIRT and VAEIRT.
        """
        self.algorithm.fit(
            train_data=train_data,
            **kwargs
        )

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
        bit_score_start_z: torch.tensor = None,
        bit_score_start_z_guessing_probabilities: list[float] = None,
        bit_score_start_z_guessing_iterations: int = 10000,
        bit_score_items: list[int] = None
    ) -> torch.Tensor:
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
            For computing bit scores. A 2D tensor containing the pre-estimated z scores for each respondent in the data. If not provided, will be estimated using z_estimation_method. Each row corresponds to one respondent and each column represents a latent variable. (default is None)
        z_estimation_method : str, optional
            Method used to obtain the z scores. Also used for bit scores as they require the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'ML')
        ml_map_device : str, optional
            For ML and MAP. The device to use for the LBFGS optimizer. (default is "cuda" if available else "cpu")
        lbfgs_learning_rate: float, optional
            For ML and MAP. The learning rate to use for the LBFGS optimizer. (default is 0.3)
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
        return self.scorer.latent_scores(
            data,
            scale,
            z,
            z_estimation_method,
            ml_map_device,
            lbfgs_learning_rate,
            eap_z_integration_points,
            bit_score_one_dimensional,
            bit_score_population_z,
            bit_score_grid_points,
            bit_score_z_grid_method,
            bit_score_start_z,
            bit_score_start_z_guessing_probabilities,
            bit_score_start_z_guessing_iterations,
            bit_score_items,
        )

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
        -----------
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
        return self.scorer.bit_scores_from_z(
            z=z,
            start_z=start_z,
            population_z=population_z,
            one_dimensional=one_dimensional,
            z_estimation_method=z_estimation_method,
            ml_map_device=ml_map_device,
            lbfgs_learning_rate=lbfgs_learning_rate,
            grid_points=grid_points,
            items=items,
            start_z_guessing_probabilities=start_z_guessing_probabilities,
            start_z_guessing_iterations=start_z_guessing_iterations,
        )

    def sum_score_probabilities(
        self,
        latent_density_method: str = "data",
        population_data: torch.Tensor = None,
        trapezoidal_segments: int = 1000,
        sample_size: int = 100000,
    ) -> torch.Tensor:
        """
        Computes the marginal probabilities for each sum score, averged over the latent space density. For 'qmvn' and 'gmm' densities, the trapezoidal rule is used for integral approximation.

        Parameters
        ----------
        latent_density_method : str, optional
            Specifies the method used to approximate the latent space density.
            Possible options are
            - 'data' averages over the z scores from the population data.
            - 'encoder sampling' samples z scores from the encoder. Only available for VariationalAutoencoderIRT models
            - 'qmvn' for quantile multivariate normal approximation of a multivariate joint density function (QuantileMVNormal class).
            - 'gmm' for an sklearn gaussian mixture model.

        population_data : torch.Tensor, optional
            The population data used for approximating sum score probabilities. Default is None and uses the training data.
        trapezoidal_segments : int, optional
            The number of integration approximation intervals for each z dimension. (Default is 1000)
        sample_size : int, optional
            Sample size for the 'encoder sampling' method. (Default is 100000)

        Returns
        -------
        torch.Tensor
            A 1D tensor with the probability for each total score.
        """
        return self.evaluator.sum_score_probabilities(
            latent_density_method, population_data, trapezoidal_segments, sample_size
        )

    def expected_item_sum_score(self, z: torch.Tensor, return_item_scores: bool = True) -> torch.Tensor:
        """
        Computes the model expected item scores/sum scores for each respondent.

        Parameters
        ----------
        z : torch.Tensor
            A 2D tensor with z scores. Each row represents one respondent, and each column represents a latent variable.
        return_item_scores : bool, optional
            Whether to return the expected item scores. If False, returns the expected sum scores (default is True)

        Returns
        -------
        torch.Tensor
            A tensor with the expected scores for each respondent.
        """
        return self.model.expected_item_sum_score(z, return_item_scores)
    
    def expected_item_score_slopes(
        self,
        z: torch.Tensor = None,
        bit_scores: torch.Tensor = None,
        rescale_by_item_score: bool = True,
    ):
        """
        Computes the slope of the expected item scores, averaged over the sample in z. Similar to loadings in traditional factor analysis. For each separate latent variable, the slope is computed as the average of the slopes of the expected item scores for each item, using the median z scores for the other latent variables.

        Parameters
        ----------
        z : torch.Tensor, optional
            A 2D tensor with latent z scores from the population of interest. Each row represents one respondent, and each column represents a latent variable. If not provided, uses the training z scores. (default is None)
        bit_scores : torch.Tensor, optional
            A 2D tensor with bit scores corresponding to each z score in z. If provided, slopes will be computed on the bit scales. (default is None)
        rescale_by_item_score : bool, optional
            Whether to rescale the expected items scores to have a max of one by dividing by the max item score. (default is True)

        Returns
        -------
        torch.Tensor
            A tensor with the expected item score slopes.
        """
        if z is None:
            z = self.algorithm.training_z_scores
        return self.model.expected_item_score_slopes(z, bit_scores, rescale_by_item_score)
        

    def accuracy(
        self,
        data: torch.Tensor = None,
        z: torch.Tensor = None,
        z_estimation_method: str = "ML",
        level: str = "all",
    ) -> torch.Tensor:
        """
        Calculate the prediction accuracy of the model for the supplied data.

        Parameters
        ----------
        data : torch.Tensor
            The input data.
        z: torch.Tensor, optional
            The latent variable z scores for the provided data. If not provided, they will be computed using z_estimation_method.
        z_estimation_method : str, optional
            Method used to obtain the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively.
        level: str = "all", optional
            Specifies the level at which the accuracy is calculated. Can be 'all', 'item' or 'respondent'. For example, for 'item' the accuracy is calculated for each item. (default is 'all')

        Returns
        -------
        torch.Tensor
            The accuracy.
        """
        return self.evaluator.accuracy(data, z, z_estimation_method, level)
    
    def residuals(
        self,
        data: torch.Tensor = None,
        z: torch.Tensor = None,
        z_estimation_method: str = "ML",
        average_per: str = "none",
    ) -> torch.Tensor:
        """
        Calculate the residuals of the model for the supplied data.

        Parameters
        ----------
        data : torch.Tensor
            The input data.
        z: torch.Tensor, optional
            The latent variable z scores for the provided data. If not provided, they will be computed using z_estimation_method.
        z_estimation_method : str, optional
            Method used to obtain the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively.
        average_per: str = "none", optional
            Whether to average the residuals and over which level. Can be 'all', 'item' or 'respondent'. Use 'none' for no average. For example, with 'item' the average residuals is calculated for each item. (default is 'none')
            
        Returns
        -------
        torch.Tensor
            The residuals.
        """
        return self.evaluator.residuals(data, z, z_estimation_method, average_per)
    
    def log_likelihood(
        self,
        data: torch.Tensor = None,
        z: torch.Tensor = None,
        z_estimation_method: str = "ML",
        reduction: str = "sum",
        level: str = "all",
    ) -> torch.Tensor:
        """
        Calculate the log-likelihood for the provided data.

        If 'data' is not supplied, the function defaults to using the model's training data.

        Parameters
        ----------
        data : torch.Tensor, optional
            A 2D tensor containing test data. Each row corresponds to one respondent and each column represents a latent variable. (default is None)
        z : torch.Tensor, optional
            A 2D tensor containing latent variable z scores. Each row corresponds to one respondent and each column represents a latent variable. (default is None)
        z_estimation_method : str, optional
            Method used to obtain the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'NN')
        reduction : str, optional
            Specifies the reduction method for the log-likelihood. Can be 'sum', 'none' or 'mean'. (default is 'sum')
        level : str, optional
            For reductions other than 'none', specifies the level at which the log-likelihood is summed/averaged. Can be 'all', 'item' or 'respondent'. For example, for 'item' the log-likelihood is summed/averaged for each item. (default is 'all')
            
        Returns
        -------
        torch.Tensor
            The log-likelihood for the provided data.
        """
        return self.evaluator.log_likelihood(data, z, z_estimation_method, reduction, level)

    def group_fit_log_likelihood(
        self,
        groups: int = 10,
        latent_variable: int = 1,
        data: torch.Tensor = None,
        z: torch.Tensor = None,
        z_estimation_method: str = "ML",
    ):
        """
        Group the respondents based on their ordered latent variable scores.
        Calculate the log-likelihood of the data within each group.

        If 'data' is not supplied, the function defaults to using the model's training data.

        Parameters
        ----------
        groups: int
            The number of groups. (default is 10)
        latent_variable: int, optional
            Specifies the latent variable along which ordering and grouping should be performed. (default is 1)
        data : torch.Tensor, optional
            A 2D tensor containing test data. Each row corresponds to one respondent and each column represents a latent variable. (default is None)
        z : torch.Tensor, optional
            A 2D tensor containing the pre-estimated z scores for each person in the data. Each row corresponds to one respondent and each column represents a latent variable. (default is None)
        z_estimation_method : str, optional
            Method used to obtain the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'NN')

        Returns
        -------
        torch.Tensor
            The average log-likelihood for each group.
        """
        return self.evaluator.group_fit_log_likelihood(data, z, z_estimation_method, groups, latent_variable)
    
    def group_fit_residuals(
        self,
        data: torch.Tensor = None,
        z: torch.Tensor = None,
        z_estimation_method: str = "ML",
        standardize: bool = True,
        groups: int = 10,
        latent_variable: int = 1,
        scale: str = "z",
        bit_score_start_z: torch.tensor = None,
        population_z: torch.Tensor = None,
        bit_score_grid_points: int = 300,
        bit_score_z_grid_method: int = "ML",
        bit_score_start_z_guessing_probabilities: list[float] = None,
        bit_score_start_z_guessing_iterations: int = 10000,
        bit_score_items: list[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Group the respondents based on their ordered latent variable scores.
        Calculate the residuals between the model estimated and observed data within each group.
        See ch. 20 in Handbook of Item Response Theory, Volume Two: Statistical Tools for more details.

        If 'data' is not supplied, the function defaults to using the model's training data.

        Parameters
        ----------
        data : torch.Tensor, optional
            A 2D tensor containing test data. Each row corresponds to one respondent and each column represents a latent variable. (default is None)
        z : torch.Tensor, optional
            A 2D tensor containing the pre-estimated z scores for each respondent in the data. Each row corresponds to one respondent and each column represents a latent variable. (default is None)
        z_estimation_method : str, optional
            Method used to obtain the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'NN')
        standardize : bool, optional
            Specifies whether the residuals should be standardized. (default is True)
        groups: int
            The number of groups. (default is 10)
        latent_variable: int, optional
            Specifies the latent variable along which ordering and grouping should be performed. (default is 1)
        scale : str, optional
            The grouping method scale, which can either be 'bit' or 'z'. Note: for uni-dimensional
            models, 'z' and 'bit' are equivalent. (default is 'z')
        bit_score_start_z : int, optional
            The z score used as the starting point for bit score computation. Computed automatically if not provided. (default is 'None')
        population_z : torch.Tensor, optional
            Only for bit scores. The latent variable z scores for the population. If not provided, they will be computed using z_estimation_method with the model training data. (default is None)
        bit_score_grid_points : int, optional
            The number of points to use for computing bit score. More steps lead to more accurate results. (default is 300)
        bit_score_z_grid_method : str, optional
            Method used to obtain the z score grid for bit score computation. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'ML')
        bit_score_start_z_guessing_probabilities: list[float], optional
            Custom guessing probabilities for each item. The same length as the number of items. Guessing is not supported for polytomously scored items and the probabilities for them will be ignored. (default is None and uses no guessing or, for multiple choice models, 1 over the number of item categories)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple with torch tensors. The first one holds the residuals for each group and has dimensions (groups, items, item categories). The second one is a 1D tensor and holds the mid points of the groups.
        """
        return self.evaluator.group_fit_residuals(
            data = data,
            z = z,
            z_estimation_method = z_estimation_method,
            standardize = standardize,
            groups = groups,
            latent_variable = latent_variable,
            scale = scale,
            bit_score_start_z = bit_score_start_z,
            population_z = population_z,
            bit_score_grid_points = bit_score_grid_points,
            bit_score_z_grid_method = bit_score_z_grid_method,
            bit_score_start_z_guessing_probabilities = bit_score_start_z_guessing_probabilities,
            bit_score_start_z_guessing_iterations = bit_score_start_z_guessing_iterations,
            bit_score_items = bit_score_items,
        )

    def save_model(self, path: str) -> None:
        """
        Save the fitted model.

        Parameters
        -------
        path : str
            Where to save fitted model.
        """
        # TODO save training history
        if self.algorithm.train_data is None:
            logger.error("Attempted to save model before fitting.")
            raise AttributeError("Cannot save model before fitting.")
        
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "train_data": self.algorithm.train_data,
            "training_z_scores": self.algorithm.training_z_scores,
        }
        if isinstance(self.algorithm, AEIRT) or isinstance(self.algorithm, VAEIRT):
            to_save["encoder_state_dict"] = self.algorithm.encoder.state_dict()
        torch.save(to_save, path)

    def load_model(self, path: str) -> None:
        """
        Loads the model from a file. The initialized model should have the same structure and hyperparameter settings as the fitted model that is being loaded (e.g., the same number of latent variables).

        Parameters
        -------
        path : str
            Where to load fitted model from.
        """
        # TODO load training history
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.algorithm.train_data = checkpoint["train_data"]
        self.algorithm.training_z_scores = checkpoint["training_z_scores"]
        if isinstance(self.algorithm, AEIRT) or isinstance(self.algorithm, VAEIRT):
            self.algorithm.encoder.load_state_dict(checkpoint["encoder_state_dict"])

    def plot_training_history(self) -> tuple[Figure, Axes]:
        """
        Plots the training history of the model.

        Returns
        -------
        tuple[Figure, Axes]
            The matplotlib Figure and Axes objects for the plot.
        """
        return self.plotter.plot_training_history()

    def plot_latent_score_distribution(
        self,
        scores_to_plot: torch.Tensor = None,
        population_data: torch.Tensor = None,
        scale: str = "bit",
        latent_variables_to_plot: tuple[int] = (1,),
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        color: str = None,
        contour_colorscale: str = "Plasma",
        contour_plot_bins: int = None,
        **kwargs
    ) -> go.Figure:
        """
        Plots the distribution of latent scores.

        Parameters
        ----------
        scores_to_plot : torch.Tensor, optional
            If provided, the requested latent variable distributions are plotted directly.
            If None, scores are computed from the population data or the model training data. (default is None)
        population_data : torch.Tensor, optional
            The data used to compute the latent scores. If None, uses the training data. (default is None)
        scale : str, optional
            The scale to plot against. Can be 'bit' or 'z'. (default is 'bit')
        latent_variables_to_plot : tuple[int], optional
            The latent dimensions to include in the plot. (default is (1,))
        title : str, optional
            The title for the plot. (default is None)
        x_label : str, optional
            The label for the X-axis. (default is None and uses "Latent variable" for one latent variable and "Latent variable 1" for two latent variables)
        y_label : str, optional
            The label for the Y-axis. (default is None and uses "Density" for one latent variable and "Latent variable 2" for two latent variables)
        color : str, optional
            The color to use for plots with one latent variable. (default is None and uses the default color sequence for the plotly_white template)
        contour_colorscale : str, optional
            Sets the colorscale for the multiple latent variable contour plots. See https://plotly.com/python/builtin-colorscales/ (default is "Plasma")
        countor_plot_bins : int, optional
            The number of histogram bins to use for creating the contour plot. (default is None and uses Sturges’ Rule)
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the latent_scores method if scores_to_plot is not provided. See :meth:`latent_scores` for details.

        Returns
        -------
        go.Figure
            The Plotly Figure object for the plot.
        """
        return self.plotter.plot_latent_score_distribution(
            scores_to_plot=scores_to_plot,
            population_data=population_data,
            scale=scale,
            latent_variables_to_plot=latent_variables_to_plot,
            title=title,
            x_label=x_label,
            y_label=y_label,
            color=color,
            contour_colorscale=contour_colorscale,
            contour_plot_bins=contour_plot_bins,
            **kwargs
        )
    
    def plot_information(
        self,
        items: list[int] = None,
        scale: str = "bit",
        latent_variables: tuple[int] = (1,),
        degrees: list[int] = None,
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        color: str = None,
        colorscale: str = "Plasma",
        z_range: tuple[float, float] = None,
        second_z_range: tuple[float, float] = None,
        steps: int = None,
        fixed_zs: torch.Tensor = None,
        **kwargs
    ) -> go.Figure:
        """
        Plots the information function for the model.
        Supports both item and test information.

        Parameters
        ----------
        items : list[int], optional
            The items to plot. If None, the full test information is plotted. (default is None)
        scale : str, optional
            The scale to plot against. Can be 'bit' or 'z'. (default is 'bit')
        latent_variables : tuple[int], optional
            The latent space variables to plot. (default is (1,))
        degrees : list[int], optional
            A list of angles in degrees between 0 and 90. One degree for each latent variable.
            Only applicable when the model is multidimensional.
            Information will be computed in the direction of the angles. (default is None)
        title : str, optional
            The title for the plot. (default is None)
        x_label : str, optional
            The label for the X-axis. (default is None and uses "Latent variable" for one latent variable and "Latent variable 1" for two latent variables)
        y_label : str, optional
            The label for the Y-axis. (default is None and uses "Information" for one latent variable and "Latent variable 2" for two latent variables)
        color : str, optional
            The color to use for plots with one latent variable. (default is None and uses the default color sequence for the plotly_white template)
        colorscale : str, optional
            Sets the colorscale for the multiple latent variable surface plots. See https://plotly.com/python/builtin-colorscales/ (default is "Plasma")
        z_range : tuple[float, float], optional
            Only for scale = 'z'. The z range for plotting. (default is None and uses limits based on training data)
        second_z_range : tuple[float, float], optional
            Only for scale = 'z'. The range for plotting for the second latent variable. (default is None and uses limits based on training data)
        steps : int, optional
            The number of steps along each z axis to construct the latent variable grid for which information is evaluated at. (default is None and uses 100 for one latent variable and 18 for two latent variables)
        fixed_zs: torch.Tensor, optional
            Only for multdimensional models. Fixed values for latent space variable not plotted. (default is None and uses the medians in the training data)
        **kwargs : dict, optional
            Additional keyword arguments used for bit score computation. See :meth:`bit_scores_from_z` for details. 
        """
        return self.plotter.plot_information(
            items=items,
            scale=scale,
            latent_variables=latent_variables,
            degrees=degrees,
            title=title,
            x_label=x_label,
            y_label=y_label,
            color=color,
            colorscale=colorscale,
            z_range=z_range,
            second_z_range=second_z_range,
            steps=steps,
            fixed_zs=fixed_zs,
            **kwargs
        )

    @torch.inference_mode()
    def plot_item_probabilities(
        self,
        item: int,
        scale: str = "bit",
        latent_variables: tuple = (1, ),
        fixed_zs: torch.Tensor = None,
        steps: int = 1000,
        bit_score_start_z: torch.tensor = None,
        bit_score_grid_points: int = 300,
        bit_score_z_grid_method: int = "ML",
        bit_score_start_z_guessing_probabilities: list[float] = None,
        bit_score_start_z_guessing_iterations: int = 10000,
        bit_score_items: list[int] = None,
        z_range: tuple[float, float] = None,
        second_z_range: tuple[float, float] = None,
        plot_group_fit: bool = False,
        group_fit_groups: int = 10,
        group_fit_data: int = None,
        group_fit_population_z: torch.Tensor = None,
        group_z_estimation_method: str = "ML",
        plot_derivative: bool = False,
        grayscale: bool = False,
    ) -> tuple[Figure, Axes]:
        """
        Plots the item probability curves for a given item. Supports 2D and 3D plots.

        Parameters
        ----------
        item : int
            The item to plot.
        scale : str, optional
            The scale to plot against. Can be 'bit' or 'z'. (default is 'bit')
        latent_variables : tuple, optional
            The latent space variables to plot. (default is (1,))
        fixed_zs: torch.Tensor, optional
            Only for multdimensional models. Fixed values for latent space variable not plotted. (default is None and uses the medians in the training data)
        steps : int, optional
            The number of steps along each z axis used for probability evaluation. (default is 1000)
        bit_score_start_z : int, optional
            The z score used as the starting point for bit score computation. Computed automatically if not provided. (default is 'None')
        bit_score_grid_points : int, optional
            The number of points to use for computing bit score. More steps lead to more accurate results. (default is 300)
        bit_score_z_grid_method : str, optional
            Method used to obtain the z score grid for bit score computation. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'ML')
        bit_score_start_z_guessing_probabilities: list[float], optional
            Custom guessing probabilities for each item. The same length as the number of items. Guessing is not supported for polytomously scored items and the probabilities for them will be ignored. (default is None and uses no guessing or, for multiple choice models, 1 over the number of item categories)
        bit_score_start_z_guessing_iterations: int, optional
            The number of iterations to use for approximating a minimum z when guessing is incorporated. (default is 10000)
        bit_score_items: list[int], optional
            The item indices for the items to use to compute the bit scores. (default is 'None' and uses all items)
        z_range : tuple, optional
            Only for scale = 'z'. The z range for plotting. (default is None and uses limits based on training data)
        second_z_range : tuple, optional
            Only for scale = 'z'. The range for plotting for the second latent variable. (default is None and uses limits based on training data)
        plot_group_fit : bool, optional
            Plot group average probabilities to assess fit. (default is False)
        group_fit_groups : int, optional
            Only for plot_group_fit = True. The number of groups. (default is 10)
        group_fit_data: torch.tensor, optional
            Only for plot_group_fit = True. The data used for group fit plots. Uses training data if not provided. (default is None)
        group_fit_population_z : torch.tensor, optional
            Only for plot_group_fit = True. The z scores corresponding to group_fit_data. Will be estimated using group_z_estimation_method if not provided. (default is None)
        group_z_estimation_method : str, optional
            The method used for computing z-scores for the groups. Can be 'NN', 'ML', 'MAP' or 'EAP'. (default is 'ML')
        plot_derivative : bool, optional
            If true, plots the derivative of the item probabilitiy curves. Note that this feature is not yet implemented and will raise a TypeError if set to true. (default is False)
        grayscale : bool, optional
            Grayscale plot. (default is False)

        Returns
        -------
        tuple[Figure, Axes]
            The matplotlib Figure and Axes objects for the plot.
        """
        return self.plotter.plot_item_probabilities(
            item=item,
            scale=scale,
            latent_variables=latent_variables,
            fixed_zs=fixed_zs,
            steps=steps,
            bit_score_start_z=bit_score_start_z,
            bit_score_grid_points=bit_score_grid_points,
            bit_score_z_grid_method=bit_score_z_grid_method,
            bit_score_start_z_guessing_probabilities=bit_score_start_z_guessing_probabilities,
            bit_score_start_z_guessing_iterations=bit_score_start_z_guessing_iterations,
            bit_score_items=bit_score_items,
            z_range=z_range,
            second_z_range=second_z_range,
            plot_group_fit=plot_group_fit,
            group_fit_groups=group_fit_groups,
            group_fit_data=group_fit_data,
            group_fit_population_z=group_fit_population_z,
            group_z_estimation_method=group_z_estimation_method,
            plot_derivative=plot_derivative,
            grayscale=grayscale,
        )

    def plot_item_entropy(
        self,
        item: int,
        scale="bit",
        latent_variables: int = 1,
        steps: int = 1000,
        z_range: tuple[float, float] = (-4, 4),
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """
        Plot the entropy of item responses against latent variables.

        Parameters
        ----------
        item : int
            The item to plot.
        scale : str, optional
            The scale to plot against. Can be 'bit' or 'z'. (default is 'bit')
        latent_variables : int, optional
            The latent variable dimension to plot. (default is 1)
        steps : int, optional
            The number of steps along the latent variable scale used for entropy evaluation. (default is 1000)
        bit_score_method : str, optional
            The method used for scale value calculation. Only used if scale is 'bit'. (default is 'tanh')
        bit_score_grid_steps : int, optional
            The number of steps used to calculate bit scores. Only used if scale is 'bit'. (default is 300)
        z_range : tuple[float, float], optional
            The range for z-values for plotting. Only used if scale is 'z'. (default is (-4, 4))
        **kwargs
            Additional keyword arguments to pass to the bit score computation.

        Returns
        -------
        tuple[Figure, Axes]
            The matplotlib Figure and Axes objects for the plot.
        """
        return self.plotter.plot_item_entropy(
            item=item,
            scale=scale,
            latent_variables=latent_variables,
            steps=steps,
            z_range=z_range,
            **kwargs,
        )

    def plot_item_latent_variable_relationships(
        self,
        relationships: torch.Tensor,
        title: str = "Relationships: Items vs. latent variables",
        x_label: str = "Latent variables",
        y_label: str = "Items",
        cmap: str = "inferno",
    ) -> tuple[Figure, Axes]:
        """
        Create a heatmap of item-latent variable relationships.

        Parameters
        ----------
        relationships : torch.Tensor
            A tensor of item-latent variable relationships. Typically the returned tensor from expected_item_score_slopes() where each row represents an item and each column represents a latent variable.
        title : str, optional
            The title for the plot. (default is "Relationships: Items vs. latent variables")
        x_label : str, optional
            The label for the X-axis. (default is "Latent variables")
        y_label : str, optional
            The label for the Y-axis. (default is "Items")
        cmap : str, optional
            The matplotlib color map to use for the plot. Use for example "Greys" for black and white. (default is "inferno")

        Returns
        -------
        tuple[Figure, Axes]
            The matplotlib Figure and Axes objects for the plot.
        """
        return self.plotter.plot_item_latent_variable_relationships(
            relationships, title, x_label, y_label, cmap
        )