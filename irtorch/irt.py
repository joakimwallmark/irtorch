import logging
import torch
import pandas as pd
from plotly import graph_objects as go
from irtorch.models import BaseIRTModel, MonotoneNN, OneParameterLogistic, TwoParameterLogistic, GeneralizedPartialCredit, NominalResponse
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
        - "NR": Nominal response model.
        - "MNN": Monotone neural network model.
        - "MMCNN": Monotone multiple choice neural network model.
        
        Default is None and uses either MNN or MMCNN depending on whether mc_correct is provided or not. 
        An instantiated model inheriting :class:`irtorch.models.BaseIRTModel` can also be provided, see :doc:`irt_models` for model specific details.
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

            if model == "1PL":
                self.model = OneParameterLogistic(
                    latent_variables=latent_variables,
                    items=len(item_categories),
                    item_z_relationships=item_z_relationships
                )
            elif model == "2PL":
                self.model = TwoParameterLogistic(
                    latent_variables=latent_variables,
                    items=len(item_categories),
                    item_z_relationships=item_z_relationships
                )
            elif model == "GPC":
                self.model = GeneralizedPartialCredit(
                    latent_variables=latent_variables,
                    item_categories=item_categories,
                    item_z_relationships=item_z_relationships
                )
            elif model == "NR":
                self.model = NominalResponse(
                    latent_variables=latent_variables,
                    item_categories=item_categories,
                    item_z_relationships=item_z_relationships,
                    model_missing=model_missing,
                    mc_correct=mc_correct,
                    reference_category=nominal_reference_category
                )
            elif model in ["MMCNN", "MNN"] or model is None:
                if hidden_layers_decoder is None:  # 1 layer with 2x number of categories as neurons is default
                    hidden_layers_decoder = [3]

                self.model = MonotoneNN(
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
            )
        elif estimation_algorithm == "VAE":
            self.algorithm = VAEIRT(
                model=self.model,
                encoder=encoder,
                one_hot_encoded=one_hot_encoded,
                hidden_layers_encoder=hidden_layers_encoder,
                nonlinear_encoder=nonlinear_encoder,
                batch_normalization_encoder=batch_normalization_encoder,
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
            Additional keyword arguments to pass to the estimation algorithm. For details, see the documentation for each respective fit estimation algorithm for details.
            Currently supported algorithms are:

            - 'AEIRT': See :class:`irtorch.estimation_algorithms.aeirt.AEIRT`
            - 'VAEIRT': See :class:`irtorch.estimation_algorithms.vaeirt.VAEIRT`
        """
        self.algorithm.fit(
            train_data=train_data,
            **kwargs
        )

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
        return self.scorer.bit_score_gradients(
            z=z,
            h=h,
            independent_z=independent_z,
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

    def expected_scores(self, z: torch.Tensor, return_item_scores: bool = True) -> torch.Tensor:
        """
        Computes the model expected item scores/test scores for each respondent.

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
        with torch.no_grad():
            return self.model.expected_scores(z, return_item_scores).detach()
    
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
        if z is None:
            z = self.algorithm.training_z_scores
        return self.scorer.expected_item_score_slopes(z, scale, bit_scores, rescale_by_item_score, **kwargs)

    def bit_score_starting_z(
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
        Computes the starting z score from which to compute bit scores. 
        A z score at or below this starting z score will result in a bit score of zero when supplied to, i.e., :meth:`bit_scores_from_z` or :meth:`latent_scores`.
        
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
        return self.scorer.bit_score_starting_z(
            z_estimation_method=z_estimation_method,
            ml_map_device=ml_map_device,
            lbfgs_learning_rate=lbfgs_learning_rate,
            items=items,
            start_all_incorrect=start_all_incorrect,
            train_z=train_z,
            guessing_probabilities=guessing_probabilities,
            guessing_iterations=guessing_iterations,
        )

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
        scale: str = "z",
        latent_variable: int = 1,
        standardize: bool = True,
        groups: int = 10,
        z_estimation_method: str = "ML",
        population_z: torch.Tensor = None,
        **kwargs
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
        scale : str, optional
            The grouping method scale, which can either be 'bit' or 'z'. Note: for uni-dimensional
            models, 'z' and 'bit' are equivalent. (default is 'z')
        latent_variable: int, optional
            Specifies the latent variable along which ordering and grouping should be performed. (default is 1)
        standardize : bool, optional
            Specifies whether the residuals should be standardized. (default is True)
        groups: int
            The number of groups. (default is 10)
        z_estimation_method : str, optional
            Method used to obtain the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively. (default is 'NN')
        population_z : torch.Tensor, optional
            Only for bit scores. The latent variable z scores for the population. If not provided, they will be computed using z_estimation_method with the model training data. (default is None)
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the bit_scores_from_z method if scale is 'bit'. See :meth:`bit_scores_from_z` for details.
            
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
            population_z = population_z,
            **kwargs
        )

    def item_parameters(self, irt_format = False) -> pd.DataFrame:
        """
        For parametric models, get the item parameters for a fitted model.

        Parameters
        ----------
        irt_format : bool, optional
            Only for unidimensional parametric models. Whether to return the item parameters in traditional IRT format. Otherwise returns weights and biases. (default is False)

        Returns
        -------
        pd.DataFrame
            A dataframe with the item parameters.
        """
        if hasattr(self.model, 'item_parameters'):
            return self.model.item_parameters(irt_format)
        else:
            raise AttributeError("item_parameters method is not available for the chosen IRT model.")

    def infit_outfit(
        self,
        data: torch.Tensor = None,
        z: torch.Tensor = None,
        z_estimation_method: str = "ML",
        level: str = "item",
    ):
        """
        Calculate person or item infit and outfit statistics. These statistics help identifying items that do not behave as expected according to the model
        or respondents with unusual response patterns. Items that do not behave as expectedly can be reviewed for possible revision or removal 
        to improve the overall test quality and reliability. Respondents with unusual response patterns can be reviewed for possible cheating or other issues.

        Parameters
        ----------
        data : torch.Tensor
            The input data.
        z: torch.Tensor, optional
            The latent variable z scores for the provided data. If not provided, they will be computed using z_estimation_method.
        z_estimation_method : str, optional
            Method used to obtain the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively.
        level: str = "item", optional
            Specifies whether to compute item or respondent statistics. Can be 'item' or 'respondent'. (default is 'item')

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple with the infit and outfit statistics. The first tensor holds the infit statistics and the second tensor holds the outfit statistics.

        Notes
        -----
        Infit and outift are computed as follows :cite:p:`vanderLinden1997`:

        .. math::
            \\begin{align}
            \\text{Item j infit} &= \\frac{\\sum_{i=1}^{n} (O_{ij} - E_{ij})^2}{\\sum_{i=1}^{n} W_{ij}} \\\\
            \\text{Respondent i infit} &= \\frac{\\sum_{j=1}^{J} (O_{ij} - E_{ij})^2}{\\sum_{j=1}^{J} W_{ij}} \\\\
            \\text{Item j outfit} &= \\frac{\\sum_{i=1}^{n} (O_{ij} - E_{ij})^2/W_{ij}}{n} \\\\
            \\text{Respondent i outfit} &= \\frac{\\sum_{j=1}^{J} (O_{ij} - E_{ij})^2/W_{ij}}{J}
            \\end{align}

        Where:

        - :math:`J` is the number of items,
        - :math:`n` is the number of respondents,
        - :math:`O_{ij}` is the observed score on the :math:`j`-th item from the :math:`i`-th respondent.
        - :math:`E_{ij}` is the expected score on the :math:`j`-th item from the :math:`i`-th respondent, calculated from the IRT model.
        - :math:`W_{ij}` is the weight on the :math:`j`-th item from the :math:`j`-th respondent. This is the variance of the item score :math:`W_{ij}=\\sum^{M_j}_{m=0}(m-E_{ij})^2P_{ijk}` where :math:`M_j` is the maximum item score and :math:`P_{ijk}` is the model probability of a score :math:`k` on the :math:`j`-th item from the :math:`i`-th respondent.
        
        """
        return self.evaluator.infit_outfit(data, z, z_estimation_method, level)

    def information(self, z: torch.Tensor, item: bool = True, degrees: list[int] = None) -> torch.Tensor:
        """
        Calculate the Fisher information matrix for the z scores (or the information in the direction supplied by degrees).

        Parameters
        ----------
        z : torch.Tensor
            A 2D tensor containing latent variable z scores for which to compute the information. Each column represents one latent variable.
        item : bool, optional
            Whether to compute the information for each item (True) or for the test as a whole (False). Default is True.
        degrees : list[int], optional
            A list of angles in degrees between 0 and 90, one for each latent variable. Specifies the direction in which to compute the information. Default is None.

        Returns
        -------
        torch.Tensor
            A tensor with the information for each z score. Dimensions are:
            
            - By default: (z rows, items, FIM rows, FIM columns).
            - If degrees are specified: (z rows, items).
            - If item is False: (z rows, FIM rows, FIM columns).
            - If degrees are specified and item is False: (z rows).

        Notes
        -----
        In the context of IRT, the Fisher information matrix measures the amount of information
        that a test taker's responses :math:`X` carries about the latent variable(s)
        :math:`\\mathbf{z}`.

        The formula for the Fisher information matrix in the case of multiple parameters is:

        .. math::

            I(\\mathbf{z}) = E\\left[ \\left(\\frac{\\partial \\ell(X; \\mathbf{z})}{\\partial \\mathbf{z}}\\right) \\left(\\frac{\\partial \\ell(X; \\mathbf{z})}{\\partial \\mathbf{z}}\\right)^T \\right] = -E\\left[\\frac{\\partial^2 \\ell(X; \\mathbf{z})}{\\partial \\mathbf{z} \\partial \\mathbf{z}^T}\\right]

        Where:

        - :math:`I(\\mathbf{z})` is the Fisher Information Matrix.
        - :math:`\ell(X; \\mathbf{z})` is the log-likelihood of :math:`X`, given the latent variable vector :math:`\\mathbf{z}`.
        - :math:`\\frac{\\partial \\ell(X; \\mathbf{z})}{\\partial \\mathbf{z}}` is the gradient vector of the first derivatives of the log-likelihood of :math:`X` with respect to :math:`\\mathbf{z}`.
        - :math:`\\frac{\\partial^2 \\log f(X; \\mathbf{z})}{\\partial \\mathbf{z} \\partial \\mathbf{z}^T}` is the Hessian matrix of the second derivatives of the log-likelihood of :math:`X` with respect to :math:`\\mathbf{z}`.
        
        For additional details, see :cite:t:`Chang2017`.
        """
        return self.scorer.information(z, item, degrees)

    def latent_scores(
        self,
        data: torch.Tensor,
        scale: str = "z",
        standard_errors: bool = False,
        z: torch.Tensor = None,
        z_estimation_method: str = "ML",
        ml_map_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lbfgs_learning_rate: float = 0.25,
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
            The scoring method to use. Can be 'bit' or 'z'. (default is 'z')
        standard_errors : bool, optional
            Whether to return standard errors for the latent scores. (default is False)
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
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            A 2D tensor of latent scores, with latent variables as columns. If standard_errors is True, returns a tuple with the latent scores and the standard errors.
        """
        return self.scorer.latent_scores(
            data,
            scale,
            standard_errors,
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

    def residuals(
        self,
        data: torch.Tensor = None,
        z: torch.Tensor = None,
        z_estimation_method: str = "ML",
        average_over: str = "none",
    ) -> torch.Tensor:
        """
        Compute model residuals using the supplied data.  
        
        For multiple choice models, the residuals are computed as 1 - the probability of the selected response option.
        For other models, the residuals are computed as the difference between the observed and model expected item scores.

        Parameters
        ----------
        data : torch.Tensor
            The input data.
        z: torch.Tensor, optional
            The latent variable z scores for the provided data. If not provided, they will be computed using z_estimation_method.
        z_estimation_method : str, optional
            Method used to obtain the z scores. Can be 'NN', 'ML', 'EAP' or 'MAP' for neural network, maximum likelihood, expected a posteriori or maximum a posteriori respectively.
        average_over: str = "none", optional
            Whether to average the residuals and over which level. Can be 'everything', 'items', 'respondents' or 'none'. Use 'none' for no average. For example, with 'respondent' the residuals are averaged over all respondents and is thus an average per item. (default is 'none')
            
        Returns
        -------
        torch.Tensor
            The residuals.
        """
        return self.evaluator.residuals(data, z, z_estimation_method, average_over)

    def sample_test_data(self, z: torch.Tensor) -> torch.Tensor:
        """
        Sample test data for the provided z scores.

        Parameters
        ----------
        z : torch.Tensor
            The latent scores.

        Returns
        -------
        torch.Tensor
            The sampled test data.
        """
        return self.model.sample_test_data(z)

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
            - 'gmm' for a gaussian mixture model.

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

    def save_model(self, path: str) -> None:
        """
        Save the fitted model.

        Parameters
        -------
        path : str
            Where to save fitted model.
        """
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "train_data": self.algorithm.train_data,
            "training_z_scores": self.algorithm.training_z_scores,
            "training_history": self.algorithm.training_history,
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
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "train_data" in checkpoint:
            self.algorithm.train_data = checkpoint["train_data"]
        if "training_z_scores" in checkpoint:
            self.algorithm.training_z_scores = checkpoint["training_z_scores"]
        if "training_history" in checkpoint:
            self.algorithm.training_history = checkpoint["training_history"]
        if isinstance(self.algorithm, AEIRT) or isinstance(self.algorithm, VAEIRT):
            self.algorithm.encoder.load_state_dict(checkpoint["encoder_state_dict"])

    def plot_training_history(self) -> go.Figure:
        """
        Plots the training history of the model.

        Returns
        -------
        go.Figure
            The Plotly Figure object for the plot.
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
    
    def plot_expected_sum_score(
        self,
        items: list[int] = None,
        scale: str = "bit",
        latent_variables: tuple[int] = (1,),
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
        Plots the expected sum score from the model against the latent variable(s).
        Supports full test scores, a single item or a subset of items.

        Parameters
        ----------
        items : list[int], optional
            The items used to compte the sum score. If None, all items are used. (default is None)
        scale : str, optional
            The scale to plot against. Can be 'bit' or 'z'. (default is 'bit')
        latent_variables : tuple[int], optional
            The latent variables to plot. (default is (1,))
        title : str, optional
            The title for the plot. (default is None)
        x_label : str, optional
            The label for the X-axis. (default is None and uses "Latent variable" for one latent variable and "Latent variable 1" for two latent variables)
        y_label : str, optional
            The label for the Y-axis. (default is None and uses "Expected sum score" or "Expected item score" for one latent variable, and "Latent variable 2" for two latent variables)
        color : str, optional
            The color to use for plots with one latent variable. (default is None and uses the default color sequence for the plotly_white template)
        colorscale : str, optional
            Sets the colorscale for the multiple latent variable surface plots. See https://plotly.com/python/builtin-colorscales/ (default is "Plasma")
        z_range : tuple[float, float], optional
            Only for scale = 'z'. The z range for plotting. (default is None and uses limits based on training data)
        second_z_range : tuple[float, float], optional
            Only for scale = 'z'. The range for plotting for the second latent variable. (default is None and uses limits based on training data)
        steps : int, optional
            The number of steps along each z axis to construct the latent variable grid for which the sum score is evaluated at. (default is None and uses 100 for one latent variable and 18 for two latent variables)
        fixed_zs: torch.Tensor, optional
            Only for multdimensional models. Fixed values for latent space variable not plotted. (default is None and uses the medians in the training data)
        **kwargs : dict, optional
            Additional keyword arguments used for bit score computation. See :meth:`irtorch.irt.IRT.bit_scores_from_z` for details. 
        """
        return self.plotter.plot_expected_sum_score(
            items=items,
            scale=scale,
            latent_variables=latent_variables,
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

    def plot_information(
        self,
        items: list[int] = None,
        scale: str = "z",
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
        Plots the Fisher information function against the latent variable(s).
        Supports both item and test information. See :meth:`information` for details.

        Parameters
        ----------
        items : list[int], optional
            The items to plot. If None, the full test information is plotted. (default is None)
        scale : str, optional
            The scale to plot against. Can be 'bit' or 'z'. (default is 'bit')
        latent_variables : tuple[int], optional
            The latent variables to plot. (default is (1,))
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

    def plot_item_probabilities(
        self,
        item: int,
        scale: str = "bit",
        latent_variables: tuple = (1, ),
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        z_range: tuple[float, float] = None,
        second_z_range: tuple[float, float] = None,
        steps: int = 300,
        fixed_zs: torch.Tensor = None,
        plot_group_fit: bool = False,
        group_fit_groups: int = 10,
        group_fit_data: int = None,
        group_fit_population_z: torch.Tensor = None,
        grayscale: bool = False,
        plot_derivative: bool = False,
        **kwargs
    ) -> go.Figure:
        """
        Plots the item probability curves for a given item. Supports 2D and 3D plots.

        Parameters
        ----------
        item : int
            The item to plot.
        scale : str, optional
            The scale to plot against. Can be 'bit' or 'z'. (default is 'bit')
        latent_variables : tuple, optional
            The latent variables to plot. (default is (1,))
        title : str, optional
            The title for the plot. (default is None and uses "IRF - Item {item}")
        x_label : str, optional
            The label for the X-axis. (default is None and uses "Latent variable" for one latent variable and "Latent variable 1" for two latent variables)
        y_label : str, optional
            The label for the Y-axis. (default is None and uses "Probability")
        z_range : tuple, optional
            Only for scale = 'z'. The z range for plotting. (default is None and uses limits based on training data)
        second_z_range : tuple, optional
            Only for scale = 'z'. The range for plotting for the second latent variable. (default is None and uses limits based on training data)
        steps : int, optional
            The number of steps along each z axis used for probability evaluation. (default is 300)
        fixed_zs: torch.Tensor, optional
            Only for multdimensional models. Fixed values for latent space variable not plotted. (default is None and uses the medians in the training data)
        plot_group_fit : bool, optional
            Plot group average probabilities to assess fit. (default is False)
        group_fit_groups : int, optional
            Only for plot_group_fit = True. The number of groups. (default is 10)
        group_fit_data: torch.tensor, optional
            Only for plot_group_fit = True. The data used for group fit plots. Uses training data if not provided. (default is None)
        group_fit_population_z : torch.tensor, optional
            Only for plot_group_fit = True. The z scores corresponding to group_fit_data. Will be estimated using group_z_estimation_method if not provided. (default is None)
        grayscale : bool, optional
            Plot the item probability curves in grey scale. (default is False)
        plot_derivative : bool, optional
            Plot the first derivative of the item probability curves. Only for plots with one latent variable. (default is False)
        **kwargs : dict, optional
            Additional keyword arguments used for bit score computation. See :meth:`irtorch.irt.IRT.bit_scores_from_z` for details. 

        Returns
        -------
        go.Figure
            The Plotly Figure object for the plot.
        """
        return self.plotter.plot_item_probabilities(
            item=item,
            scale=scale,
            latent_variables=latent_variables,
            title=title,
            x_label=x_label,
            y_label=y_label,
            z_range=z_range,
            second_z_range=second_z_range,
            steps=steps,
            fixed_zs=fixed_zs,
            plot_group_fit=plot_group_fit,
            group_fit_groups=group_fit_groups,
            group_fit_data=group_fit_data,
            group_fit_population_z=group_fit_population_z,
            grayscale=grayscale,
            plot_derivative=plot_derivative,
            **kwargs
        )

    def plot_item_entropy(
        self,
        item: int,
        scale: str = "bit",
        latent_variables: tuple[int] = (1,),
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
        Plot the entropy of an item against the latent variable(s).

        Parameters
        ----------
        item : int
            The item for which to plot the entropy.
        scale : str, optional
            The scale to plot against. Can be 'bit' or 'z'. (default is 'bit')
        latent_variables : tuple[int], optional
            The latent variables to plot. (default is (1,))
        title : str, optional
            The title for the plot. (default is None)
        x_label : str, optional
            The label for the X-axis. (default is None and uses "Latent variable" for one latent variable and "Latent variable 1" for two latent variables)
        y_label : str, optional
            The label for the Y-axis. (default is None and uses 'Entropy' for one latent variable, and "Latent variable 2" for two latent variables)
        color : str, optional
            The color to use for plots with one latent variable. (default is None and uses the default color sequence for the plotly_white template)
        colorscale : str, optional
            Sets the colorscale for the multiple latent variable surface plots. See https://plotly.com/python/builtin-colorscales/ (default is "Plasma")
        z_range : tuple[float, float], optional
            Only for scale = 'z'. The z range for plotting. (default is None and uses limits based on training data)
        second_z_range : tuple[float, float], optional
            Only for scale = 'z'. The range for plotting for the second latent variable. (default is None and uses limits based on training data)
        steps : int, optional
            The number of steps along each z axis to construct the latent variable grid for which the sum score is evaluated at. (default is None and uses 100 for one latent variable and 18 for two latent variables)
        fixed_zs: torch.Tensor, optional
            Only for multdimensional models. Fixed values for latent space variable not plotted. (default is None and uses the medians in the training data)
        **kwargs : dict, optional
            Additional keyword arguments used for bit score computation. See :meth:`irtorch.irt.IRT.bit_scores_from_z` for details. 

        Returns
        -------
        go.Figure
            The Plotly Figure object for the plot.
        """
        return self.plotter.plot_item_entropy(
            item=item,
            scale=scale,
            latent_variables=latent_variables,
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

    def plot_item_latent_variable_relationships(
        self,
        relationships: torch.Tensor,
        title: str = "Relationships: Items vs. latent variables",
        x_label: str = "Latent variable",
        y_label: str = "Items",
        colorscale: str = "Plasma",
    ) -> go.Figure:
        """
        Create a heatmap of item-latent variable relationships.

        Parameters
        ----------
        relationships : torch.Tensor
            A tensor of item-latent variable relationships. Typically the returned tensor from expected_item_score_slopes() where each row represents an item and each column represents a latent variable.
        title : str, optional
            The title for the plot. (default is "Relationships: Items vs. latent variables")
        x_label : str, optional
            The label for the X-axis. (default is "Latent variable")
        y_label : str, optional
            The label for the Y-axis. (default is "Items")
        colorscale : str, optional
            Sets the colorscale figure. See https://plotly.com/python/builtin-colorscales/ (default is "Plasma")

        Returns
        -------
        go.Figure
            The Plotly Figure object for the plot.
        """
        return self.plotter.plot_item_latent_variable_relationships(
            relationships, title, x_label, y_label, colorscale
        )
