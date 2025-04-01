import torch
from irtorch.models import BaseIRTModel
from irtorch.torch_modules import BSplineBasisFunction

class NestedLogit(BaseIRTModel):
    r"""
    Nested logit IRT model for multiple choice items :cite:p:`Birnbaum1968`. The original paper uses a 3PL model :class:`irtorch.models.ThreeParameterLogistic` nested with a nominal response model :class:`irtorch.models.NominalResponse`.
    This implementation allows one to choose any IRT model for dichotomously scored items in place of the 3PL, and either a nominal response model or B-splines for the incorrect responses.

    Requires the correct response model to be fitted in advance. The incorrect response probabilities are then estimated using a nominal response model or B-splines.

    Parameters
    ----------
    mc_correct : list[int]
        Only for multiple choice data. The correct response option for each item. (Default: None)
    correct_response_model : BaseIRTModel, optional
        A fitted IRT model instance to use for the correct response.
    Incorrect response model : str, optional
        The model to use for the incorrect responses. Can be 'nominal' or 'bspline'. (Default: 'nominal')
    data: torch.Tensor, optional
        A 2D torch tensor with test data. Used to automatically compute item_categories. Columns are items and rows are respondents. (default is None)
    latent_variables : int, optional
        Number of latent variables. (default is 1)
    item_categories : list[int], optional
        Number of categories for each item. One integer for each item. Missing responses exluded. (default is None)
    item_theta_relationships : torch.Tensor, optional
        A boolean tensor of shape (items, latent_variables). If specified, the model will have connections between latent dimensions and items where the tensor is True. If left out, all latent variables and items are related (Default: None)
    degree : int, optional
        The degree of the B-spline basis functions when using the B-spline model for the incorrect responses. (Default: 3)
    knots : list[float], optional
        The positions of the internal knots (bounds excluded) for the B-spline basis functions when using the B-spline model for the incorrect responses. If not provided, defaults to
        [-1.7, -0.8, -0.3, 0, 0.3, 0.8, 1.7].
        
    """
    def __init__(
        self,
        mc_correct: list[int],
        correct_response_model: BaseIRTModel,
        incorrect_response_model: str = "nominal",
        data: torch.Tensor = None,
        latent_variables: int = 1,
        item_categories: list[int] = None,
        item_theta_relationships: torch.Tensor = None,
        degree: int = 3,
        knots: list[float] = None,
    ):
        if item_categories is None:
            if data is None:
                raise ValueError("Either item_categories or data must be provided to initialize the model.")
            else:
                # replace nan with -inf to get max
                item_categories = (torch.where(~data.isnan(), data, torch.tensor(float('-inf'))).max(dim=0).values + 1).int().tolist()

        super().__init__(latent_variables=latent_variables, item_categories=item_categories, mc_correct=mc_correct)
        if item_theta_relationships is None:
            item_theta_relationships = torch.tensor([[True] * latent_variables] * self.items, dtype=torch.bool)
        else:
            if item_theta_relationships.shape != (len(item_categories), latent_variables):
                raise ValueError(
                    f"latent_item_connections must have shape ({len(item_categories)}, {latent_variables})."
                )
            assert(item_theta_relationships.dtype == torch.bool), "latent_item_connections must be boolean type."
            assert(torch.all(item_theta_relationships.sum(dim=1) > 0)), "all items must have a relationship with a least one latent variable."


        self.correct_response_model: BaseIRTModel = correct_response_model
        self.incorrect_response_model = incorrect_response_model
        self.correct_mask = torch.zeros(self.items, self.max_item_responses, dtype=torch.bool)
        for item in range(self.items):
            self.correct_mask[item, mc_correct[item]] = True

        incorrect_item_categories = [item_cat - 1 for item_cat in item_categories]
        if incorrect_response_model == "nominal":
            from irtorch.models import NominalResponse
            self.nominal_response_model = NominalResponse(
                latent_variables=latent_variables,
                item_categories=incorrect_item_categories,
                item_theta_relationships=item_theta_relationships,
            )
        elif incorrect_response_model == "bspline":
            if knots is None:
                knots = torch.tensor([-1.7, -0.8, -0.3, 0, 0.3, 0.8, 1.7])
            else:
                knots = torch.tensor(knots)

            self.basis = None
            knots = knots.sigmoid()
            knots = torch.cat((torch.zeros(degree+1), knots, torch.ones(degree+1)))
            self.register_buffer('knots', knots)
            self.n_bases = len(knots) - degree - 1
            self.degree = degree
            # (lv, n_bases, items, max(incorrect_item_categories)) with True where splines are supposed to be
            spline_mask = item_theta_relationships.T.reshape(self.latent_variables, 1, -1, 1)
            spline_mask = spline_mask.repeat(1, self.n_bases, 1, max(incorrect_item_categories))
            response_mask = torch.ones((self.items, max(incorrect_item_categories)), dtype=torch.bool)
            # remove entries for items with fewer responses
            for item, item_cat in enumerate(incorrect_item_categories):
                spline_mask[:, :, item, item_cat:] = False
                response_mask[item, item_cat:] = False

            self.register_buffer('spline_mask', spline_mask)
            self.register_buffer('response_mask', response_mask)
            self.coefficients = torch.nn.Parameter(torch.randn((spline_mask.sum(), ))*0.01)
        else:
            raise ValueError("Incorrect response model must be 'nominal' or 'bspline'.")

    def _dichotomize_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Dichotomize the data by replacing all incorrect responses with 0 and all correct responses with 1.
        """
        dichotomized_data = torch.zeros_like(data)
        for item, correct_response in enumerate(self.mc_correct):
            dichotomized_data[:, item] = (data[:, item] == correct_response).float()
        return dichotomized_data

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        theta : torch.Tensor
            2D tensor with latent variables. Rows are respondents and latent variables are columns. 

        Returns
        -------
        output : torch.Tensor
            2D tensor. Rows are respondents and columns are item category logits.
        """
        all_probabilities = torch.zeros(theta.shape[0], self.items, self.max_item_responses, device=theta.device)
        corrrect_output = self.correct_response_model(theta)
        correct_probabilities = self.correct_response_model.probabilities_from_output(corrrect_output)
        if self.incorrect_response_model == "nominal":
            incorrect_output = self.nominal_response_model(theta)
            incorrect_probabilities = self.nominal_response_model.probabilities_from_output(incorrect_output)
        elif self.incorrect_response_model == "bspline":
            if self.basis is not None: # use precomputed basis if available
                basis = self.basis
            else:
                rescaled_theta = theta.sigmoid()
                basis = BSplineBasisFunction.apply(rescaled_theta.T.flatten(), self.knots, self.degree)
                basis = basis.view(self.latent_variables, -1, self.n_bases)  # (lv, batch, basis)
            
            all_coef = torch.zeros(self.latent_variables, self.n_bases, self.items, max(self.item_categories)-1, device=theta.device)
            all_coef[self.spline_mask] = self.coefficients # (lv, basis, items, max(incorrect_item_categories))
            logits = torch.einsum('...pb,...bic->...pic', basis, all_coef) # (lv, batch, items, max(incorrect_item_categories))
            logit_sums = logits.sum(dim=0) # (batch, items, max(incorrect_item_categories))
            logit_sums[:, ~self.response_mask] = -torch.inf
            incorrect_probabilities = torch.softmax(logit_sums, dim=2)
        else:
            raise ValueError("Incorrect response model must be 'nominal' or 'bspline'.")

        incorrect_probabilities = incorrect_probabilities * correct_probabilities[:, :, 0:1]
        all_probabilities[:, self.correct_mask] = correct_probabilities[:, :, 1]
        all_probabilities[:, ~self.correct_mask] = incorrect_probabilities.view(theta.shape[0], -1)
        return all_probabilities

    @torch.no_grad()
    def item_theta_relationship_directions(self, theta:torch.Tensor = None) -> torch.Tensor:
        """
        Get the relationship directions between each item and latent variable for a fitted model.

        Parameters
        ----------
        theta : torch.Tensor, optional
            Only needed for NR models. A 2D tensor with latent theta scores from the population of interest. Each row represents one respondent, and each column represents a latent variable.

        Returns
        -------
        torch.Tensor
            A 2D tensor with the relationships between the items and latent variables. Items are rows and latent variables are columns.
        """
        return self.correct_response_model.item_theta_relationship_directions(theta)

    def probabilities_from_output(self, output: torch.Tensor) -> torch.Tensor:
        """
        Probabilities forwarded from the output tensor from the forward method.
        For this model, the output is already the probabilities and the method exists for compatibility and consistency between models.

        Parameters
        ----------
        output : torch.Tensor
            Output tensor from the forward pass.

        Returns
        -------
        torch.Tensor
            3D tensor with dimensions (respondents, items, item categories)
        """
        return output
    
    def log_likelihood(
        self,
        data: torch.Tensor,
        output: torch.Tensor,
        missing_mask: torch.Tensor = None,
        loss_reduction: str = "sum",
    ) -> torch.Tensor:
        """
        Compute the log likelihood.

        Parameters
        ----------
        data: torch.Tensor
            A 2D tensor with test data. Columns are items and rows are respondents
        output: torch.Tensor
            A 2D tensor with output. Columns are item response categories and rows are respondents
        missing_mask: torch.Tensor, optional
            A 2D tensor with missing data mask. (default is None)
        loss_reduction: str, optional 
            The reduction argument. (default is 'sum')
        
        Returns
        -------
        torch.Tensor
            The log likelihood.
        """
        data = data.long().view(-1)
        reshaped_probabilities = output.reshape(-1, self.max_item_responses)

        if missing_mask is not None:
            missing_mask = missing_mask.view(-1)
            reshaped_probabilities = reshaped_probabilities[~missing_mask]
            respondents = data.size(0)
            data = data[~missing_mask]

        ll = reshaped_probabilities[torch.arange(data.size(0)), data].log()
        if loss_reduction == "sum":
            return ll.sum()
        elif loss_reduction == "none":
            if missing_mask is not None:
                ll_masked = torch.full((respondents, ), torch.nan, device= ll.device)
                ll_masked[~missing_mask] = ll
                return ll_masked
            else:
                return ll
        else:
            raise ValueError("loss_reduction must be 'sum' or 'none'")
        
    def precompute_basis(self, theta: torch.Tensor):
        rescaled_theta = theta.sigmoid()
        basis = BSplineBasisFunction.apply(rescaled_theta.T.flatten(), self.knots, self.degree)
        self.basis = basis.view(self.latent_variables, -1, self.n_bases)
