import torch
from irtorch.models import BaseIRTModel
from irtorch.torch_modules import BSplineBasisFunction

class NestedLogit(BaseIRTModel):
    r"""
    Nested logit IRT model for multiple choice items :cite:p:`Suh2010`. The original paper uses a 3PL model :class:`irtorch.models.ThreeParameterLogistic` nested with a nominal response model :class:`irtorch.models.NominalResponse`.
    This implementation allows one to choose any IRT model for dichotomously scored items in place of the 3PL, and either a nominal response model or B-splines for the incorrect responses.

    Requires the correct response model to be specified in advance. It can also be a previously fitted model.

    Parameters
    ----------
    mc_correct : list[int]
        Only for multiple choice data. The correct response option for each item. (Default: None)
    correct_response_model : BaseIRTModel, optional
        A fitted IRT model instance to use for the correct response.
    incorrect_response_model : str, optional
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
        [-1.7, -0.7, 0, 0.7, 1.7].

    Notes
    -----
    For an item :math:`j` with :math:`m=0, 1, 2, \ldots, M_j` possible item scores, the model defines the probability for responding with a score of :math:`x` as follows:

    .. math::

        P(X_j=x | \mathbf{\theta}) = \begin{cases}
            P(X_j=c_j|\mathbf{\theta}), & \text{if } x = c_j\\
            (1-P(X_j=c_j|\mathbf{\theta}))P(X_j=x|x\neq c_j, \mathbf{\theta}), & \text{otherwise}
        \end{cases}
    
    where:

    - :math:`\mathbf{\theta}` is a vector of latent variables.
    - :math:`c_j` is the correct response option for item :math:`j`.
    - :math:`P(X_j=c_j|\mathbf{\theta})` is computed using the model supplied as correct_response_model.
    - The conditional probabilities :math:`P(X_j=x|x\neq c_j, \mathbf{\theta})` are estimated using either a nominal response model or B-splines.

    Examples
    --------
    >>> from irtorch.models import NestedLogit, ThreeParameterLogistic
    >>> from irtorch.estimation_algorithms import MML
    >>> from irtorch.load_dataset import swedish_sat_quantitative
    >>> data, correct_responses = swedish_sat_quantitative()
    >>> model_3pl = ThreeParameterLogistic(items = data.shape[1])
    >>> model_nested = NestedLogit(correct_responses, model_3pl, data=data)
    >>> model_nested.fit(train_data=data, algorithm=MML())
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
            # replace nan with -inf to get max
            item_categories = (torch.where(~data.isnan(), data, torch.tensor(float('-inf'))).max(dim=0).values + 1).int().tolist()

        super().__init__(latent_variables=latent_variables, item_categories=item_categories, mc_correct=mc_correct)
        if item_theta_relationships is None:
            item_theta_relationships = torch.tensor([[True] * latent_variables] * self.items, dtype=torch.bool)
        else:
            if item_theta_relationships.shape != (len(item_categories), latent_variables):
                raise ValueError(
                    f"item_theta_relationships must have shape ({len(item_categories)}, {latent_variables})."
                )
            if not isinstance(item_theta_relationships, torch.Tensor):
                item_theta_relationships = torch.tensor(item_theta_relationships, dtype=torch.bool)
            elif item_theta_relationships.dtype != torch.bool:
                try:
                    item_theta_relationships = item_theta_relationships.bool()
                except RuntimeError as exc:
                    raise TypeError("item_theta_relationships must be convertible to boolean type.") from exc
            if not torch.all(item_theta_relationships.sum(dim=1) > 0):
                raise ValueError("all items must have a relationship with a least one latent variable.")

        self.correct_response_model: BaseIRTModel = correct_response_model
        self.incorrect_response_model = incorrect_response_model

        correct_mask = torch.zeros(self.items, self.max_item_responses, dtype=torch.bool)
        for item in range(self.items):
            if 0 > mc_correct[item] or mc_correct[item] >= self.item_categories[item]:
                raise ValueError(f"mc_correct[{item}]={mc_correct[item]} is out of bounds for item {item} with {self.item_categories[item]} categories.")
            correct_mask[item, mc_correct[item]] = True
        self.register_buffer('correct_mask', correct_mask)

        incorrect_item_categories = [item_cat - 1 for item_cat in item_categories]
        if incorrect_response_model == "nominal":
            from irtorch.models import NominalResponse
            self.nominal_response_model = NominalResponse(
                latent_variables=latent_variables,
                item_categories=incorrect_item_categories,
                item_theta_relationships=item_theta_relationships,
            )
        elif incorrect_response_model == "bspline":
            knots = torch.tensor([-1.7, -0.7, 0, 0.7, 1.7]) if knots is None else torch.tensor(knots)
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

        # --- Precompute Incorrect Index Mapping ---
        self._max_incorrect_cats =max(incorrect_item_categories)
        # Map from compact incorrect index (0 to k-1) to full response index (0 to max_responses-1)
        map_indices = torch.full((self.items, self._max_incorrect_cats), -1, dtype=torch.long) # Fill with -1 initially

        # Find the incorrect item indices
        full_range = torch.arange(self.max_item_responses)
        for i in range(self.items):
            map_indices[i] = full_range[~self.correct_mask[i]]

        self.register_buffer('incorrect_indices_map', map_indices)

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
        batch_size = theta.shape[0]
        device = theta.device
        target_shape = (batch_size, self.items, self.max_item_responses)

        all_probabilities = torch.zeros(theta.shape[0], self.items, self.max_item_responses, device=device)
        corrrect_output = self.correct_response_model(theta)
        correct_probabilities = self.correct_response_model.probabilities_from_output(corrrect_output)
        p_correct_total = correct_probabilities[:, :, 1]  # (batch, items)
        p_incorrect_total = correct_probabilities[:, :, 0] # (batch, items)

        if self.incorrect_response_model == "nominal":
            incorrect_output = self.nominal_response_model(theta)
            # Shape: (batch, items, _max_incorrect_cats)
            p_k_given_incorrect = self.nominal_response_model.probabilities_from_output(incorrect_output)
        elif self.incorrect_response_model == "bspline":
            if self.basis is not None: # use precomputed basis if available
                basis = self.basis
            else:
                rescaled_theta = theta.sigmoid()
                basis = BSplineBasisFunction.apply(rescaled_theta.T.flatten(), self.knots, self.degree)
                basis = basis.view(self.latent_variables, -1, self.n_bases)  # (lv, batch, basis)

            all_coef = torch.zeros(self.latent_variables, self.n_bases, self.items, max(self.item_categories)-1, device=device)
            all_coef[self.spline_mask] = self.coefficients # (lv, basis, items, max(incorrect_item_categories))
            logits = torch.einsum('...pb,...bic->...pic', basis, all_coef) # (lv, batch, items, max(incorrect_item_categories))
            logit_sums = logits.sum(dim=0) # (batch, items, max(incorrect_item_categories))
            # logit_sums[:, ~self.response_mask] = -torch.inf

            # Apply response mask (broadcasted) before softmax
            batch_response_mask = self.response_mask.unsqueeze(0).expand(batch_size, -1, -1) # (batch, items, max_incorrect)
            logit_sums = torch.where(batch_response_mask, logit_sums, torch.tensor(float('-inf'), device=device))

            # Shape: (batch, items, _max_incorrect_cats)
            p_k_given_incorrect = torch.softmax(logit_sums, dim=2)
        else:
            raise ValueError("Incorrect response model must be 'nominal' or 'bspline'.")

        p_k_incorrect_scaled = p_k_given_incorrect * p_incorrect_total.unsqueeze(-1)
        all_probabilities = torch.zeros(target_shape, device=device)

        # Get indices for correct responses (precomputed mask -> indices tensor)
        mc_correct_tensor = torch.tensor(self.mc_correct, dtype=torch.long, device=device)
        correct_indices = mc_correct_tensor.view(1, -1, 1).expand(batch_size, -1, -1) # (batch, items, 1)
        # Assemble correct probabilities using Scatter (out-of-place)
        all_probabilities = all_probabilities.scatter(
            dim=-1,
            index=correct_indices,
            src=p_correct_total.unsqueeze(-1) # (batch, items, 1)
        )

        # Expand map and mask for batch
        # Note: Buffers are automatically on the correct device
        map_expanded = self.incorrect_indices_map.unsqueeze(0).expand(batch_size, -1, -1) # (batch, items, max_incorrect)

        # Use scatter_add to add the scaled incorrect probabilities
        # Indices from map_expanded, source from masked_src
        all_probabilities = all_probabilities.scatter_add(
            dim=-1,
            index=map_expanded, # Target indices in full response range
            src=p_k_incorrect_scaled     # Scaled incorrect probs (masked)
        )

        # Optional: Clamp probabilities slightly away from 0/1 for numerical stability downstream
        # all_probabilities = torch.clamp(all_probabilities, min=1e-9, max=1.0 - 1e-9)
        # Optional: Renormalize - should ideally sum to 1 already
        # all_probabilities = all_probabilities / all_probabilities.sum(dim=-1, keepdim=True)

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
        if loss_reduction == "none":
            if missing_mask is not None:
                ll_masked = torch.full((respondents, ), torch.nan, device= ll.device)
                ll_masked[~missing_mask] = ll
                return ll_masked
            return ll
        raise ValueError("loss_reduction must be 'sum' or 'none'")

    def precompute_basis(self, theta: torch.Tensor):
        rescaled_theta = theta.sigmoid()
        basis = BSplineBasisFunction.apply(rescaled_theta.T.flatten(), self.knots, self.degree)
        self.basis = basis.view(self.latent_variables, -1, self.n_bases)
