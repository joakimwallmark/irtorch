import torch
from splinetorch.b_spline_basis import b_spline_basis, b_spline_basis_derivative
from irtorch.models.base_irt_model import BaseIRTModel

class BSplineBasisFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, knots, degree):
        # Save inputs needed for backward.
        ctx.save_for_backward(x, knots)
        ctx.degree = degree
        # Compute the forward B-spline basis using your original function.
        # (This is the non-smooth version, so autograd's default backward would be zero.)
        basis = b_spline_basis(x, knots, degree)
        return basis

    @staticmethod
    def backward(ctx, grad_output):
        x, knots = ctx.saved_tensors
        degree = ctx.degree
        # Compute the analytic derivative of the B-spline basis with respect to x.
        d_basis_dx = b_spline_basis_derivative(x, knots, degree, order=1)
        # grad_output has shape (n_points, n_bases). For each x[i], the contribution is:
        # dLoss/dx[i] = sum_j grad_output[i,j] * d_basis_dx[i,j]
        grad_x = (grad_output * d_basis_dx).sum(dim=1)
        # We assume no gradients are needed with respect to knots or degree.
        return grad_x, None, None

class SurprisalSpline(BaseIRTModel):
    r"""
    Surprisals (negative logarithm of probabilities) are modeled using monotone cubic B-splines, hence the name.
    Note that this model requires each latent variable to be positively related to all items.
    Ordered polytomously scored data is supported.

    Parameters
    ----------
    data: torch.Tensor, optional
        A 2D torch tensor with test data. Used to automatically compute item_categories. Columns are items and rows are respondents. (default is None)
    latent_variables : int
        Number of latent variables.
    item_categories : list[int]
        Number of categories for each item. One integer for each item. Missing responses exluded.
    item_theta_relationships : torch.Tensor, optional
        Boolean tensor of shape (items, latent_variables) specifying which latent variables
        affect which items. True indicates a relationship exists.
    knots : list[float], optional
        Positions of the internal knots (bounds excluded) for the B-spline basis functions. If not provided, defaults to
        [-1.7, -0.8, -0.3, 0, 0.3, 0.8, 1.7].
    degree : int, optional
        Degree of the B-spline polynomials. Defaults to 3 (cubic splines).

    Notes
    -----
    For an item :math:`j` with ordered item scores :math:`x=0, 1, 2, ..., M_j` the model defines the probability for responding with a score of :math:`x` as follows:

    .. math::

        P(X_j = x | \mathbf{\theta}) = \left(1-\exp[-S_x(\mathbf{\theta})]\right)\prod^{M_j}_{m=x+1}\exp(-S_m(\mathbf{\theta}))

    where :math:`\mathbf{\theta}` is a vector of latent variables and the :math:`S_x(\mathbf{\theta})` functions are surprisals (negative logarithms of probabilities) defined as:

    .. math::
        S_{M_j}(\mathbf{\theta})   & = -\log P(X_j < M_j | \mathbf{\theta}) \\
        S_{M_j-1}(\mathbf{\theta}) & = -\log P(X_j < M_j-1 | X_j < M_j, \mathbf{\theta}) \\
        S_{M_j-2}(\mathbf{\theta}) & = -\log P(X_j < M_j-2 | X_j-1 < M_j, \mathbf{\theta}) \\
        ... \\
        S_{1}(\mathbf{\theta}) & = -\log P(X_j < 1 | X_j < 2, \mathbf{\theta}) \\
        S_{0}(\mathbf{\theta}) & = -\infty

    Monotone B-splines are used to model the surprisal functions after transforming the latent variables to the unit interval using the sigmoid function.

    Examples
    --------
    >>> from irtorch.models import SurprisalSpline
    >>> from irtorch.estimation_algorithms import MML
    >>> from irtorch.load_dataset import swedish_national_mathematics_1
    >>> data = swedish_national_mathematics_1()
    >>> model = SurprisalSpline(data)
    >>> model.fit(train_data=data, algorithm=MML())
    """
    def __init__(
        self,
        data: torch.Tensor = None,
        latent_variables: int = 1,
        item_categories: list[int] = None,
        item_theta_relationships: torch.Tensor = None,
        knots: list[float] = None,
        degree: int = 3,
    ):
        if item_categories is None:
            if data is None:
                raise ValueError("Either item_categories or data must be provided to initialize the model.")
            else:
                # replace nan with -inf to get max
                item_categories = (torch.where(~data.isnan(), data, torch.tensor(float('-inf'))).max(dim=0).values + 1).int().tolist()

        super().__init__(latent_variables=latent_variables, item_categories=item_categories)
        if item_theta_relationships is None:
            item_theta_relationships = torch.tensor([[True] * latent_variables] * self.items, dtype=torch.bool)
        else:
            if item_theta_relationships.shape != (len(item_categories), latent_variables):
                raise ValueError(
                    f"latent_item_connections must have shape ({len(item_categories)}, {latent_variables})."
                )
            assert(item_theta_relationships.dtype == torch.bool), "latent_item_connections must be boolean type."
            assert(torch.all(item_theta_relationships.sum(dim=1) > 0)), "all items must have a relationship with a least one latent variable."

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
        # (lv, n_bases, items, max(item_categories)-1) with True where splines are supposed to be
        spline_mask = item_theta_relationships.T.reshape(self.latent_variables, 1, -1, 1)
        spline_mask = spline_mask.repeat(1, self.n_bases, 1, max(item_categories)-1)
        # remove entries for items with fewer responses
        for item, item_cat in enumerate(item_categories):
            spline_mask[:, :, item, :(max(item_categories)-item_cat)] = False

        self.register_buffer('spline_mask', spline_mask)
        self.coefficients = torch.nn.Parameter(torch.full((spline_mask.sum(), ), -2.5))

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass of the model.

        Parameters
        ----------
        theta : torch.Tensor
            2D tensor with latent variables. Rows are respondents and latent variables are columns. 

        Returns
        -------
        output : torch.Tensor
            3D tensor of response probabilities with dimensions (respondents, items, item categories)
        """
        rescaled_theta = theta.sigmoid()
        all_coef = torch.zeros(self.latent_variables, self.n_bases, self.items, max(self.item_categories)-1, device=theta.device)
        # (thetas, basis)

        if self.basis is not None: # use precomputed basis if available
            basis = self.basis
        else:
            basis = BSplineBasisFunction.apply(rescaled_theta.T.flatten(), self.knots, self.degree)
            basis = basis.view(self.latent_variables, -1, self.n_bases)  # (lv, batch, basis)
        
        # (lv, basis, items, max(item_categories)-1)
        all_coef[self.spline_mask] = torch.nn.functional.softplus(self.coefficients)
        all_coef = all_coef.cumsum(dim=1)

        surp = torch.einsum('...pb,...bic->...pic', basis, all_coef) # (lv, batch, items, max(item_categories)-1)
        surprisal = surp.sum(dim=0) # (batch, items, max(item_categories)-1)

        ones = torch.ones(*surprisal.shape[:-1], 1, device=theta.device)
        a = torch.cat((ones, 1/surprisal.cumsum(dim=-1).exp()), dim=-1)
        b = torch.cat((1-1/surprisal.exp(), ones), dim=-1)
        probabilities = a*b
        return torch.flip(probabilities, dims=[-1])

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
