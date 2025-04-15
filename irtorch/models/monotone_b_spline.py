import torch
from torch import nn
from irtorch.models.base_irt_model import BaseIRTModel
from irtorch.torch_modules import BSplineBasisFunction

class MonotoneBSpline(BaseIRTModel):
    r"""
    Monotone B-Spline IRT model for polytomously scored items with ordered responses.

    Parameters
    ----------
    data: torch.Tensor, optional
        A 2D torch tensor with test data. Used to automatically compute item_categories. Columns are items and rows are respondents. (default is None)
    latent_variables : int
        Number of latent variables.
    item_categories : list[int]
        Number of categories for each item. One integer for each item. Missing responses exluded.
    knots : list[float], optional
        Positions of the internal knots (bounds excluded) for the B-spline basis functions. If not provided, defaults to
        [-1.7, -0.7, 0, 0.7, 1.7].
    degree : int, optional
        Degree of the B-spline polynomials. Defaults to 3 (cubic splines).
    separate : str, optional
        Whether to fit separate latent variable functions for items or item response categories. Can be 'items' or 'categories'. 
        Note that 'categories' results in a more flexible model with more parameters. (Default: 'items')
        

    Notes
    -----
    For an item :math:`j` with :math:`m=0, 1, 2, \ldots, M_j` possible item responses/scores, the model defines the probability for responding with a score of :math:`x` as follows:

    .. math::

        P(X_j=x | \mathbf{\theta}) = \begin{cases}
            \dfrac{1}
            {1+\sum_{g=1}^{M_i}\exp \left(d_{jg}+\sum_{m=1}^g\delta_{jm}(\mathbf{\theta})\right)}, & \text{if } x = 0\\
            \dfrac{\exp \left(d_{jx}+\sum_{m=1}^x\delta_{jm}(\mathbf{\theta})\right)}
            {1+\sum_{g=1}^{M_i}\exp \left(d_{jg}+\sum_{m=1}^g\delta_{jm}(\mathbf{\theta})\right)}, & \text{otherwise}
        \end{cases}

    where:

    - :math:`\mathbf{\theta}` is a vector of latent variables.
    - :math:`\delta_{jm}(\mathbf{\theta})` is a monotonic B-spline.
    - :math:`b_{jm}` is a bias term.
    - Note that when separate='items', :math:`\delta_{jm}(\mathbf{\theta})` is the same for all response categories for the same item.
    
    Examples
    --------
    >>> from irtorch.models import MonotoneBSpline
    >>> from irtorch.estimation_algorithms import MML
    >>> from irtorch.load_dataset import swedish_national_mathematics_1
    >>> data = swedish_national_mathematics_1()
    >>> model = MonotoneBSpline(data)
    >>> model.fit(train_data=data, algorithm=MML())
    """
    def __init__(
        self,
        data: torch.Tensor = None,
        latent_variables: int = 1,
        item_categories: list[int] = None,
        knots: list[float] = None,
        degree: int = 3,
        separate: str = "items",
    ):
        if item_categories is None and data is None:
            raise ValueError("Either item_categories or data must be provided to initialize the model.")
        
        if item_categories is None:
            # replace nan with -inf to get max
            data_no_nan = torch.where(torch.isnan(data), float("-inf"), data)
            item_categories = (data_no_nan.max(dim=0).values + 1).int().tolist()
                
        super().__init__(latent_variables, item_categories)

        self.separate = separate

        if knots is None:
            knots = torch.tensor([-1.7, -0.7, 0, 0.7, 1.7])
        else:
            knots = torch.tensor(knots)

        self.basis = None
        knots = knots.sigmoid()
        knots = torch.cat((torch.zeros(degree+1), knots, torch.ones(degree+1)))
        self.register_buffer('knots', knots)
        self.n_bases = len(knots) - degree - 1
        self.degree = degree

        if separate == "items":
            self.coefficients = torch.nn.Parameter(torch.full((self.latent_variables, self.n_bases, self.items, 1), -2.5))
            self.register_buffer("gpc_spline_multiplier", torch.arange(0, self.max_item_responses).view(1, 1, 1, -1))
        else:
            # (lv, n_bases, items, max(item_categories))
            # with True where splines are supposed to be
            spline_mask = torch.ones((1, self.n_bases, self.items, max(item_categories)), dtype=bool)
            spline_mask[:, :, :, 0] = False
            # remove entries for items with fewer possible responses
            for item, item_cat in enumerate(item_categories):
                spline_mask[:, :, item, item_cat:] = False

            self.register_buffer('spline_mask', spline_mask)
            self.coefficients = torch.nn.Parameter(torch.full((spline_mask.sum(), ), -2.5))

        # (1, items, max(item_categories))
        missing_categories = torch.zeros(self.items, self.max_item_responses, dtype=torch.int)
        for item, item_cat in enumerate(self.item_categories):
            missing_categories[item, item_cat:self.max_item_responses] = 1

        free_bias = (1 - missing_categories).bool()
        free_bias[:, 0] = False
        self.register_buffer("missing_categories", missing_categories.bool())
        self.register_buffer("free_bias", free_bias)
        self.bias_param = nn.Parameter(torch.zeros(free_bias.sum()))


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
            3D tensor (respondents, items, item categories).
        """
        if self.basis is not None: # use precomputed basis if available
            basis = self.basis
        else:
            rescaled_theta = theta.sigmoid()
            basis = BSplineBasisFunction.apply(rescaled_theta.T.flatten(), self.knots, self.degree)
            basis = basis.view(self.latent_variables, -1, self.n_bases)  # (lv, batch, basis)

        if self.separate == "items":
            # (lv, basis, items, 1)
            all_coef = torch.nn.functional.softplus(self.coefficients)
            all_coef = all_coef.cumsum(dim=1) # monotonicity
            # (lv, batch, items, 1)
            splines = torch.einsum('...pb,...bic->...pic', basis, all_coef)
            output = splines.repeat_interleave(self.max_item_responses, dim=3)
            output = output * self.gpc_spline_multiplier
            output = output.sum(dim=0)
        else:
            all_coef = torch.zeros(self.latent_variables, self.n_bases, self.items, self.max_item_responses, device=theta.device)
            # (lv, basis, items, max(self.item_categories))
            all_coef[self.spline_mask] = torch.nn.functional.softplus(self.coefficients)
            all_coef = all_coef.cumsum(dim=1) # monotonicity
            # (lv, batch, items, max(self.item_categories))
            splines = torch.einsum('...pb,...bic->...pic', basis, all_coef)
            output = splines.cumsum(dim=3) # sum splines over categories (instead of gpc_multiplier)
            output = output.sum(dim=0)

        bias = torch.zeros((1, self.items, self.max_item_responses), device=theta.device)
        bias[:, self.free_bias] = self.bias_param
        output += bias
        output[:, self.missing_categories] = -torch.inf
        return output

    def probabilities_from_output(self, output: torch.Tensor) -> torch.Tensor:
        """
        Compute item probabilities from the output tensor.

        Parameters
        ----------
        output : torch.Tensor
            Output tensor from the forward pass.

        Returns
        -------
        torch.Tensor
            3D torch tensor with dimensions (respondents, items, item categories).
        """
        return nn.functional.softmax(output, dim=2)


    def item_theta_relationship_directions(self, *args) -> torch.Tensor:
        """
        Get the relationship directions between each item and latent variable for a fitted model.

        Returns
        -------
        torch.Tensor
            A 2D tensor with the relationships between the items and latent variables. Items are rows and latent variables are columns.
        """
        return torch.ones(self.items, self.latent_variables).int()


    def precompute_basis(self, theta: torch.Tensor):
        """Precompute the B-spline basis functions for the given theta values. Primarily used by certain fitting algorithms.

        Parameters
        ----------
        theta : torch.Tensor
            A 2D tensor with latent variables. Rows are respondents and latent variables are columns.
        """
        rescaled_theta = theta.sigmoid()
        basis = BSplineBasisFunction.apply(rescaled_theta.T.flatten(), self.knots, self.degree)
        self.basis = basis.view(self.latent_variables, -1, self.n_bases)
