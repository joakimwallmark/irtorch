import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class Spline(nn.Module):
    """
    This module implements a B-spline.

    Parameters
    ----------
    variables : int
        The number of variables (dimensions) to transform.
    num_bins : int, optional
        The number of bins to use for the spline (default is 30).
    lower_input_bound : float, optional
        The left boundary of the transformation interval (default is 0.0).
    upper_input_bound : float, optional
        The right boundary of the transformation interval (default is 1.0).
    lower_output_bound : float, optional
        The bottom boundary of the transformation interval (default is 0.0).
    upper_output_bound : float, optional
        The top boundary of the transformation interval (default is 1.0).
    min_bin_width : float, optional
        The minimum width of each bin (default is 1e-3).
    min_bin_height : float, optional
        The minimum height of each bin (default is 1e-3).
    min_derivative : float, optional
        The minimum derivative value at the knots (default is 1e-3).
    free_endpoints : bool, optional
        When free_endpoints=True, the lower and upper output bounds become trainable parameters (default is False).
    """
    def __init__(
        self,
        variables: int,
        num_bins=30,
        lower_input_bound=0.0,
        upper_input_bound=1.0,
        lower_output_bound=0.0, upper_output_bound=1.0,
        min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3,
        free_endpoints=False
    ):
        super().__init__()
        self.num_bins = num_bins
        self.lower_input_bound = lower_input_bound
        self.upper_input_bound = upper_input_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.free_endpoints = free_endpoints

        # Define output bounds as learnable parameters if free_endpoints==True.
        if free_endpoints:
            self.lower_output_bound = nn.Parameter(torch.tensor(lower_output_bound, dtype=torch.float))
            self.upper_output_bound = nn.Parameter(torch.tensor(upper_output_bound, dtype=torch.float))
        else:
            self.register_buffer("lower_output_bound", torch.tensor(lower_output_bound, dtype=torch.float))
            self.register_buffer("upper_output_bound", torch.tensor(upper_output_bound, dtype=torch.float))
        
        # Use the original bounds to compute a default derivative.
        default_deriv = (upper_output_bound - lower_output_bound) / (upper_input_bound - lower_input_bound)
        self.deriv_outside_lower = default_deriv
        self.deriv_outside_upper = default_deriv

        # Unnormalized spline parameters.
        self.unnormalized_widths = nn.Parameter(torch.zeros(variables, num_bins))
        self.unnormalized_heights = nn.Parameter(torch.zeros(variables, num_bins))
        self.unnormalized_derivatives = nn.Parameter(torch.full((variables, num_bins - 1), 0.5405))

    def bspline_basis(self, x, knots, deg):
        r"""
        Compute B-spline basis functions of a given degree at specified points.

        Parameters
        ----------
        x : array_like, shape (n_points,)
            The input points where the basis functions are evaluated.
        knots : array_like, shape (n_knots,)
            The non-decreasing sequence of knots.
        deg : int
            The degree of the B-spline basis functions (non-negative integer).

        Returns
        -------
        B : ndarray, shape (n_points, n_bases)
            A 2D array where each column corresponds to a B-spline basis function
            evaluated at the points in `x`.

        Notes
        -----
        The number of B-spline basis functions, :math:`n_{\text{bases}}`, is determined
        by the number of knots and the degree:

        .. math::

            n_{\text{bases}} = \text{len(knots)} - \text{deg} - 1

        The basis functions are computed using the Cox-de Boor recursion formula.

        **Cox-de Boor recursion formula:**

        The B-spline basis functions of degree :math:`p` are defined recursively as:

        **Base case (degree 0):**

        .. math::

            N_{i,0}(x) = 
            \begin{cases}
                1, & \text{if } t_i \leq x < t_{i+1}, \\
                0, & \text{otherwise}.
            \end{cases}

        **Recursive case:**

        For degrees :math:`p \geq 1`:

        .. math::

            N_{i,p}(x) = \frac{x - t_i}{t_{i+p} - t_i} N_{i,p-1}(x) + \frac{t_{i+p+1} - x}{t_{i+p+1} - t_{i+1}} N_{i+1,p-1}(x)

        If a denominator is zero, the corresponding term is defined to be zero to maintain numerical stability.

        See Also
        --------
        bspline_basis_function : Compute an individual B-spline basis function.

        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> x = np.linspace(0, 1, 100)
        >>> knots = np.linspace(0, 1, 10)
        >>> deg = 3
        >>> B = bspline_basis(x, knots, deg)
        >>> for i in range(B.shape[1]):
        ...     plt.plot(x, B[:, i], label=f'Basis {i}')
        >>> plt.legend()
        >>> plt.xlabel('x')
        >>> plt.ylabel('Basis function value')
        >>> plt.title('B-spline Basis Functions')
        >>> plt.show()
        """
        n_bases = len(knots) - deg - 1
        B = np.zeros((len(x), n_bases))
        for i in range(n_bases):
            B[:, i] = self.bspline_basis_function(x, knots, deg, i)
        return B

    def bspline_basis_function(self, x, knots, deg, i):
        """
        Compute an individual B-spline basis function using the recursive Cox-de Boor formula.

        Parameters
        ----------
        x : array_like, shape (n_points,)
            The input points where the basis function is evaluated.
        knots : array_like, shape (n_knots,)
            The non-decreasing sequence of knots.
        deg : int
            The degree of the B-spline basis function (non-negative integer).
        i : int
            The index of the basis function (0 ≤ i < n_bases).

        Returns
        -------
        N : ndarray, shape (n_points,)
            The values of the B-spline basis function :math:`N_{i,p}(x)` evaluated at points `x`.
        """
        if deg == 0:
            # Handle edge case at the last knot
            if i == len(knots) - 2:
                return np.where((knots[i] <= x) & (x <= knots[i+1]), 1.0, 0.0)
            return np.where((knots[i] <= x) & (x < knots[i+1]), 1.0, 0.0)
        
        # Pre-compute terms for better efficiency
        term1 = np.zeros_like(x, dtype=float)
        term2 = np.zeros_like(x, dtype=float)
        
        denom1 = knots[i+deg] - knots[i]
        if denom1 > 1e-10:  # Numerical stability threshold
            term1 = ((x - knots[i]) / denom1) * self.bspline_basis_function(x, knots, deg-1, i)
            
        denom2 = knots[i+deg+1] - knots[i+1]
        if denom2 > 1e-10:  # Numerical stability threshold
            term2 = ((knots[i+deg+1] - x) / denom2) * self.bspline_basis_function(x, knots, deg-1, i+1)
            
        return term1 + term2








    def forward(self, inputs, inverse=False):
        # Expand parameters to match the batch dimension.

    @torch.no_grad()
    def plot_spline(self, num_points: int = 1000, spline_idx: int = 0):
        x = torch.linspace(self.lower_input_bound, self.upper_input_bound, num_points)
        x_expanded = x.unsqueeze(-1).repeat(1, self.unnormalized_widths.shape[0])
        y, _ = self.forward(x_expanded)
        y = y[:, spline_idx]

        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        
        df = pd.DataFrame({"x": x_np, "y": y_np})
        fig = px.line(df, x="x", y="y", title="Rational Quadratic Spline")
        
        cumwidths, _ = self._compute_cumparams(
            self.unnormalized_widths, self.lower_input_bound, self.upper_input_bound, self.min_bin_width
        )
        lower_output = self.lower_output_bound
        upper_output = self.upper_output_bound
        cumheights, _ = self._compute_cumparams(
            self.unnormalized_heights, lower_output, upper_output, self.min_bin_height
        )
        
        cumwidths_np = cumwidths[spline_idx].detach().cpu().numpy().flatten()
        cumheights_np = cumheights[spline_idx].detach().cpu().numpy().flatten()

        fig.add_trace(go.Scatter(
            x=cumwidths_np,
            y=cumheights_np,
            mode='markers',
            name='Spline Knots'
        ))
        return fig
