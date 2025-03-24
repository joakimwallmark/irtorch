import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union

class CPspline(nn.Module):
    """
    Constrained P-spline for fitting smooth curves to data.

    Parameters
    ----------
    deg : int, default=3
        Degree of the B-spline basis functions.
    ord_d : int, default=2
        Refers to the degree of smoothness enforcement in the penalized spline regression.
        It determines which derivative (or finite difference) of the spline coefficients is penalized to control the smoothness of the fitted spline curve.
        Order of the difference penalty matrix.
    n_int : int, default=40
        Number of intervals for the B-spline basis functions. 
    x_range : tuple, optional
        Tuple specifying the range of the input variable.
    int_constraints : dict, optional
        Dictionary specifying interval constraints for derivatives.
        The keys are the derivative orders and the values are dictionaries
        with keys as constraint senses ('+' or '-') and values as threshold values.
    pt_constraints : dict, optional
        Dictionary specifying point constraints for derivatives.
        The keys are the derivative orders and the values are dictionaries
        with keys as constraint senses ('greaterThan', 'lessThan', 'equalsTo')
        and values as pandas DataFrames with columns 'x' and 'y'.
    pdf_constraint : bool, default=False
        Whether to enforce non-negativity and integration to 1 constraints.
    """
    def __init__(
        self,
        deg: int = 3,
        ord_d: int = 2,
        n_int: int = 40,
        x_range: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
        int_constraints: Optional[Dict[int, Dict[str, Union[int, float]]]] = None,
        pt_constraints: Optional[Dict[int, Dict[str, pd.DataFrame]]] = None,
        pdf_constraint: bool = False,
    ):
        super().__init__()
        # Parameter validation
        if deg < 0:
            raise ValueError("Degree must be non-negative")
        if ord_d < 0:
            raise ValueError("Order of difference must be non-negative")
        if n_int <= 0:
            raise ValueError("Number of intervals must be positive")
        if x_range is not None:
            if x_range[1] <= x_range[0]:
                raise ValueError("x_range[1] must be greater than x_range[0]")
        
        self.deg = deg
        self.ord_d = ord_d
        self.n_int = n_int
        self.x_range = x_range
        self.int_constraints = int_constraints
        self.pt_constraints = pt_constraints
        self.pdf_constraint = pdf_constraint
        self.knots = None
        self.theta = None
        self.basis_matrix = None
        self.penalty_matrix = None
        self.smoothing_param_unconstrained = None

    def generate_knots(self, x_min, x_max):
        """
        Generate knot sequence with n_int intervals and degree deg.
        """
        knots = np.linspace(x_min, x_max, self.n_int + 1)
        # Add deg knots at the start and end for clamping
        knots = np.concatenate((
            np.full(self.deg, x_min),
            knots,
            np.full(self.deg, x_max)
        ))
        return knots

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

    def difference_matrix(self, n_bases, ord_d):
        """
        Create a difference matrix of order ord_d for penalty computation.
        """
        return np.diff(np.eye(n_bases), n=ord_d, axis=0)

    def fit(self, data: pd.DataFrame, y_col: str, num_epochs: int = 1000, lr: float = 0.04,
            constraint_penalty_weight: float = 1.0, smoothing_penalty_weight: float = 1.0,
            early_stopping_patience: int = 150, early_stopping_tol: float = 1e-8):
        """
        Fit the model to the provided data with early stopping.
        """
        x_data = data.drop(columns=y_col).values.flatten()
        y_data = data[y_col].values.flatten()
        y_data = torch.tensor(y_data, dtype=torch.float32)
        
        # Data validation
        if len(x_data) < self.deg + 1:
            raise ValueError(f"Need at least {self.deg + 1} data points for degree {self.deg}")
        
        if self.x_range is not None:
            x_min, x_max = self.x_range
        else:
            x_min, x_max = x_data.min(), x_data.max()
            # Add small padding to avoid boundary issues
            padding = 0.01 * (x_max - x_min)
            x_min -= padding
            x_max += padding
            self.x_range = (x_min, x_max)

        self.knots = self.generate_knots(x_min, x_max)
        
        # Setup matrices
        b_spline_basis_matrix = self.bspline_basis(x_data, self.knots, self.deg)
        n_bases = b_spline_basis_matrix.shape[1]
        b_spline_basis_matrix = torch.tensor(b_spline_basis_matrix, dtype=torch.float32)
        
        difference_matrix = self.difference_matrix(n_bases, self.ord_d)
        penalty_matrix = difference_matrix.T @ difference_matrix
        penalty_matrix = torch.tensor(penalty_matrix, dtype=torch.float32)
        
        # Initialize parameters
        if self.pdf_constraint:
            # Initialize theta to ensure the spline starts close to being a valid PDF
            mean_y = torch.mean(y_data)
            self.theta = nn.Parameter(torch.full((n_bases,), mean_y/n_bases, dtype=torch.float32))
        else:
            self.theta = nn.Parameter(torch.zeros(n_bases, dtype=torch.float32))
        
        self.smoothing_param_unconstrained = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        optimizer = optim.Adam([self.theta, self.smoothing_param_unconstrained], lr=lr)
        
        # Early stopping setup
        best_loss = float('inf')
        best_theta = None
        best_smoothing_param = None
        patience_counter = 0
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            y_pred = b_spline_basis_matrix @ self.theta
            
            data_loss = torch.mean((y_data - y_pred) ** 2)
            smoothing_param = torch.nn.functional.softplus(self.smoothing_param_unconstrained)
            smoothing_penalty = smoothing_param * (self.theta @ penalty_matrix @ self.theta)
            constraint_penalty = self.compute_constraint_penalty(b_spline_basis_matrix)
            
            loss = (data_loss + 
                    constraint_penalty_weight * constraint_penalty + 
                    smoothing_penalty_weight * smoothing_penalty)
            
            loss.backward()
            optimizer.step()
            
            # Early stopping check
            current_loss = loss.item()
            if current_loss < best_loss - early_stopping_tol:
                best_loss = current_loss
                best_theta = self.theta.detach().clone()
                best_smoothing_param = smoothing_param.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {current_loss:.6f}, "
                      f"Smoothing Param: {smoothing_param.item():.6f}")
        
        # Restore best parameters
        if best_theta is not None:
            self.theta.data = best_theta
            # Convert float to tensor before applying operations
            best_smoothing_param_tensor = torch.tensor(best_smoothing_param, dtype=torch.float32)
            self.smoothing_param_unconstrained.data = torch.log(torch.exp(best_smoothing_param_tensor) - 1)
        
        self.basis_matrix = b_spline_basis_matrix.detach()

    def compute_constraint_penalty(self, B):
        """
        Compute penalties for constraints and add them to the loss.
        """
        penalty = 0.0

        # Interval constraints
        if self.int_constraints is not None:
            # For derivative orders specified in int_constraints
            for deriv_order, constraints in self.int_constraints.items():
                # Compute derivative of B-spline basis
                B_deriv = self.bspline_basis_derivative(B.shape[1], deriv_order)
                B_deriv = torch.tensor(B_deriv, dtype=torch.float32)
                y_deriv = B_deriv @ self.theta

                # Apply constraints
                for sense, threshold in constraints.items():
                    if sense == '+':
                        # Enforce y_deriv >= threshold
                        penalty += torch.mean(torch.relu(threshold - y_deriv))
                    elif sense == '-':
                        # Enforce y_deriv <= threshold
                        penalty += torch.mean(torch.relu(y_deriv - threshold))
                    else:
                        raise ValueError(f"Unknown constraint sense '{sense}'.")

        # Point constraints
        if self.pt_constraints is not None:
            for deriv_order, constraints in self.pt_constraints.items():
                for sense, df in constraints.items():
                    x_points = df.drop(columns=['y']).values.flatten()
                    y_values = df['y'].values.flatten()

                    # Compute B-spline basis at x_points
                    B_points = self.bspline_basis(x_points, self.knots, self.deg)
                    if deriv_order > 0:
                        B_points = self.bspline_basis_derivative(B_points.shape[1], deriv_order, x=x_points)

                    B_points = torch.tensor(B_points, dtype=torch.float32)
                    y_pred_points = B_points @ self.theta

                    # Apply constraints
                    y_values = torch.tensor(y_values, dtype=torch.float32)
                    if sense == 'greaterThan':
                        penalty += torch.mean(torch.relu(y_values - y_pred_points))
                    elif sense == 'lessThan':
                        penalty += torch.mean(torch.relu(y_pred_points - y_values))
                    elif sense == 'equalsTo':
                        penalty += torch.mean((y_pred_points - y_values) ** 2)
                    else:
                        raise ValueError(f"Unknown constraint sense '{sense}'.")

        if self.pdf_constraint:
            # Non-negativity constraint using the input basis matrix B
            penalty += torch.mean(torch.relu(-(B @ self.theta)))
            
            # Integration constraint (approximate integration using trapezoidal rule)
            # CURRENT IMPLEMENTATION (problematic):
            # dx = 1.0 / (B.shape[0] - 1)
            # integral = torch.sum(B @ self.theta) * dx
            
            # FIXED IMPLEMENTATION:
            x = torch.linspace(self.knots[0], self.knots[-1], B.shape[0])
            dx = x[1] - x[0]
            y_values = B @ self.theta
            # Trapezoidal rule: 0.5 * (y[0] + y[-1] + 2 * sum(y[1:-1])) * dx
            integral = 0.5 * (y_values[0] + y_values[-1] + 2 * torch.sum(y_values[1:-1])) * dx
            
            # Stronger penalty for deviation from 1
            penalty += 10.0 * (integral - 1.0) ** 2

        return penalty

    def bspline_basis_derivative(self, n_bases, deriv_order, x=None):
        """
        Compute derivative of B-spline basis functions.
        """
        if x is None:
            x = np.linspace(self.knots[self.deg], self.knots[-self.deg-1], 100)
        B_deriv = np.zeros((len(x), n_bases))
        for i in range(n_bases):
            B_deriv[:, i] = self.bspline_basis_derivative_function(x, self.knots, self.deg, i, deriv_order)
        return B_deriv

    def bspline_basis_derivative_function(self, x, knots, deg, i, deriv_order):
        """
        Compute derivative of B-spline basis function using recursive formula.
        """
        if deriv_order == 0:
            return self.bspline_basis_function(x, knots, deg, i)
        else:
            denom1 = knots[i+deg] - knots[i]
            denom2 = knots[i+deg+1] - knots[i+1]
            term1 = 0.0
            term2 = 0.0
            if denom1 > 0:
                term1 = deg / denom1 * self.bspline_basis_derivative_function(x, knots, deg-1, i, deriv_order - 1)
            if denom2 > 0:
                term2 = -deg / denom2 * self.bspline_basis_derivative_function(x, knots, deg-1, i+1, deriv_order - 1)
            return term1 + term2

    def predict(self, data: Union[pd.Series, pd.DataFrame]) -> np.ndarray:
        """
        Generate predictions for the input data.
        """
        if data.empty:
            return np.array([])

        if isinstance(data, pd.Series):
            x_new = data.values.flatten()
        elif isinstance(data, pd.DataFrame):
            x_new = data.values.flatten()
        else:
            raise ValueError("Input data must be a pandas Series or DataFrame.")

        B_new = self.bspline_basis(x_new, self.knots, self.deg)
        B_new = torch.tensor(B_new, dtype=torch.float32)

        with torch.no_grad():
            y_pred = B_new @ self.theta
        return y_pred.numpy()

    def evaluate_derivative(self, data: Union[pd.Series, pd.DataFrame], derivative_order: int = 1) -> np.ndarray:
        """
        Evaluate the derivative of the fitted spline at given points.

        Parameters
        ----------
        data : Union[pd.Series, pd.DataFrame]
            The input data points where to evaluate the derivative.
        derivative_order : int, default=1
            The order of the derivative to compute.

        Returns
        -------
        np.ndarray
            The values of the specified derivative at the input points.
        """
        if data.empty:
            return np.array([])

        if isinstance(data, pd.Series):
            x_new = data.values.flatten()
        elif isinstance(data, pd.DataFrame):
            x_new = data.values.flatten()
        else:
            raise ValueError("Input data must be a pandas Series or DataFrame.")

        # Compute derivative of B-spline basis at new points
        B_deriv = self.bspline_basis_derivative(self.theta.shape[0], derivative_order, x=x_new)
        B_deriv = torch.tensor(B_deriv, dtype=torch.float32)

        with torch.no_grad():
            y_deriv = B_deriv @ self.theta
        return y_deriv.numpy()
