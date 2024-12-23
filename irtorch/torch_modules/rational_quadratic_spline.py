import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class RationalQuadraticSpline(nn.Module):
    """
    This module implements a rational quadratic spline, as described in the paper
    "Neural Spline Flows" by :cite:t:`Durkan2019`.

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
    derivative_outside_lower_input_bound : float, optional
        The derivative value outside the lower input bound (default is None).
    derivative_outside_upper_input_bound : float, optional
        The derivative value outside the upper input bound (default is None).
    """
    def __init__(
        self,
        variables: int,
        num_bins=30,
        lower_input_bound=0.0,
        upper_input_bound=1.0,
        lower_output_bound=0.0,
        upper_output_bound=1.0,
        min_bin_width=1e-3,
        min_bin_height=1e-3,
        min_derivative=1e-3,
        derivative_outside_lower_input_bound=None,
        derivative_outside_upper_input_bound=None,
    ):
        super(RationalQuadraticSpline, self).__init__()
        self.num_bins = num_bins
        self.lower_input_bound = lower_input_bound
        self.upper_input_bound = upper_input_bound
        self.lower_output_bound = lower_output_bound
        self.upper_output_bound = upper_output_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        if derivative_outside_lower_input_bound is None:
            self.derivative_outside_lower_input_bound = \
            (upper_output_bound-lower_output_bound) / (upper_input_bound-lower_input_bound)
        else:
            self.derivative_outside_lower_input_bound = derivative_outside_lower_input_bound
        if derivative_outside_upper_input_bound is None:
            self.derivative_outside_upper_input_bound = \
            (upper_output_bound-lower_output_bound) / (upper_input_bound-lower_input_bound)
        else:
            self.derivative_outside_upper_input_bound = derivative_outside_upper_input_bound

        self.unnormalized_widths = nn.Parameter(torch.rand(variables, num_bins))
        self.unnormalized_heights = nn.Parameter(torch.rand(variables, num_bins))
        self.unnormalized_derivatives = nn.Parameter(torch.rand(variables, num_bins - 1))

    def forward(self, inputs, inverse=False):
        """
        Apply the rational quadratic spline transformation.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor to transform.
        inverse : bool, optional
            If True, computes the inverse transformation (default is False).

        Returns
        -------
        outputs : torch.Tensor
            The transformed outputs.
        logabsdet : torch.Tensor
            The logarithm of the absolute value of the determinant of the Jacobian.
        """
        return self._unconstrained_rational_quadratic_spline(
            inputs,
            self.unnormalized_widths.unsqueeze(0).expand(inputs.shape[0], *self.unnormalized_widths.shape),
            self.unnormalized_heights.unsqueeze(0).expand(inputs.shape[0], *self.unnormalized_heights.shape),
            self.unnormalized_derivatives.unsqueeze(0).expand(inputs.shape[0], *self.unnormalized_derivatives.shape),
            inverse=inverse,
        )

    def _unconstrained_rational_quadratic_spline(
        self,
        inputs,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=False,
    ):
        """
        Compute the unconstrained rational quadratic spline transformation (linear relationship outside the interval).

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor to transform.
        unnormalized_widths : torch.Tensor
            The unnormalized widths for the bins.
        unnormalized_heights : torch.Tensor
            The unnormalized heights for the bins.
        unnormalized_derivatives : torch.Tensor
            The unnormalized derivatives at the knots.
        inverse : bool, optional
            If True, computes the inverse transformation (default is False).

        Returns
        -------
        outputs : torch.Tensor
            The transformed outputs.
        logabsdet : torch.Tensor
            The logarithm of the absolute value of the determinant of the Jacobian.
        """
        # Create masks for input intervals
        if inverse:
            below_lower_bound_mask = inputs < self.lower_output_bound
            above_upper_bound_mask = inputs > self.upper_output_bound
        else:
            below_lower_bound_mask = inputs < self.lower_input_bound
            above_upper_bound_mask = inputs > self.upper_input_bound
        inside_interval_mask = ~below_lower_bound_mask & ~above_upper_bound_mask
        outside_interval_mask = ~inside_interval_mask

        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros_like(inputs)

        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - self.min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        if inverse:
            outputs = torch.where(
                below_lower_bound_mask,
                self.lower_input_bound + (inputs - self.lower_output_bound) / self.derivative_outside_lower_input_bound,
                outputs
            )
            outputs = torch.where(
                above_upper_bound_mask,
                self.upper_input_bound + (inputs - self.upper_output_bound) / self.derivative_outside_upper_input_bound,
                outputs
            )
        else:
            outputs = torch.where(
                below_lower_bound_mask,
                self.lower_output_bound + (inputs - self.lower_input_bound) * self.derivative_outside_lower_input_bound,
                outputs
            )
            outputs = torch.where(
                above_upper_bound_mask,
                self.upper_output_bound + (inputs - self.upper_input_bound) * self.derivative_outside_upper_input_bound,
                outputs
            )
        logabsdet = torch.where(
            outside_interval_mask,
            torch.zeros_like(logabsdet),
            logabsdet
        )

        if torch.any(inside_interval_mask):
            (
                outputs[inside_interval_mask],
                logabsdet[inside_interval_mask],
            ) = self._rational_quadratic_spline(
                inputs=inputs[inside_interval_mask],
                unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
                unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
                unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
                inverse=inverse
            )

        return outputs, logabsdet

    def _rational_quadratic_spline(
        self,
        inputs,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=False
    ):
        """
        Compute the rational quadratic spline transformation.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor to transform.
        unnormalized_widths : torch.Tensor
            The unnormalized widths for the bins.
        unnormalized_heights : torch.Tensor
            The unnormalized heights for the bins.
        unnormalized_derivatives : torch.Tensor
            The unnormalized derivatives at the knots.
        inverse : bool, optional
            If True, computes the inverse transformation (default is False).

        Returns
        -------
        outputs : torch.Tensor
            The transformed outputs.
        logabsdet : torch.Tensor
            The logarithm of the absolute value of the determinant of the Jacobian.

        Raises
        ------
        ValueError
            If minimal bin width or height is too large for the number of bins.
        """
        if torch.min(inputs) < self.lower_input_bound or torch.max(inputs) > self.upper_input_bound:
            pass

        num_bins = unnormalized_widths.shape[-1]

        if self.min_bin_width * num_bins > 1.0:
            raise ValueError("Minimal bin width too large for the number of bins")
        if self.min_bin_height * num_bins > 1.0:
            raise ValueError("Minimal bin height too large for the number of bins")

        widths = F.softmax(unnormalized_widths, dim=-1)
        widths = self.min_bin_width + (1 - self.min_bin_width * num_bins) * widths
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
        cumwidths = (self.upper_input_bound - self.lower_input_bound) * cumwidths + self.lower_input_bound
        cumwidths[..., 0] = self.lower_input_bound
        cumwidths[..., -1] = self.upper_input_bound
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]

        derivatives = self.min_derivative + F.softplus(unnormalized_derivatives)

        heights = F.softmax(unnormalized_heights, dim=-1)
        heights = self.min_bin_height + (1 - self.min_bin_height * num_bins) * heights
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
        cumheights = (self.upper_output_bound - self.lower_output_bound) * cumheights + self.lower_output_bound
        cumheights[..., 0] = self.lower_output_bound
        cumheights[..., -1] = self.upper_output_bound
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        if inverse:
            cumheights[..., -1] += 1e-6
            bin_idx = (torch.sum(inputs[..., None] >= cumheights, dim=-1) - 1)[..., None]
        else:
            cumwidths[..., -1] += 1e-6
            bin_idx = (torch.sum(inputs[..., None] >= cumwidths, dim=-1) - 1)[..., None]

        input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
        input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

        input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
        delta = heights / widths
        input_delta = delta.gather(-1, bin_idx)[..., 0]

        input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
        input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

        input_heights = heights.gather(-1, bin_idx)[..., 0]

        if inverse:
            a = (inputs - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta
            ) + input_heights * (input_delta - input_derivatives)
            b = input_heights * input_derivatives - (inputs - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta
            )
            c = -input_delta * (inputs - input_cumheights)

            discriminant = b.pow(2) - 4 * a * c
            assert (discriminant >= 0).all()

            root = (2 * c) / (-b - torch.sqrt(discriminant))
            outputs = root * input_bin_widths + input_cumwidths

            theta_one_minus_theta = root * (1 - root)
            denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                * theta_one_minus_theta
            )
            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * root.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - root).pow(2)
            )
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

            return outputs, -logabsdet
        else:
            theta = (inputs - input_cumwidths) / input_bin_widths
            theta_one_minus_theta = theta * (1 - theta)

            numerator = input_heights * (
                input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
            )
            denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                * theta_one_minus_theta
            )
            outputs = input_cumheights + numerator / denominator

            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * theta.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - theta).pow(2)
            )
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

            return outputs, logabsdet

    @torch.no_grad()
    def plot_spline(self, num_points=1000, spline_idx=0):
        """
        Plot the rational quadratic spline.

        Parameters
        ----------
        spline_idx : int, optional
            The index of the spline to plot (default is 0).
        num_points : int, optional
            The number of points to use for the plot (default is 1000).
        """
        x = torch.linspace(self.lower_input_bound, self.upper_input_bound, num_points)
        y, _ = self.forward(x.unsqueeze(-1).repeat(1, self.unnormalized_widths.shape[0]))
        y = y[:, spline_idx]
        df = pd.DataFrame({
            "x": x.cpu().detach().numpy() if x.is_cuda else x.detach().numpy(), 
            "y": y.cpu().detach().numpy() if y.is_cuda else y.detach().numpy()
        })
        fig = px.line(df, x="x", y="y")

        widths = F.softmax(self.unnormalized_widths, dim=-1)
        widths = self.min_bin_width + (1 - self.min_bin_width * self.num_bins) * widths
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
        cumwidths = (self.upper_input_bound - self.lower_input_bound) * cumwidths + self.lower_input_bound
        cumwidths[..., 0] = self.lower_input_bound
        cumwidths[..., -1] = self.upper_input_bound

        heights = F.softmax(self.unnormalized_heights, dim=-1)
        heights = self.min_bin_height + (1 - self.min_bin_height * self.num_bins) * heights
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
        cumheights = (self.upper_output_bound - self.lower_output_bound) * cumheights + self.lower_output_bound
        cumheights[..., 0] = self.lower_output_bound
        cumheights[..., -1] = self.upper_output_bound

        cumwidths = cumwidths[spline_idx].cpu().flatten().detach().numpy() if cumwidths.is_cuda else cumwidths[spline_idx].flatten().detach().numpy()
        cumheights = cumheights[spline_idx].cpu().flatten().detach().numpy() if cumheights.is_cuda else cumheights[spline_idx].flatten().detach().numpy()

        fig.add_trace(go.Scatter(
            x=cumwidths,
            y=cumheights,
            mode='markers',
            name='Points'
        ))

        return fig
