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
        super().__init__()
        self.num_bins = num_bins
        self.lower_input_bound = lower_input_bound
        self.upper_input_bound = upper_input_bound
        self.lower_output_bound = lower_output_bound
        self.upper_output_bound = upper_output_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        default_deriv = (upper_output_bound - lower_output_bound) / (upper_input_bound - lower_input_bound)
        self.derivative_outside_lower_input_bound = (
            default_deriv if derivative_outside_lower_input_bound is None else derivative_outside_lower_input_bound
        )
        self.derivative_outside_upper_input_bound = (
            default_deriv if derivative_outside_upper_input_bound is None else derivative_outside_upper_input_bound
        )

        self.unnormalized_widths = nn.Parameter(torch.rand(variables, num_bins))
        self.unnormalized_heights = nn.Parameter(torch.rand(variables, num_bins))
        self.unnormalized_derivatives = nn.Parameter(torch.rand(variables, num_bins - 1))

    def forward(self, inputs, inverse: bool = False):
        batch_shape = (inputs.shape[0],) + self.unnormalized_widths.shape
        widths = self.unnormalized_widths.unsqueeze(0).expand(batch_shape)
        heights = self.unnormalized_heights.unsqueeze(0).expand(batch_shape)
        derivs = self.unnormalized_derivatives.unsqueeze(0).expand(
            (inputs.shape[0],) + self.unnormalized_derivatives.shape
        )
        return self._unconstrained_rational_quadratic_spline(inputs, widths, heights, derivs, inverse)

    def _compute_cumparams(self, unnormalized, num_bins, lower_bound, upper_bound, min_size):
        params = F.softmax(unnormalized, dim=-1)
        params = min_size + (1 - min_size * num_bins) * params
        cum_params = torch.cumsum(params, dim=-1)
        cum_params = F.pad(cum_params, pad=(1, 0), mode="constant", value=0.0)
        scale = upper_bound - lower_bound
        cum_params = scale * cum_params + lower_bound
        cum_params[..., 0] = lower_bound
        cum_params[..., -1] = upper_bound
        params = cum_params[..., 1:] - cum_params[..., :-1]
        return cum_params, params

    def _unconstrained_rational_quadratic_spline(
        self, inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse: bool = False
    ):
        if inverse:
            lower_bound, upper_bound = self.lower_output_bound, self.upper_output_bound
            out_lower_transform = lambda x: self.lower_input_bound + (x - self.lower_output_bound) / self.derivative_outside_lower_input_bound
            out_upper_transform = lambda x: self.upper_input_bound + (x - self.upper_output_bound) / self.derivative_outside_upper_input_bound
        else:
            lower_bound, upper_bound = self.lower_input_bound, self.upper_input_bound
            out_lower_transform = lambda x: self.lower_output_bound + (x - self.lower_input_bound) * self.derivative_outside_lower_input_bound
            out_upper_transform = lambda x: self.upper_output_bound + (x - self.upper_input_bound) * self.derivative_outside_upper_input_bound

        below_mask = inputs < lower_bound
        above_mask = inputs > upper_bound
        inside_mask = ~(below_mask | above_mask)

        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros_like(inputs)

        constant = np.log(np.exp(1 - self.min_derivative) - 1)
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs = torch.where(below_mask, out_lower_transform(inputs), outputs)
        outputs = torch.where(above_mask, out_upper_transform(inputs), outputs)
        logabsdet = torch.where(~inside_mask, torch.zeros_like(logabsdet), logabsdet)

        if torch.any(inside_mask):
            in_out, in_logdet = self._rational_quadratic_spline(
                inputs=inputs[inside_mask],
                unnormalized_widths=unnormalized_widths[inside_mask, :],
                unnormalized_heights=unnormalized_heights[inside_mask, :],
                unnormalized_derivatives=unnormalized_derivatives[inside_mask, :],
                inverse=inverse,
            )
            outputs[inside_mask] = in_out
            logabsdet[inside_mask] = in_logdet

        return outputs, logabsdet

    def _rational_quadratic_spline(
        self, inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse: bool = False
    ):
        num_bins = unnormalized_widths.shape[-1]
        if self.min_bin_width * num_bins > 1.0:
            raise ValueError("Minimal bin width too large for the number of bins")
        if self.min_bin_height * num_bins > 1.0:
            raise ValueError("Minimal bin height too large for the number of bins")

        cumwidths, widths = self._compute_cumparams(
            unnormalized_widths, num_bins, self.lower_input_bound, self.upper_input_bound, self.min_bin_width
        )
        cumheights, heights = self._compute_cumparams(
            unnormalized_heights, num_bins, self.lower_output_bound, self.upper_output_bound, self.min_bin_height
        )
        derivatives = self.min_derivative + F.softplus(unnormalized_derivatives)
        delta = heights / widths

        eps = 1e-6
        if inverse:
            cum_boundaries = cumheights.clone()
            cum_boundaries[..., -1] += eps
        else:
            cum_boundaries = cumwidths.clone()
            cum_boundaries[..., -1] += eps

        # Replace torch.bucketize (which requires 1D boundaries) with a vectorized comparison.
        bin_idx = (inputs.unsqueeze(-1) >= cum_boundaries).sum(dim=-1) - 1
        bin_idx = bin_idx.unsqueeze(-1)

        input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
        input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
        input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
        input_delta = delta.gather(-1, bin_idx)[..., 0]
        input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
        input_derivatives_plus = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
        input_heights = heights.gather(-1, bin_idx)[..., 0]

        if inverse:
            a = (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus - 2 * input_delta) + \
                input_heights * (input_delta - input_derivatives)
            b = input_heights * input_derivatives - (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus - 2 * input_delta)
            c = -input_delta * (inputs - input_cumheights)
            discriminant = b.pow(2) - 4 * a * c
            if not torch.all(discriminant >= 0):
                raise RuntimeError("Negative discriminant encountered in inverse spline computation.")
            root = (2 * c) / (-b - torch.sqrt(discriminant))
            outputs = root * input_bin_widths + input_cumwidths

            theta_one_minus_theta = root * (1 - root)
            denominator = input_delta + (input_derivatives + input_derivatives_plus - 2 * input_delta) * theta_one_minus_theta
            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus * root.pow(2) +
                2 * input_delta * theta_one_minus_theta +
                input_derivatives * (1 - root).pow(2)
            )
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
            return outputs, -logabsdet
        else:
            theta = (inputs - input_cumwidths) / input_bin_widths
            theta_one_minus_theta = theta * (1 - theta)
            numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
            denominator = input_delta + (input_derivatives + input_derivatives_plus - 2 * input_delta) * theta_one_minus_theta
            outputs = input_cumheights + numerator / denominator

            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus * theta.pow(2) +
                2 * input_delta * theta_one_minus_theta +
                input_derivatives * (1 - theta).pow(2)
            )
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
            return outputs, logabsdet

    @torch.no_grad()
    def plot_spline(self, num_points: int = 1000, spline_idx: int = 0):
        x = torch.linspace(self.lower_input_bound, self.upper_input_bound, num_points)
        x_expanded = x.unsqueeze(-1).repeat(1, self.unnormalized_widths.shape[0])
        y, _ = self.forward(x_expanded)
        y = y[:, spline_idx]

        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        df = pd.DataFrame({"x": x_np, "y": y_np})
        fig = px.line(df, x="x", y="y")

        cumwidths, _ = self._compute_cumparams(
            self.unnormalized_widths, self.num_bins, self.lower_input_bound, self.upper_input_bound, self.min_bin_width
        )
        cumheights, _ = self._compute_cumparams(
            self.unnormalized_heights, self.num_bins, self.lower_output_bound, self.upper_output_bound, self.min_bin_height
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
