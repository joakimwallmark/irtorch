import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

class RationalQuadraticSpline(nn.Module):
    def __init__(
        self,
        variables: int,
        num_bins=30,
        tail_bound=5.0,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE,
    ):
        """
        This module implements a rational quadratic spline, as described in the paper
        "Neural Spline Flows" by :cite:t:`Durkan2019`.

        Parameters
        ----------
        variables : int
            The number of variables (dimensions) to transform.
        num_bins : int, optional
            The number of bins to use for the spline (default is 14).
        tail_bound : float, optional
            The boundary beyond which the transformation is linear (default is 1.0).
        min_bin_width : float, optional
            The minimum width of each bin (default is 1e-3).
        min_bin_height : float, optional
            The minimum height of each bin (default is 1e-3).
        min_derivative : float, optional
            The minimum derivative value at the knots (default is 1e-3).
        """
        super(RationalQuadraticSpline, self).__init__()
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

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
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

    def _unconstrained_rational_quadratic_spline(
        self,
        inputs,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=False,
        tail_bound=1.0,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE,
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
        tail_bound : float, optional
            The boundary beyond which the transformation is linear (default is 1.0).
        min_bin_width : float, optional
            The minimum width of each bin (default is 1e-3).
        min_bin_height : float, optional
            The minimum height of each bin (default is 1e-3).
        min_derivative : float, optional
            The minimum derivative value at the knots (default is 1e-3).

        Returns
        -------
        outputs : torch.Tensor
            The transformed outputs.
        logabsdet : torch.Tensor
            The logarithm of the absolute value of the determinant of the Jacobian.
        """
        inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
        outside_interval_mask = ~inside_interval_mask

        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros_like(inputs)

        # Linear relationships outside the interval
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0

        if torch.any(inside_interval_mask):
            (
                outputs[inside_interval_mask],
                logabsdet[inside_interval_mask],
            ) = self._rational_quadratic_spline(
                inputs=inputs[inside_interval_mask],
                unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
                unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
                unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
                inverse=inverse,
                left=-tail_bound,
                right=tail_bound,
                bottom=-tail_bound,
                top=tail_bound,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
                min_derivative=min_derivative,
            )

        return outputs, logabsdet

    def _rational_quadratic_spline(
        self,
        inputs,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=False,
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=1.0,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE,
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
        left : float, optional
            The left boundary of the transformation interval (default is 0.0).
        right : float, optional
            The right boundary of the transformation interval (default is 1.0).
        bottom : float, optional
            The bottom boundary of the transformation interval (default is 0.0).
        top : float, optional
            The top boundary of the transformation interval (default is 1.0).
        min_bin_width : float, optional
            The minimum width of each bin (default is 1e-3).
        min_bin_height : float, optional
            The minimum height of each bin (default is 1e-3).
        min_derivative : float, optional
            The minimum derivative value at the knots (default is 1e-3).

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
        if torch.min(inputs) < left or torch.max(inputs) > right:
            pass

        num_bins = unnormalized_widths.shape[-1]

        if min_bin_width * num_bins > 1.0:
            raise ValueError("Minimal bin width too large for the number of bins")
        if min_bin_height * num_bins > 1.0:
            raise ValueError("Minimal bin height too large for the number of bins")

        widths = F.softmax(unnormalized_widths, dim=-1)
        widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
        cumwidths = (right - left) * cumwidths + left
        cumwidths[..., 0] = left
        cumwidths[..., -1] = right
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]

        derivatives = min_derivative + F.softplus(unnormalized_derivatives)

        heights = F.softmax(unnormalized_heights, dim=-1)
        heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
        cumheights = (top - bottom) * cumheights + bottom
        cumheights[..., 0] = bottom
        cumheights[..., -1] = top
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
