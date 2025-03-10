import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class RationalQuadraticSpline(nn.Module):
    """
    A simplified rational quadratic spline with optional learnable output bounds.
    
    When free_endpoints=True, the lower and upper output bounds become trainable.
    """
    def __init__(self, variables: int, num_bins=30,
                 lower_input_bound=0.0, upper_input_bound=1.0,
                 lower_output_bound=0.0, upper_output_bound=1.0,
                 min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3,
                 free_endpoints=False):
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

    def forward(self, inputs, inverse=False):
        # Expand parameters to match the batch dimension.
        bs = inputs.shape[0]
        widths = self.unnormalized_widths.unsqueeze(0).expand(bs, -1, -1)
        heights = self.unnormalized_heights.unsqueeze(0).expand(bs, -1, -1)
        derivs = self.unnormalized_derivatives.unsqueeze(0).expand(bs, -1, -1)
        return self._spline_transform(inputs, widths, heights, derivs, inverse)
    
    def _compute_cumparams(self, unnormalized, lower, upper, min_size):
        params = F.softmax(unnormalized, dim=-1)
        params = min_size + (1 - min_size * self.num_bins) * params
        cum_params = torch.cumsum(params, dim=-1)
        cum_params = F.pad(cum_params, (1, 0), mode="constant", value=0.0)
        scale = upper - lower
        cum_params = scale * cum_params + lower
        return cum_params, cum_params[..., 1:] - cum_params[..., :-1]
    
    def _pad_derivatives(self, derivs):
        # Build a new tensor for padded derivatives without in-place modification.
        const_val = torch.tensor(np.log(np.exp(1 - self.min_derivative) - 1),
                                 dtype=derivs.dtype, device=derivs.device)
        padded = F.pad(derivs, (1, 1))
        first = const_val.expand(*padded.shape[:-1], 1)
        last = const_val.expand(*padded.shape[:-1], 1)
        return torch.cat([first, padded[..., 1:-1], last], dim=-1)
    
    def _spline_transform(self, inputs, widths_raw, heights_raw, derivs_raw, inverse):
        # Set up the out-of-bound transformations.
        if inverse:
            lb = self.lower_output_bound
            ub = self.upper_output_bound
            lower_trans = lambda x: self.lower_input_bound + (x - lb) / self.deriv_outside_lower
            upper_trans = lambda x: self.upper_input_bound + (x - ub) / self.deriv_outside_upper
        else:
            lb, ub = self.lower_input_bound, self.upper_input_bound
            lb_out = self.lower_output_bound
            ub_out = self.upper_output_bound
            lower_trans = lambda x: lb_out + (x - lb) * self.deriv_outside_lower
            upper_trans = lambda x: ub_out + (x - ub) * self.deriv_outside_upper
        
        below = inputs < self.lower_input_bound
        above = inputs > self.upper_input_bound
        inside = ~(below | above)
        
        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros_like(inputs)
        derivs_padded = self._pad_derivatives(derivs_raw)
        
        outputs = torch.where(below, lower_trans(inputs), outputs)
        outputs = torch.where(above, upper_trans(inputs), outputs)
        logabsdet = torch.where(~inside, torch.zeros_like(logabsdet), logabsdet)
        
        if inside.any():
            out_inside, logdet_inside = self._rational_quadratic_spline(
                inputs[inside], widths_raw[inside], heights_raw[inside], derivs_padded[inside], inverse)
            outputs[inside] = out_inside
            logabsdet[inside] = logdet_inside
        return outputs, logabsdet
    
    def _rational_quadratic_spline(self, inputs, widths_raw, heights_raw, derivs, inverse):
        num_bins = widths_raw.shape[-1]
        if self.min_bin_width * self.num_bins > 1.0 or self.min_bin_height * self.num_bins > 1.0:
            raise ValueError("Min bin width/height too large for the number of bins")
        
        cum_widths, widths = self._compute_cumparams(widths_raw, self.lower_input_bound, self.upper_input_bound, self.min_bin_width)
        lb_out = self.lower_output_bound
        ub_out = self.upper_output_bound
        cum_heights, heights = self._compute_cumparams(heights_raw, lb_out, ub_out, self.min_bin_height)
        
        derivs = self.min_derivative + F.softplus(derivs)
        delta = heights / widths
        
        eps = 1e-6
        if inverse:
            cum_boundaries = cum_heights.clone()
            cum_boundaries[..., -1] += eps
        else:
            cum_boundaries = cum_widths.clone()
            cum_boundaries[..., -1] += eps
        
        # Bucketize inputs: clamp indices so they are at most num_bins-1.
        bin_idx = (inputs.unsqueeze(-1) >= cum_boundaries).sum(dim=-1) - 1
        bin_idx = torch.clamp(bin_idx, max=num_bins - 1)
        bin_idx = bin_idx.unsqueeze(-1)
        
        input_cumwidths = cum_widths.gather(-1, bin_idx)[..., 0]
        input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
        input_cumheights = cum_heights.gather(-1, bin_idx)[..., 0]
        input_delta = delta.gather(-1, bin_idx)[..., 0]
        input_derivs = derivs.gather(-1, bin_idx)[..., 0]
        input_derivs_plus = derivs[..., 1:].gather(-1, bin_idx)[..., 0]
        input_heights = heights.gather(-1, bin_idx)[..., 0]
        
        if inverse:
            a = (inputs - input_cumheights) * (input_derivs + input_derivs_plus - 2 * input_delta) \
                + input_heights * (input_delta - input_derivs)
            b = input_heights * input_derivs - (inputs - input_cumheights) * (input_derivs + input_derivs_plus - 2 * input_delta)
            c = -input_delta * (inputs - input_cumheights)
            disc = b.pow(2) - 4 * a * c
            root = (2 * c) / (-b - torch.sqrt(disc))
            outputs = root * input_bin_widths + input_cumwidths
            theta1mt = root * (1 - root)
            denom = input_delta + (input_derivs + input_derivs_plus - 2 * input_delta) * theta1mt
            num = input_delta.pow(2) * (input_derivs_plus * root.pow(2) +
                                         2 * input_delta * theta1mt +
                                         input_derivs * (1 - root).pow(2))
            logabsdet = torch.log(num) - 2 * torch.log(denom)
            return outputs, -logabsdet
        else:
            theta = (inputs - input_cumwidths) / input_bin_widths
            theta1mt = theta * (1 - theta)
            num = input_heights * (input_delta * theta.pow(2) + input_derivs * theta1mt)
            denom = input_delta + (input_derivs + input_derivs_plus - 2 * input_delta) * theta1mt
            outputs = input_cumheights + num / denom
            num_deriv = input_delta.pow(2) * (input_derivs_plus * theta.pow(2) +
                                              2 * input_delta * theta1mt +
                                              input_derivs * (1 - theta).pow(2))
            logabsdet = torch.log(num_deriv) - 2 * torch.log(denom)
            return outputs, logabsdet

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
