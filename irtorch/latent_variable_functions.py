import logging
import torch

logger = logging.getLogger('irtorch')

def interp(x_new, x, y):
    """
    Linear interpolation function with simple linear extrapolation at edges

    Parameters
    ----------
    x_new : torch.Tensor
        The x-coordinates at which to evaluate the interpolated values.
    x : torch.Tensor
        The x-coordinates of the data points.
    y : torch.Tensor
        The y-coordinates of the data points.

    Returns
    -------
    torch.Tensor
        The interpolated values at x_new.
    """

    x_new = x_new.float()
    x = x.float()
    y = y.float()

    # Sort x and y
    sorted_indices = torch.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]

    # Create a mask for x_new within the range of x
    mask = (x_new >= x[0]) & (x_new <= x[-1])

    y_new = torch.empty_like(x_new)

    # Handle interpolation for x_new within the range of x
    if mask.any():
        # Find indices where x_new should be inserted in x to maintain order
        indices = torch.searchsorted(x, x_new[mask])

        # Compute the fraction of the way that x_new is between x[indices - 1] and x[indices]
        fraction = (x_new[mask] - x[indices - 1]) / (x[indices] - x[indices - 1])

        # Use torch.lerp() to interpolate
        y_new[mask] = torch.lerp(y[indices - 1], y[indices], fraction)

    # Handle extrapolation for x_new outside the range of x
    if (~mask).any():
        # Compute the slopes at the edges for extrapolation
        left_slope = (y[1] - y[0]) / (x[1] - x[0])
        right_slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        y_new[~mask] = torch.where(
            x_new[~mask] < x[0],
            y[0] + left_slope * (x_new[~mask] - x[0]),
            y[-1] + right_slope * (x_new[~mask] - x[-1]),
        )

    return y_new


def quantile_transform(tensor):
    """
    Transforms each column of a 2D tensor to follow a standard normal distribution.

    Parameters
    ----------
        tensor : torch.Tensor
            A 2D tensor.

    Returns
    -------
    torch.Tensor
        A tensor of the same shape as the input, but with each column transformed to follow a standard normal distribution.
    """
    normal = torch.distributions.Normal(0, 1)

    # Compute the ranks of the data along each column
    ranks = tensor.argsort(dim=0).argsort(dim=0)

    # Compute the empirical CDF
    cdf = (ranks + 1).float() / (tensor.shape[0] + 1)

    # Transform the CDF to a standard normal distribution
    tensor_transformed = normal.icdf(cdf)

    return tensor_transformed
