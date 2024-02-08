import torch


def interp(x_new, x, y):
    """
    Linear interpolation function with simple linear extrapolation at edges

    Args:
        x_new: torch.Tensor
            The x-coordinates at which to evaluate the interpolated values.
        x: torch.Tensor
            The x-coordinates of the data points.
        y: torch.Tensor
            The y-coordinates of the data points.

    Returns:
        torch.Tensor: The interpolated values at x_new.
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

    Args:
        tensor (torch.Tensor): A 2D tensor.

    Returns:
        torch.Tensor: A tensor of the same shape as the input, but with each column transformed to follow a standard normal distribution.
    """
    normal = torch.distributions.Normal(0, 1)

    # Compute the ranks of the data along each column
    ranks = tensor.argsort(dim=0).argsort(dim=0)

    # Compute the empirical CDF
    cdf = (ranks + 1).float() / (tensor.shape[0] + 1)

    # Transform the CDF to a standard normal distribution
    tensor_transformed = normal.icdf(cdf)

    return tensor_transformed

# TODO: Remove unused functions below from this file before publish
# def inverse_quantile_transform(new_values_transformed, tensor, tensor_transformed):
#     """
#     Transforms a 2D tensor from a standard normal distribution back to the original distribution along each column.
#     The columns are variables to be transformed accross the rows.

#     Args:
#         new_values_transformed (torch.Tensor): A 2D tensor in the transformed distribution.
#         tensor (torch.Tensor): The original 2D tensor before transformation.
#         tensor_transformed (torch.Tensor): The transformed original 2D tensor.

#     Returns:
#         torch.Tensor: A tensor of the same shape as the input, but with each column transformed back to the original distribution.
#     """
#     if not (
#         new_values_transformed.device == new_values_transformed.device
#         and new_values_transformed.device == tensor.device
#     ):
#         raise ValueError("All input tensors must be on the same device")
#     new_values = torch.empty_like(new_values_transformed)

#     # Loop over each variable in the tensor
#     for i in range(tensor.shape[1]):
#         column = tensor[:, i]
#         column_transformed = tensor_transformed[:, i]

#         # Sort the columns
#         sorted_column, _ = torch.sort(column)
#         sorted_column_transformed, _ = torch.sort(column_transformed)

#         # Use linear interpolation to estimate the untransformed value for the new value
#         new_values[:, i] = interp(
#             new_values_transformed[:, i], sorted_column_transformed, sorted_column
#         )

#     return new_values


# def tanh_latent_score_grid(
#     z: torch.Tensor,
#     steps: int,
#     inverted_scale_multiplier: torch.Tensor,
#     z_means,
#     z_stds,
# ):
#     """Computes a linear path to z for each element in z using the standardized tanh method

#     Parameters
#     ----------
#     z: torch.Tensor
#         A 2D torch tensor with columns corresponding to each latent variable
#         and rows corresponding to each respondent
#     steps: int
#         The number of steps in each path
#     inverted_scale_multiplier:
#         A torch tensor with -1 for the inverted latent variables and 1 for the rest
#     z_means: torch.Tensor
#         A torch tensor with the means of each latent variable
#     z_stds: torch.Tensor
#         A torch tensor with the standard deviations of each latent variable

#     Returns:
#     ----------
#     mean_grids: torch.Tensor
#         A 2D torch tensor with each z path stacked on top of each other (for later use with decoder)
#     """

#     # get min value(s) from standardized tanh scale
#     min_tanh_z = torch.full((1, z.shape[1]), -0.9999)
#     # min_tanh_z = (
#     #     torch.linspace(-1, 1, steps)[1].repeat(z.shape[1]).unsqueeze(0)
#     # )
#     min_tanh_z *= inverted_scale_multiplier

#     # transform z to standardized tanh scale
#     std_z, _, _ = rescale_latent_variable(z, mean_or_min=z_means, std_or_max=z_stds)
#     tanh_z = torch.tanh(std_z)
#     # get tanh_grid
#     mean_grids = latent_score_grid(tanh_z, steps, min_tanh_z)
#     # transform grid back to z scale
#     mean_grids = torch.atanh(mean_grids)
#     mean_grids = inverse_rescale_latent_variable(
#         mean_grids, "standardize", z_means, z_stds
#     )

#     return mean_grids

# def rescale_latent_variable(
#     latent_variable_tensor, method="standardize", mean_or_min=None, std_or_max=None
# ):
#     """
#     Rescale the latent variable represented by each column in the tensor.

#     Parameters:
#     latent_variable_tensor: torch.Tensor
#         A 2D tensor with the latent variables in the columns.
#     method: str, optional
#         The rescaling method. Can be 'standardize' or 'normalize'. (default is 'standardize')
#     mean_or_min: float, optional
#         The mean used for standardizing. (default is the mean in the latent_variable_tensor)
#         The minimum value used for when normalizing. (default is the minimum value in the latent_variable_tensor)
#     std_or_max: float, optional
#         The standard deviation used for standardizing. (default is the standard deviation in the latent_variable_tensor)
#         The maximum value used for when normalizing. (default is the maximum value in the latent_variable_tensor)

#     Returns:
#     rescaled_tensor: torch.Tensor
#         The rescaled tensor.
#     mean_or_min: torch.Tensor
#         The means (if standardized) or minima (if normalized) used in the rescaling.
#     std_or_max: torch.Tensor
#         The standard deviations (if standardized) or maxima (if normalized) used in the rescaling.
#     """
#     if method == "standardize":
#         if mean_or_min is None:
#             mean_or_min = torch.mean(latent_variable_tensor, dim=0)
#         if std_or_max is None:
#             std_or_max = torch.std(latent_variable_tensor, dim=0)
#         return (latent_variable_tensor - mean_or_min) / std_or_max, mean_or_min, std_or_max
#     elif method == "normalize":
#         if mean_or_min is None:
#             mean_or_min = torch.min(latent_variable_tensor, dim=0)[0]
#         if std_or_max is None:
#             std_or_max = torch.max(latent_variable_tensor, dim=0)[0]
#         return (
#             (latent_variable_tensor - mean_or_min) / (std_or_max - mean_or_min),
#             mean_or_min,
#             std_or_max,
#         )
#     else:
#         raise ValueError("Invalid method. Choose either 'standardize' or 'normalize'.")
    
# def inverse_rescale_latent_variable(latent_variable_tensor, method, mean_or_min, std_or_max):
#     """
#     Apply the inverse transformation of rescale_latent_variable.

#     Parameters:
#     ----------
#     tensor: torch.Tensor
#         A 2D tensor with the rescaled latent variables in the columns.
#     method: str
#         The rescaling method that was used. Must be either 'standardize' or 'normalize'.
#     mean_or_min: torch.Tensor
#         The means (if standardized) or minima (if normalized) that were used in the rescaling.
#     std_or_max: torch.Tensor
#         The standard deviations (if standardized) or maxima (if normalized) that were used in the rescaling.

#     Returns:
#     ----------
#     original_tensor: torch.Tensor
#         The tensor after applying the inverse transformation.
#     """
#     if method == "standardize":
#         return latent_variable_tensor * std_or_max + mean_or_min
#     elif method == "normalize":
#         return latent_variable_tensor * (std_or_max - mean_or_min) + mean_or_min
#     else:
#         raise ValueError("Invalid method. Choose either 'standardize' or 'normalize'.")

# def entropy_distance(entropies: torch.Tensor):
#     """Compute entropy distance as a measure of ability for given latent variable path.

#     Parameters
#     ----------
#     entropies: torch.Tensor
#         3D torch tensor of entropy values where
#         - first dimension are the respondents
#         - second are the items
#         - third is the path

#     Returns:
#     ----------
#     entropy: torch.Tensor
#         1D tensor with the entropy distance for each respondent along the path
#     """
#     # Compute the absolute difference
#     diff = torch.abs(entropies - torch.roll(entropies, shifts=1, dims=2))
#     # Note that for each sub-tensor in the third dimension, the first element will
#     # be the difference with the last element of the previous sub-tensor because
#     # of the roll function. We set this to 0
#     diff[:, :, 0] = 0
#     return diff.sum(dim=(1, 2))


# def latent_score_grid(z, grid_start, grid_points: int):
#     """Compute a linear path from grid_start to z for each element in z

#     Parameters
#     ----------
#     z: torch.Tensor
#         A 2D torch tensor with columns corresponding to each latent variable
#         and rows corresponding to each respondent
#     grid_start: torch.Tensor
#         A 2D torch tensor with one row
#         The starting coordinates of each path
#     grid_points: int
#         The number of points to use for computing entropy distance.

#     Returns:
#     ----------
#     mean_grids: torch.Tensor
#         A 2D torch tensor with each z path stacked on top of each other (for later use with decoder)
#     """
#     mean_grids = torch.empty(z.shape[0] * grid_points, z.shape[1])
#     for person in range(z.shape[0]):
#         mean_grids[person * grid_points : person * grid_points + grid_points, :] = torch.stack(
#             [
#                 torch.linspace(grid_start[0, i], z[person, i], steps=grid_points)
#                 for i in range(z.shape[1])
#             ],
#             dim=1,
#         )

#     return mean_grids