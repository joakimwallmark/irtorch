import pytest
import torch
import scipy.stats as stats
from irtorch.latent_variable_functions import (
    # rescale_latent_variable,
    # inverse_rescale_latent_variable,
    # entropy_distance,
    # latent_score_grid,
    # tanh_latent_score_grid,
    interp,
    quantile_transform,
    # inverse_quantile_transform,
)

# TODO: Remove unused functions above from this file

def test_interp():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([0.0, 1.0, 4.0, 9.0, 16.0])  # y = x^2
    x_new = torch.tensor(
        [-1.0, 2.0, 0.6, 1.5, 5.0, 2.5, 3.5]
    )  # Added edge cases -1.0 and 5.0
    y_new_expected = torch.tensor(
        [-1.0, 4.0, 0.6, 2.5, 23.0, 6.5, 12.5]
    )  # Expected values for y_new

    y_new = interp(x_new, x, y)

    assert torch.allclose(
        y_new, y_new_expected, atol=1e-6
    ), "The interp function does not return the expected results"


def test_quantile_transform():
    # Uniform tensor
    data_tensor = torch.arange(1, 1001, step=1).repeat(3, 1).T.float()
    # Shuffle each column
    for i in range(data_tensor.shape[1]):
        data_tensor[:, i] = data_tensor[:, i][torch.randperm(data_tensor.shape[0])]

    tensor_transformed = quantile_transform(data_tensor)
    assert tensor_transformed.shape == data_tensor.shape
    assert torch.allclose(
        tensor_transformed.mean(dim=0),
        torch.tensor([0.0, 0.0, 0.0]),
        atol=1e-6,
    )
    assert torch.allclose(
        tensor_transformed.std(dim=0),
        torch.tensor([1.0, 1.0, 1.0]),
        atol=1e-1,
    )

    # test for normality, should be high
    shapiro_test_p = stats.shapiro(tensor_transformed[:, 0].cpu())[1]
    assert shapiro_test_p > 0.96

    # Multimodal distribution from multiple Gaussians
    means, std_devs = torch.tensor([1.0, 3.0, 5.0]), torch.tensor([0.5, 0.5, 0.5])
    # Define probabilities for each Gaussian and generate a categorical distribution
    probs = torch.tensor([0.3, 0.4, 0.3])
    cat = torch.distributions.Categorical(probs)
    # Generate samples
    data_tensor = (
        torch.stack(
            [
                torch.normal(means[cat.sample()], std_devs[cat.sample()])
                for _ in range(1000 * 3)
            ]
        )
        .reshape(1000, 3)
    )

    tensor_transformed = quantile_transform(data_tensor)
    assert tensor_transformed.shape == data_tensor.shape
    assert torch.allclose(
        tensor_transformed.mean(dim=0),
        torch.tensor([0.0, 0.0, 0.0]),
        atol=1e-6,
    )
    assert torch.allclose(
        tensor_transformed.std(dim=0),
        torch.tensor([1.0, 1.0, 1.0]),
        atol=1e-1,
    )

    # test for normality, should be high
    shapiro_test_p = stats.shapiro(tensor_transformed[:, 0].cpu())[1]
    assert shapiro_test_p > 0.96

    # What about duplicates in a 1D tensor?
    data_tensor = torch.tensor([1, 2, 2, 3, 4])
    tensor_transformed = quantile_transform(data_tensor)
    assert tensor_transformed.shape == data_tensor.shape
    # we expect the duplicates to have different normal values next to each other
    # each obs. corresponds to the same range in normal distribution even if they happen to be the same value
    assert torch.allclose(
        tensor_transformed,
        torch.tensor([-0.9674, -0.4307, 0.0000, 0.4307, 0.9674]),
        atol=1e-4,
    ) or torch.allclose(
        tensor_transformed,
        torch.tensor([-0.9674, 0.0000, -0.4307, 0.4307, 0.9674]),
        atol=1e-4,
    )


# def test_inverse_quantile_transform():
#     data_tensor = torch.arange(1, 1001, step=1).repeat(3, 1).T.float()
#     # Shuffle each column
#     for i in range(data_tensor.shape[1]):
#         data_tensor[:, i] = data_tensor[:, i][torch.randperm(data_tensor.shape[0])]
#     tensor_transformed = quantile_transform(data_tensor)
#     new_value_transformed = torch.tensor(
#         [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32
#     )

#     new_value = inverse_quantile_transform(
#         new_value_transformed, data_tensor, tensor_transformed
#     )
#     assert new_value.shape == new_value_transformed.shape
#     assert torch.allclose(
#         new_value,
#         torch.tensor(
#             [[500.5000, 500.5000, 500.5000], [842.1858, 842.1858, 842.1858]]
#         ),
#         atol=1e-4,
#     )
#     # Same new values should return original tensor
#     inverse_values = inverse_quantile_transform(
#         tensor_transformed, data_tensor, tensor_transformed
#     )
#     assert torch.allclose(
#         inverse_values,
#         data_tensor,
#         atol=1e-4,
#     )

# def test_rescale_latent_variable():
#     tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

#     processed, mean, std = rescale_latent_variable(tensor, "standardize")
#     assert torch.allclose(processed, (tensor - mean) / std)

#     processed, min_val, max_val = rescale_latent_variable(tensor, "normalize")
#     assert torch.allclose(processed, (tensor - min_val) / (max_val - min_val))

#     with pytest.raises(ValueError):
#         rescale_latent_variable(tensor, "invalid_method")


# def test_inverse_rescale_latent_variable():
#     tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

#     processed, mean, std = rescale_latent_variable(tensor, "standardize")
#     reconstructed = inverse_rescale_latent_variable(processed, "standardize", mean, std)
#     assert torch.allclose(tensor, reconstructed, atol=1e-7)

#     processed, min_val, max_val = rescale_latent_variable(tensor, "normalize")
#     reconstructed = inverse_rescale_latent_variable(
#         processed, "normalize", min_val, max_val
#     )
#     assert torch.allclose(tensor, reconstructed, atol=1e-7)

#     with pytest.raises(ValueError):
#         inverse_rescale_latent_variable(tensor, "invalid_method", 0, 1)


# def test_latent_score_grid(z_scores):
#     steps = 5
#     min_z = torch.full((1, z_scores.shape[1]), z_scores.min() - 1)

#     output = latent_score_grid(z_scores, steps, min_z)

#     # Check the shape of the output
#     assert output.shape == (steps * z_scores.shape[0], z_scores.shape[1])

#     # Check the values of the output
#     expected_output = (
#         torch.tensor(
#             [
#                 0.0000,
#                 0.2500,
#                 0.5000,
#                 0.7500,
#                 1.0000,
#                 0.0000,
#                 0.5000,
#                 1.0000,
#                 1.5000,
#                 2.0000,
#             ]
#         )
#         .repeat(z_scores.shape[1], 1)
#         .T
#     )

#     assert torch.allclose(output, expected_output, atol=1e-5)


# def test_tanh_latent_score_grid():
#     z = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
#     steps = 20
#     inverted_scale_multiplier = torch.tensor([1.0, -1.0])
#     z_means = torch.mean(z, dim=0)
#     z_stds = torch.std(z, dim=0)

#     # Run the function
#     result = tanh_latent_score_grid(
#         z, steps, inverted_scale_multiplier, z_means, z_stds
#     )

#     # Check the output shape
#     assert result.shape == (z.shape[0] * steps, z.shape[1]), "Output shape mismatch"

#     expected = torch.tensor(
#         [
#             [-5.0027, 4.0027],
#             [-1.2253, 0.2253],
#             [-0.7295, -0.2705],
#             [-0.4359, -0.5641],
#             [-0.2252, -0.7748],
#             [-0.0600, -0.9400],
#             [0.0765, -1.0765],
#             [0.1933, -1.1933],
#             [0.2955, -1.2955],
#             [0.3867, -1.3867],
#             [0.4692, -1.4692],
#             [0.5448, -1.5448],
#             [0.6145, -1.6145],
#             [0.6795, -1.6795],
#             [0.7403, -1.7403],
#             [0.7976, -1.7976],
#             [0.8519, -1.8519],
#             [0.9035, -1.9035],
#             [0.9528, -1.9528],
#             [1.0000, -2.0000],
#             [-5.0027, 4.0027],
#             [-0.2045, -0.7955],
#             [0.3172, -1.3172],
#             [0.6372, -1.6372],
#             [0.8757, -1.8757],
#             [1.0705, -2.0705],
#             [1.2384, -2.2384],
#             [1.3887, -2.3887],
#             [1.5270, -2.5270],
#             [1.6571, -2.6571],
#             [1.7816, -2.7816],
#             [1.9030, -2.9030],
#             [2.0229, -3.0229],
#             [2.1431, -3.1431],
#             [2.2654, -3.2654],
#             [2.3918, -3.3918],
#             [2.5246, -3.5246],
#             [2.6670, -3.6670],
#             [2.8232, -3.8232],
#             [3.0000, -4.0000],
#         ]
#     )
#     assert torch.allclose(result, expected, atol=1e-4)


# def test_entropy_distance():
#     entropies = torch.tensor(
#         [[[1.0, 2.0, 3.0], [4.0, 7.0, 11.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
#     )

#     output = entropy_distance(entropies)

#     # Check the values of the output
#     # The expected output is the sum of the absolute differences
#     # between consecutive elements along the third dimension (path),
#     # ignoring the first element in each path (set to 0).
#     # So for the first respondent:
#     # For item 1, the differences are [0, 2-1, 3-2] = [0, 1, 1]
#     # For item 2, the differences are [0, 7-4, 11-7] = [0, 3, 4]
#     # So the total for the first respondent is 9
#     # Similarly, for the second respondent, the total is 4
#     expected_output = torch.tensor([9.0, 4.0])

#     assert torch.allclose(output, expected_output, atol=1e-5)