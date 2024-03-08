import pytest
import torch
import scipy.stats as stats
from irtorch.latent_variable_functions import interp, quantile_transform

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
