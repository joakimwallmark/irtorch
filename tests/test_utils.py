import pytest
from unittest.mock import MagicMock
import torch
from irtorch.utils import gauss_hermite, get_item_categories, split_data, impute_missing, _imw_newton_raphson, imw

def test_gauss_hermite():
    n_points = 4
    mu = torch.tensor([1, 0])
    Sigma = torch.tensor([[1.3, -0.213], [-0.213, 1.2]])
    points, weights = gauss_hermite(n_points, mu, Sigma)

    # Calculations
    mean = torch.sum((weights)[:, None] * points, dim=0)
    variance = torch.sum((weights)[:, None] * (points - mu)**2, dim=0)
    # Covariance calculation using outer product
    covfunc = lambda x: torch.ger(x - mu, x - mu)
    covariance = torch.sum((weights)[:, None, None] * torch.stack(list(map(covfunc, points))), dim=0)
    # Polynomial calculation
    polynomial = torch.sum((weights)[:, None] * (4 + 2 * points - 0.5 * points**2), dim=0)
    # Logistic function calculation
    logistic = torch.sum((weights)[:, None] * (1 / (1 + torch.exp(-points))), dim=0)
    # Assertions
    assert torch.allclose(mean, torch.tensor([1.0000, 0.0000], dtype=torch.float64), atol=1e-4)
    assert torch.allclose(variance, torch.tensor([1.3000, 1.2000], dtype=torch.float64), atol=1e-4)
    assert torch.allclose(covariance, torch.tensor([[1.3000, -0.2130], [-0.2130, 1.2000]], dtype=torch.float64), atol=1e-4)
    assert torch.allclose(polynomial, torch.tensor([4.8500, 3.4000], dtype=torch.float64), atol=1e-4)
    assert torch.allclose(logistic, torch.tensor([0.6888, 0.5000], dtype=torch.float64), atol=1e-4)


def test_get_item_categories(test_data, item_categories):
    result = get_item_categories(test_data)
    assert result == item_categories

def test_impute_missing():
    data = torch.tensor([[1, 2, 1, -1], [-1, float("nan"), 0, 2], [-1, float("nan"), 1, 2]])
    imputed_data = impute_missing(data = data, method="zero")
    assert (imputed_data == torch.tensor([[1, 2, 1, 0], [0, 0, 0, 2], [0, 0, 1, 2]])).all()

    imputed_data = impute_missing(data = data, method="mean")
    assert (imputed_data == torch.tensor([[1, 2, 1, 2], [1, 2, 0, 2], [1, 2, 1, 2]])).all()

    mc_correct = [1, 2, 1, 2]
    imputed_data = impute_missing(data = data, method = "random incorrect", mc_correct=mc_correct, item_categories=[3, 3, 2, 3])
    missing_mask = torch.logical_or(data == -1, data.isnan())
    assert (imputed_data[missing_mask] > -1).all() # all missing are replaced
    assert (imputed_data[~missing_mask] == data[~missing_mask]).all() # non missing are still the same
    assert imputed_data[0, 3] != 2 # we did not replace with true values
    assert imputed_data[1, 0] != 1
    assert imputed_data[1, 1] != 2
    assert imputed_data[2, 0] != 1
    assert imputed_data[2, 1] != 2


def test__imw_newton_raphson():
    # Negative entropy values
    negative_entropy = torch.tensor([-0.4, -0.6], requires_grad=False)
    w = _imw_newton_raphson(negative_entropy)
    negative_entropy_result = w * torch.log(w) + (1 - w) * torch.log(1 - w)
    assert torch.allclose(negative_entropy_result, negative_entropy, atol=1e-6), "The computed w does not satisfy the equation."

def test_imw():
    # Mock models
    model1 = MagicMock()
    model2 = MagicMock()
    model1.evaluate.log_likelihood = MagicMock(return_value=torch.tensor(-0.5))
    model2.evaluate.log_likelihood = MagicMock(return_value=torch.tensor(-0.2))
    # Data (can be empty since we are mocking)
    data = torch.tensor([])
    # Call imw function
    imw_value = imw(model1, model2, data)
    # Compute expected imw
    log_geometric_means = torch.tensor([-0.5, -0.2])
    w = _imw_newton_raphson(log_geometric_means)
    expected_imw = (w[1] - w[0]) / w[0]
    # Check that imw_value matches expected_imw
    assert torch.allclose(imw_value, expected_imw, atol=1e-6), "The computed IMW does not match the expected value."

def test_split_data():
    torch.manual_seed(42)
    data = torch.rand(100, 10)

    # Test with shuffle=True
    train_data, test_data = split_data(data, train_ratio=0.8, shuffle=True)
    assert train_data.shape[0] == 80  # 80% of 100 samples
    assert test_data.shape[0] == 20  # 20% of 100 samples
    assert not torch.equal(data[:80], train_data)  # Data should be shuffled

    # Test with shuffle=False
    train_data, test_data = split_data(data, train_ratio=0.8, shuffle=False)
    assert train_data.shape[0] == 80
    assert test_data.shape[0] == 20
    assert torch.equal(data[:80], train_data)  # Data should not be shuffled

    # Test with different train_ratio
    train_data, test_data = split_data(data, train_ratio=0.6, shuffle=False)
    assert train_data.shape[0] == 60
    assert test_data.shape[0] == 40

    # Test with train_ratio=1.0
    train_data, test_data = split_data(data, train_ratio=1.0, shuffle=False)
    assert train_data.shape[0] == 100
    assert test_data.shape[0] == 0

    # Test with train_ratio=0.0
    train_data, test_data = split_data(data, train_ratio=0.0, shuffle=False)
    assert train_data.shape[0] == 0
    assert test_data.shape[0] == 100
