import pytest
import torch
from irtorch.utils import gauss_hermite, get_item_categories, one_hot_encode_test_data, decode_one_hot_test_data, split_data, impute_missing

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
    # mc_correct is corresponds to correct item category and not correct score (2 means score 1 is correct)
    mc_correct = [2, 3, 2, 3]
    imputed_data = impute_missing(data = data.clone())
    assert (imputed_data == torch.tensor([[1, 2, 1, 0], [0, 0, 0, 2], [0, 0, 1, 2]])).all()

    imputed_data = impute_missing(data = data, mc_correct=mc_correct, item_categories=[3, 3, 2, 3])
    missing_mask = torch.logical_or(data == -1, data.isnan())
    assert (imputed_data[missing_mask] > -1).all() # all missing are replaced
    assert (imputed_data[~missing_mask] == data[~missing_mask]).all() # non missing are still the same
    assert imputed_data[0, 3] != 2 # we did not replace with true values
    assert imputed_data[1, 0] != 1
    assert imputed_data[1, 1] != 2
    assert imputed_data[2, 0] != 1
    assert imputed_data[2, 1] != 2

def test_one_hot_encode_test_data(device):
    # Define a small tensor of test scores and a list of maximum scores
    #
    scores = (
        torch.tensor([[0, 1, 2], [1, 2, 3], [2, 0, float("nan")], [2, 0, -1]])
        .float()
        .to(device)
    )
    item_categories = [3, 4, 4]

    # Call the function to get the one-hot encoded tensor, with encode_missing set to True
    one_hot_scores = one_hot_encode_test_data(
        scores, item_categories, encode_missing=True
    )

    expected = (
        torch.tensor(
            [
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )
        .float()
        .to(device)
    )

    assert torch.all(one_hot_scores == expected)

    # Call the function to get the one-hot encoded tensor
    one_hot_scores = one_hot_encode_test_data(
        scores, item_categories, encode_missing=False
    )

    expected = torch.tensor(
        [
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
        ]
    ).to(device)

    assert torch.all(one_hot_scores == expected)
    assert one_hot_scores.shape == (4, sum(item_categories))
    assert one_hot_scores.dtype == torch.float32
    with pytest.raises(ValueError):
        one_hot_encode_test_data(
            torch.tensor([0, 1, 2]), item_categories, encode_missing=False
        )
    with pytest.raises(ValueError):
        one_hot_encode_test_data(scores, [2, 2], encode_missing=False)

def test_decode_one_hot_test_data(device):
    scores = (
        torch.tensor([[0, 1, 2], [1, 2, 3], [2, 0, 1], [2, 0, 1]]).float().to(device)
    ).float()
    item_categories = [3, 4, 4]

    one_hot_scores = one_hot_encode_test_data(
        scores, item_categories, encode_missing=False
    )

    decoded_scores = decode_one_hot_test_data(one_hot_scores, item_categories)

    # Check that the decoded scores match the original scores
    assert torch.all(decoded_scores == scores)

    with pytest.raises(ValueError):
        decode_one_hot_test_data(torch.tensor([0, 1, 2]), item_categories)
    with pytest.raises(ValueError):
        decode_one_hot_test_data(one_hot_scores, [2, 2])

    # test with
    scores = (
        torch.tensor([[0, -1, 2], [1, 2, 3], [2, 0, 1], [2, -1, 1]]).float().to(device)
    ).float()
    item_categories = [3, 4, 4]

    # Call the function to get the one-hot encoded tensor
    one_hot_scores = one_hot_encode_test_data(
        scores, item_categories, encode_missing=True
    )

    # Call the function to decode the one-hot encoded tensor
    decoded_scores = decode_one_hot_test_data(
        one_hot_scores, [item_cat + 1 for item_cat in item_categories]
    )

    # Check that the decoded scores match the original scores
    assert torch.all(decoded_scores == scores + 1)

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
