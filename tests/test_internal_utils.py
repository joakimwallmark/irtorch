import pytest
import torch
import numpy as np
from irtorch._internal_utils import (
    linear_regression,
    is_jupyter,
    dynamic_print,
    conditional_score_distribution,
    sum_incorrect_probabilities,
    entropy,
    random_guessing_data,
    one_hot_encode_test_data,
    impute_missing_internal,
    correlation_matrix,
    joint_entropy_matrix,
)

@pytest.fixture(scope="module")
def logits(device):
    return torch.tensor(
        [
            [
                -0.2168,
                0.0781,
                -0.0588,
                0.1306,
                0.1659,
                0.1068,
                -0.3093,
                0.0875,
                -0.1423,
                -0.0832,
                0.3190,
                -0.0511,
                0.1717,
                -0.1309,
                0.3678,
                0.3022,
            ],
            [
                -0.2116,
                0.0789,
                -0.0464,
                0.1117,
                0.1522,
                0.1322,
                -0.3207,
                0.0925,
                -0.1325,
                -0.0844,
                0.3234,
                -0.0261,
                0.1774,
                -0.1098,
                0.3450,
                0.3082,
            ],
        ]
    ).to(device)


def test_is_jupyter():
    assert is_jupyter() is False

def test_dynamic_print(capsys):
    dynamic_print("arg1\narg2")

    # Capture the output again after calling dynamic_print
    captured_output = capsys.readouterr()
    assert captured_output.out == "\rarg1\narg2 "

    dynamic_print("Hello World!")
    captured_output = capsys.readouterr()

    assert captured_output.out == "\rHello World! "

def test_linear_regression():
    x = torch.tensor([[1., 2.], [2., 2.5], [5., 3.], [5., 3.5]], dtype=torch.float32)
    y = (2 + 2*x[:, 0] + 3*x[:, 1]).reshape(-1, 1)
    # y = 2 + 2*x

    coefficients = linear_regression(x, y)

    assert coefficients[0].item() == pytest.approx(2.0, abs=1e-3)  # Close to bias term 0
    assert coefficients[1].item() == pytest.approx(2.0, abs=1e-3)  # Close to weight 2 for the feature
    assert coefficients[2].item() == pytest.approx(3.0, abs=1e-3)  # Close to weight 2 for the feature

def test_conditional_scores_distribution():
    probabilities = torch.tensor([
        [[0.5, 0.5, 0.0, 0.0], [0.2, 0.3, 0.5, 0.0], [0.2, 0.2, 0.2, 0.4]],
        [[0.4, 0.6, 0.0, 0.0], [0.4, 0.2, 0.4, 0.0], [0.1, 0.3, 0.3, 0.3]]
    ])
    item_categories = [2, 3, 4]

    # Test without mc_correct
    result = conditional_score_distribution(probabilities, item_categories)

    # Define the expected output
    expected_output = torch.tensor(
        [
            [0.02, 0.07, 0.15, 0.22, 0.23, 0.21, 0.10],
            [0.016, 0.080, 0.172, 0.252, 0.252, 0.156, 0.072],
        ],
    )
    # Check if the output matches the expected output
    assert torch.allclose(result, expected_output, atol=1e-5)

    # Test with mc_correct
    mc_correct = [0, 1, 2]
    probabilities = sum_incorrect_probabilities(
        probabilities, item_categories, mc_correct
    )
    item_categories = [2] * len(item_categories)
    result = conditional_score_distribution(probabilities, item_categories)

    expected_output_mc_correct = torch.tensor(
        [[0.2800, 0.4700, 0.2200, 0.0300], [0.3360, 0.4520, 0.1880, 0.0240]],
    )

    assert torch.allclose(result, expected_output_mc_correct, atol=1e-5)


def test_sum_incorrect_probabilities():
    probabilities = torch.tensor([
        [[0.5, 0.5, 0.0, 0.0], [0.2, 0.3, 0.5, 0.0], [0.2, 0.2, 0.2, 0.4]],
        [[0.4, 0.6, 0.0, 0.0], [0.4, 0.2, 0.4, 0.0], [0.1, 0.3, 0.2, 0.4]]
    ])
    
    item_categories = [2, 3, 4]
    mc_correct = [0, 2, 1]

    result = sum_incorrect_probabilities(
        probabilities,
        item_categories,
        mc_correct
    )

    expected_output = torch.tensor([
        [[0.5, 0.5], [0.5, 0.5], [0.8, 0.2]],
        [[0.6, 0.4], [0.6, 0.4], [0.7, 0.3]]
    ])

    # Check if the output matches the expected output
    assert torch.allclose(result, expected_output, atol=1e-5)

def test_entropy():
    # Also test that if one probability is 0, we get correct response
    probabilities = torch.tensor([[0.5, 0.5, 0.0], [0.1, 0.8, 0.1]])

    bit_score_values = entropy(probabilities)
    expected_values = torch.tensor(
        [
            -0.5 * np.log2(0.5) - 0.5 * np.log2(0.5),
            -2 * 0.1 * np.log2(0.1) - 0.8 * np.log2(0.8),
        ],
        dtype=torch.float,
    )

    assert torch.allclose(bit_score_values, expected_values)

    with pytest.raises(RuntimeError):
        invalid_probabilities = torch.tensor(
            [[0.5, 0.6], [0.1, 0.9]]
        )  # probabilities do not sum to 1
        entropy(invalid_probabilities)

def test_random_guessing_data():
    # Test non multiple choice data
    item_categories = [2, 2, 2, 2, 4]
    guessing_probabilities = [0.6, 0.6, 0.6, 0.6, 0.5]
    data = random_guessing_data(item_categories, 100, guessing_probabilities)
    assert data.shape == (100, 5)
    assert data.min() == 0
    assert data.max() == 1
    assert data[:,4].max() == 0
    assert data.unique().tolist() == [0, 1]

    # Test multiple choice data
    item_categories = [3, 3, 3, 3, 3]
    guessing_probabilities = [0.5, 0.5, 0.5, 0.5, 0.5]
    mc_correct = [1, 0, 2, 0, 1]
    data = random_guessing_data(item_categories, 100, guessing_probabilities, mc_correct)
    assert data.shape == (100, 5)
    assert data.min() == 0
    assert data.max() == 2
    assert data.unique().tolist() == [0, 1, 2]

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
    one_hot_scores = one_hot_encode_test_data(scores, item_categories)

    expected = (
        torch.tensor(
            [
                [1, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0],
                [0, 1, 0,  0, 0, 1, 0,  0, 0, 0, 1],
                [0, 0, 1,  1, 0, 0, 0,  0, 0, 0, 0],
                [0, 0, 1,  1, 0, 0, 0,  0, 0, 0, 0],
            ]
        )
        .float()
        .to(device)
    )

    assert torch.all(one_hot_scores == expected)
    assert one_hot_scores.shape == (4, sum(item_categories))
    assert one_hot_scores.dtype == torch.float32
    with pytest.raises(ValueError):
        one_hot_encode_test_data(torch.tensor([0, 1, 2]), item_categories)
    with pytest.raises(ValueError):
        one_hot_encode_test_data(scores, [2, 2])

def test_impute_missing_internal():
    data = torch.tensor([[1, 2, 1, -1], [-1, float("nan"), 0, 2], [-1, float("nan"), 1, 2]])
    imputed_data = impute_missing_internal(data = data, method="zero")
    assert (imputed_data == torch.tensor([[1, 2, 1, 0], [0, 0, 0, 2], [0, 0, 1, 2]])).all()

    imputed_data = impute_missing_internal(data = data, method="mean")
    assert (imputed_data == torch.tensor([[1, 2, 1, 2], [1, 2, 0, 2], [1, 2, 1, 2]])).all()

    mc_correct = [1, 2, 1, 2]
    imputed_data = impute_missing_internal(data = data, method = "random incorrect", mc_correct=mc_correct, item_categories=[3, 3, 2, 3])
    missing_mask = torch.logical_or(data == -1, data.isnan())
    assert (imputed_data[missing_mask] > -1).all() # all missing are replaced
    assert (imputed_data[~missing_mask] == data[~missing_mask]).all() # non missing are still the same
    assert imputed_data[0, 3] != 2 # we did not replace with true values
    assert imputed_data[1, 0] != 1
    assert imputed_data[1, 1] != 2
    assert imputed_data[2, 0] != 1
    assert imputed_data[2, 1] != 2

def test_correlation_matrix():
    x = torch.tensor([[1.0, 2.0], [3.0, 3.0], [5.0, 6.0]])
    result = correlation_matrix(x)
    expected = torch.tensor([[1.0, 0.9607689], [0.9607689, 1.0]])
    assert torch.allclose(result, expected, atol=1e-5)

    # Test with a tensor containing NaNs
    x = torch.tensor([[1.0, 2.0], [3.0, float('nan')], [5.0, 6.0]])
    result = correlation_matrix(x)
    expected = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    assert torch.allclose(result, expected, atol=1e-5)

    # Test with invalid input
    x = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        correlation_matrix(x)

def test_joint_entropy_matrix():
    x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 3.0, 3.0], [1.0, 2.0, 1.0], [2.0, 3.0, 1.0]])
    result = joint_entropy_matrix(x)
    expected = torch.tensor([
        [1.5000, 1.5000, 2.0000],
        [1.5000, 1.0000, 2.0000],
        [2.0000, 2.0000, 1.0000]
    ])
    assert torch.allclose(result, expected, atol=1e-5)

    # Test with a tensor containing NaNs
    x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 3.0, 3.0], [1.0, 2.0, torch.nan], [2.0, 3.0, 1.0]])
    result = joint_entropy_matrix(x)
    expected = torch.tensor([
        [1.5000, 1.5000, 1.5850],
        [1.5000, 1.0000, 1.5850],
        [1.5850, 1.5850, 0.9183]]
    )
    assert torch.allclose(result, expected, atol=1e-4)

    # Test with invalid input
    x = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        joint_entropy_matrix(x)
