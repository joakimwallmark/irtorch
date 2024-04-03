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
    output_to_item_entropy,
    impute_missing,
    random_guessing_data
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

def testdynamic_print(capsys):
    dynamic_print("arg1\narg2")

    # Capture the output again after calling dynamic_print
    captured_output = capsys.readouterr()
    assert captured_output.out == '\rarg1\narg2 '

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
    mc_correct = [1, 2, 3]
    probabilities = sum_incorrect_probabilities(
        probabilities, item_categories, mc_correct, False
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
    mc_correct = [1, 3, 2]

    result = sum_incorrect_probabilities(
        probabilities,
        item_categories,
        mc_correct,
        missing_modeled=False
    )

    expected_output = torch.tensor([
        [[0.5, 0.5], [0.5, 0.5], [0.8, 0.2]],
        [[0.6, 0.4], [0.6, 0.4], [0.7, 0.3]]
    ])

    # Check if the output matches the expected output
    assert torch.allclose(result, expected_output, atol=1e-5)

    # Test with missing_modeled=True
    result = sum_incorrect_probabilities(
        probabilities,
        item_categories,
        [x - 1 for x in mc_correct],
        missing_modeled=True
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


def test_output_to_item_entropy():
    # Test with valid input
    output = torch.randn(100, 10)
    item_categories = [2, 2, 2, 2, 2]
    entropies = output_to_item_entropy(output, item_categories)
    assert entropies.shape == (100, 5)

    # Test with invalid input (length of item_categories is not equal to the second dimension of output divided by the sum of item_categories)
    with pytest.raises(ValueError):
        output_to_item_entropy(output, [2, 2, 3, 2, 2])

def test_impute_missing():
    data = torch.tensor([[1, 2, 1, -1], [-1, float('nan'), 0, 2], [-1, float('nan'), 1, 2]])
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
    mc_correct = [2, 1, 3, 1, 2]
    data = random_guessing_data(item_categories, 100, guessing_probabilities, mc_correct)
    assert data.shape == (100, 5)
    assert data.min() == 0
    assert data.max() == 2
    assert data.unique().tolist() == [0, 1, 2]