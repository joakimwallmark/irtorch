import pytest
import torch
from torch.nn.functional import softmax
from irtorch.models import BaseIRTModel
from irtorch.estimation_algorithms import BaseIRTAlgorithm
from irtorch.estimation_algorithms.encoders import StandardEncoder
from unittest.mock import MagicMock, patch

class ConcreteIRTModel(BaseIRTModel):
    """
    Concrete implementation of BaseIRTModel for testing purposes.
    """
    def forward(self, z):
        # Find the unique rows in the input_tensor and their inverse indices
        unique_rows, inverse_indices = torch.unique(z, dim=0, return_inverse=True)
        logits_unique = torch.randn(
            unique_rows.shape[0],
            len(self.item_categories),
            max(self.item_categories)
        )
        for item, category in enumerate(self.item_categories):
            logits_unique[:, item, category:] = -torch.inf
        # Reconstruct the logits for all rows using the inverse indices
        logits_unique = logits_unique.reshape(-1, len(self.item_categories) * max(self.item_categories))
        logits_all_rows = logits_unique[inverse_indices]
        
        return logits_all_rows

    def probabilities_from_output(self, output: torch.Tensor):
        pass

@pytest.fixture
def base_irt_model(latent_variables):
    return ConcreteIRTModel(latent_variables, [3, 4])


def test_expected_scores(base_irt_model: BaseIRTModel):
    # Create a mock for item_probabilities() method
    def item_probabilities_mock(*args, **kwargs):
        return torch.tensor([
            [[0.1, 0.1, 0.8, 0.0], [0.1, 0.1, 0.1, 0.7]],
            [[0.3, 0.3, 0.4, 0.0], [0.3, 0.5, 0.1, 0.1]]
        ])
    base_irt_model.item_probabilities = MagicMock(
        side_effect=item_probabilities_mock
    )

    expected_item_scores = base_irt_model.expected_scores(torch.randn((2, 2)), return_item_scores=True)
    assert torch.allclose(expected_item_scores, torch.tensor([[1.7, 2.4], [1.1, 1.0]]))
    assert torch.allclose(expected_item_scores.sum(dim=1), base_irt_model.expected_scores(torch.randn((2, 2)), return_item_scores=False))

    base_irt_model.model_missing = True
    expected_item_scores = base_irt_model.expected_scores(torch.randn((2, 2)), return_item_scores=True)
    assert torch.allclose(expected_item_scores, torch.tensor([[0.8, 1.5], [0.4, 0.3]]))
    assert torch.allclose(expected_item_scores.sum(dim=1), base_irt_model.expected_scores(torch.randn((2, 2)), return_item_scores=False))

    base_irt_model.model_missing = False
    base_irt_model.mc_correct = [1, 2]
    expected_item_scores = base_irt_model.expected_scores(torch.randn((2, 2)), return_item_scores=True)
    assert torch.allclose(expected_item_scores, torch.tensor([[0.1, 0.1], [0.3, 0.5]]))
    assert torch.allclose(expected_item_scores.sum(dim=1), base_irt_model.expected_scores(torch.randn((2, 2)), return_item_scores=False))
    
    base_irt_model.model_missing = True
    base_irt_model.mc_correct = [0, 1] # should give the same as above even if in practice invalid mc_correct
    expected_item_scores = base_irt_model.expected_scores(torch.randn((2, 2)), return_item_scores=True)
    assert torch.allclose(expected_item_scores, torch.tensor([[0.1, 0.1], [0.3, 0.5]]))
    assert torch.allclose(expected_item_scores.sum(dim=1), base_irt_model.expected_scores(torch.randn((2, 2)), return_item_scores=False))

    # what if we just have 1 respondent?
    def item_probabilities_mock2(*args, **kwargs):
        return torch.tensor([[[0.1, 0.1, 0.8, 0.0], [0.1, 0.1, 0.2, 0.7]]])
    
    base_irt_model.item_probabilities = MagicMock(
        side_effect=item_probabilities_mock2
    )
    base_irt_model.mc_correct = None
    base_irt_model.model_missing = False
    expected_item_scores = base_irt_model.expected_scores(torch.randn((1, 2)), return_item_scores=True)
    assert torch.allclose(expected_item_scores, torch.tensor([[1.7, 2.6]]))
    base_irt_model.mc_correct = [3, 1]
    base_irt_model.model_missing = False
    expected_item_scores = base_irt_model.expected_scores(torch.randn((1, 2)), return_item_scores=True)
    assert torch.allclose(expected_item_scores, torch.tensor([[0.8, 0.1]]))
    base_irt_model.mc_correct = None
    base_irt_model.model_missing = True
    expected_item_scores = base_irt_model.expected_scores(torch.randn((1, 2)), return_item_scores=True)
    assert torch.allclose(expected_item_scores, torch.tensor([[0.8, 1.6]]))

def test_expected_item_score_slopes(base_irt_model: BaseIRTModel):
    # Create a mock for item_probabilities() method
    def item_probabilities_mock(z):
        logits = torch.tensor([[[1, 2, 3, 0], [4, 3, 2, 1]]]).expand(3, 2, 4) * z.sum(dim=1).reshape(-1, 1, 1)
        logits[:, 0, 3] = -torch.inf
        return softmax(logits, dim=2)
    
    base_irt_model.item_probabilities = MagicMock(
        side_effect=item_probabilities_mock
    )

    input_z = torch.tensor([[-2.0, -3.0], [1.0, 2.0], [1.0, 1.0]])
    expected_item_scores = base_irt_model.expected_item_score_slopes(input_z)

    assert torch.allclose(expected_item_scores, torch.tensor([[ 0.1236,  0.0619], [-0.1075, -0.0451]]), atol=1e-3)

def test_information(base_irt_model: BaseIRTModel):
    def item_probabilities_mock(z):
        logits = torch.tensor([[[1, 2, 3, 0], [4, 3, 2, 1]]]).expand(2, 2, 4) * z.sum(dim=1).reshape(-1, 1, 1)
        return softmax(logits, dim=2)
    
    base_irt_model.item_probabilities = MagicMock(
        side_effect=item_probabilities_mock
    )

    def probability_gradients_mock(z):
        return torch.tensor([
            [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]]],
            [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]]]
        ])
    
    base_irt_model.probability_gradients = MagicMock(
        side_effect=probability_gradients_mock
    )

    input_z = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    information_matrices = base_irt_model.information(input_z)
    assert information_matrices.shape == (2, 2, 2, 2)
    expected_output = torch.tensor([
        [[[ 4184.934082,  4786.798340], [ 4786.798340,  5478.406738]], [[19931.039062, 21267.791016], [21267.791016, 22694.289062]]],
        [[[646821568, 739235008], [739235008, 844860736]], [[2972079104, 3170238464], [3170238464, 3381610496]]]
    ])
    assert torch.allclose(information_matrices, expected_output)

    # Test with degrees
    if base_irt_model.latent_variables == 2:
        information_matrices = base_irt_model.information(input_z, degrees=[45, 45])
        assert information_matrices.shape == (2, 2)
        expected_output = torch.tensor([
            [ 9618.4677734, 42580.4531250],
            [1485075968.0000000, 6347082752.0000000]]
        )
        assert torch.allclose(information_matrices, expected_output)

    # Test with item=False
    information_matrices = base_irt_model.information(input_z, item=False)
    assert information_matrices.shape == (2, 2, 2)
    expected_output = torch.tensor([
        [[24115.972656, 26054.589844], [26054.589844, 28172.695312]],
        [[3618900480, 3909473536], [3909473536, 4226471168]]
    ])
    assert torch.allclose(information_matrices, expected_output)

    # Test with item=False and degrees
    if base_irt_model.latent_variables == 2:
        information_matrices = base_irt_model.information(input_z, item=False, degrees=[0, 90])
        assert information_matrices.shape == (2, )
        expected_output = torch.tensor([24115.972656, 3618899968.000000])
        assert torch.allclose(information_matrices, expected_output)

@pytest.mark.parametrize("z_estimation_method", ["NN", "EAP"])
def test_latent_scores(base_irt_model: BaseIRTModel, z_estimation_method):
    base_irt_model.algorithm = MagicMock(spec=BaseIRTAlgorithm)
    base_irt_model.algorithm.one_hot_encoded = False
    base_irt_model.algorithm.encoder = MagicMock(spec=StandardEncoder)

    def z_scores_mock(input_tensor):
        if torch.allclose(input_tensor, torch.tensor([[0.0, 0.0]])):
            # return min z scores (all 0 response)
            # a 2D tensor with one row where the values alternate between +3 and -3
            values = [3 if i % 2 == 0 else -3 for i in range(base_irt_model.latent_variables)]
            return torch.tensor(values).unsqueeze(0).float()
        elif torch.allclose(input_tensor, torch.tensor([[1.0, 2.0]])):
            # return max z scores (all max response)
            # a 2D tensor with one row where the values alternate between +3 and -3
            values = [-3 if i % 2 == 0 else 3 for i in range(base_irt_model.latent_variables)]
            return torch.tensor(values).unsqueeze(0).float()
        else:
            return torch.randn(input_tensor.shape[0], base_irt_model.latent_variables)

    base_irt_model.algorithm.z_scores = MagicMock(side_effect=z_scores_mock)
    
    data = torch.tensor([[1.0, 0.0], [1.0, 2.0], [1.0, 2.0]])
    expected_output = torch.randn(data.shape[0], base_irt_model.latent_variables)
    scores = base_irt_model.latent_scores(data, z_estimation_method=z_estimation_method)

    if z_estimation_method == "NN":
        call_args = base_irt_model.algorithm.z_scores.call_args
        assert call_args is not None, "z_scores method was not called"
        assert torch.allclose(
            call_args[0][0], data.contiguous()
        ), "z_scores method was called with incorrect arguments"

    assert scores.shape == expected_output.shape
    assert scores.dtype == torch.float32

def test_probability_gradients(base_irt_model: BaseIRTModel):
    # Create a mock for item_probabilities() method
    def item_probabilities_mock(z):
        logits = torch.tensor([[[1, 2, 3, 0], [4, 3, 2, 1]]]).expand(1, 2, 4) * z.sum(dim=1).reshape(-1, 1, 1)
        logits[:, 0, 3] = -torch.inf
        return softmax(logits, dim=2)
    
    base_irt_model.item_probabilities = MagicMock(
        side_effect=item_probabilities_mock
    )

    input_z = torch.tensor([[-2.0, -3.0], [1.0, 2.0], [1.0, 1.0]])
    prob_gradients = base_irt_model.probability_gradients(input_z)

    assert prob_gradients.shape == (3, 2, 4, 2)

def test_sample_test_data(base_irt_model: BaseIRTModel):
    # Create a mock for item_probabilities() method
    def item_probabilities_mock(*args, **kwargs):
        return torch.tensor([
            [[0.1, 0.1, 0.8, 0.0], [0.1, 0.1, 0.1, 0.7]],
            [[0.3, 0.3, 0.4, 0.0], [0.3, 0.5, 0.1, 0.1]]
        ])
    base_irt_model.item_probabilities = MagicMock(side_effect=item_probabilities_mock)

    # Call sample_test_data
    z = torch.randn((2, 2))
    sampled_data = base_irt_model.sample_test_data(z)

    # Check the output shape and type
    assert sampled_data.shape == z.shape
    assert sampled_data.dtype == torch.float32

    # Check if the values are in the correct range (0 to number of categories - 1)
    assert torch.all((sampled_data >= 0) & (sampled_data < 4))

@pytest.mark.parametrize("grid_size", [3, 4, 5])  # Test with multiple grid sizes
def test__z_grid(base_irt_model: BaseIRTModel, grid_size):
    test_tensor = torch.tensor([[1, -4, -1], [2, 5, -3], [3, 6, -2], [4, 6, -5]])
    
    grid_combinations = base_irt_model._z_grid(test_tensor, grid_size=grid_size)
    
    # Check that the number of rows in the grid_combinations matches expected value
    expected_rows = grid_size ** test_tensor.shape[1]
    assert grid_combinations.shape[0] == expected_rows, f"Expected {expected_rows} rows but got {grid_combinations.shape[0]}"

    # Check the number of columns remains unchanged
    assert grid_combinations.shape[1] == test_tensor.shape[1], f"Expected {test_tensor.shape[1]} columns but got {grid_combinations.shape[1]}"

    # Verify the range of values in grid_combinations
    for col in range(test_tensor.shape[1]):
        min_val = test_tensor[:, col].min().item()
        max_val = test_tensor[:, col].max().item()
        
        # Check the minimum and maximum values considering the 0.25 scaling factor
        adjusted_min = min_val - 0.25 * (max_val - min_val)
        adjusted_max = max_val + 0.25 * (max_val - min_val)
        
        assert grid_combinations[:, col].min().item() >= adjusted_min, f"Column {col} min value is less than expected"
        assert grid_combinations[:, col].max().item() <= adjusted_max, f"Column {col} max value is more than expected"

    # Ensure uniqueness of each row (i.e., no repeated combinations)
    unique_rows = torch.unique(grid_combinations, dim=0)
    assert unique_rows.shape[0] == grid_combinations.shape[0], "Detected repeated combinations in the grid"