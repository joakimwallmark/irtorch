from unittest.mock import MagicMock, patch
import torch
import pytest
from irtorch.rescale import Bit
from irtorch.estimation_algorithms import AE
from irtorch.models import BaseIRTModel

@pytest.fixture
def bit(theta_scores, latent_variables):
    # Create a mock instance of AEIRTNeuralNet
    item_categories = [2, 3]
    mock_algorithm = MagicMock(spec=AE)
    mock_algorithm.training_theta_scores = theta_scores.clone().detach()
    mock_algorithm.train_data = torch.tensor([[1, 2], [0, 0], [1, 2], [1, 1]]).float()

    # Mock theta_scores method based on input
    def theta_scores_mock(input_tensor):
        if torch.allclose(input_tensor, torch.tensor([[0.0, 0.0]])):
            # return min theta scores (all 0 response)
            # a 2D tensor with one row where the values alternate between +3 and -3
            values = [3 if i % 2 == 0 else -3 for i in range(latent_variables)]
            return torch.tensor(values).unsqueeze(0).float()
        elif torch.allclose(input_tensor, torch.tensor([[1.0, 2.0]])):
            # return max theta scores (all max response)
            # a 2D tensor with one row where the values alternate between +3 and -3
            values = [-3 if i % 2 == 0 else 3 for i in range(latent_variables)]
            return torch.tensor(values).unsqueeze(0).float()
        else:
            return torch.randn(input_tensor.shape[0], latent_variables)

    def forward_mock(input_tensor):
        # Find the unique rows in the input_tensor and their inverse indices
        unique_rows, inverse_indices = torch.unique(input_tensor, dim=0, return_inverse=True)
        logits_unique = torch.randn(
            unique_rows.shape[0],
            len(item_categories),
            max(item_categories)
        )
        for item, category in enumerate(item_categories):
            logits_unique[:, item, category:] = -torch.inf
        # Reconstruct the logits for all rows using the inverse indices
        logits_unique = logits_unique.reshape(-1, len(item_categories) * max(item_categories))
        logits_all_rows = logits_unique[inverse_indices]
        
        return logits_all_rows

    def log_likelihood_mock(replicated_data, replicated_logits, loss_reduction = "none"):
        return torch.randn(replicated_data.shape[0], len(item_categories))

    def probabilities_from_output_mock(output: torch.Tensor) -> torch.Tensor:
        reshaped_output = output.reshape(-1, 3)
        return torch.functional.F.softmax(reshaped_output, dim=1).reshape(output.shape[0], len(item_categories), 3)
    def expected_scores_mock(theta: torch.Tensor, return_item_scores = False) -> torch.Tensor:
        return torch.randn(theta.size(0)).abs().round().unsqueeze(1)

    mock_model = MagicMock(spec=BaseIRTModel)
    mock_model.mc_correct = None
    mock_model.latent_variables = latent_variables
    mock_model.item_categories = item_categories
    mock_model.item_categories = item_categories
    mock_model.side_effect = forward_mock
    mock_model.log_likelihood = MagicMock(side_effect=log_likelihood_mock)
    mock_model.probabilities_from_output = MagicMock(side_effect=probabilities_from_output_mock)
    mock_model.algorithm = MagicMock(spec=AE)
    mock_model.algorithm.theta_scores = MagicMock(side_effect=theta_scores_mock)
    mock_model.algorithm.training_theta_scores = theta_scores.clone().detach()
    mock_model.algorithm.train_data = torch.tensor([[1, 2], [0, 0], [1, 2], [1, 1]]).float()
    mock_model.expected_scores = MagicMock(side_effect=expected_scores_mock)

    return Bit(mock_model)

def test_transform(bit: Bit):
    theta = torch.tensor([[0.8], [2.1], [-0.7], [-0.5]])
    theta = theta[1::2] * -1 # invert every other scale
    start_theta = torch.full((1, 1), -0.2)

    def latent_scores_mock(*args, **kwargs):
        return torch.tensor([[0.8], [-0.7], [0.8]])

    bit.model.latent_scores = MagicMock(side_effect=latent_scores_mock)

    grid_start = torch.full((1,), -0.6)
    grid_end = torch.full((1,), 2.0)
    with patch.object(Bit, "_get_grid_boundaries", return_value=(grid_start, grid_end, torch.zeros((1, 1), dtype=torch.bool))):
        with patch.object(Bit, "_compute_1d_bit_scores", return_value=torch.tensor([[0.5], [0.5], [0.5], [0.5]])):
            bit_scores = bit.transform(theta, start_theta)

    assert bit_scores.shape == (4, 1)

def test_compute_1d_bit_scores(bit: Bit, latent_variables):
    torch.manual_seed(4)
    start_theta_adjusted = torch.full((1, latent_variables), -1.5)
    grid_start = torch.tensor([[-1.2] * latent_variables])
    grid_end = torch.tensor([[1.0] * latent_variables])
    theta_adjusted = torch.randn(5, latent_variables)
    inverted_scale = torch.ones((1, latent_variables))
    inverted_scale[0, 1::2] = -1 # invert every other latent scale
    grid_points = 10

    bit_scores = bit._compute_1d_bit_scores(
        theta_adjusted,
        start_theta_adjusted,
        grid_start,
        grid_end,
        inverted_scale,
        grid_points
    )

    assert bit_scores.ndim == 2, "bit scores should have 2 dimensions"
    assert bit_scores.size(1) == 1, "bit scores should have size 1 in the second dimension"
    assert bit_scores.size(0) == 5, "bit scores should have size 1 in the second dimension"
    assert torch.all(bit_scores[(theta_adjusted < start_theta_adjusted).all(dim=1)] == 0), "Smaller than start should be set to start"

def test_bit_score_starting_theta(ae_1d_mmc_swesat_model, ae_1d_mmc_swesat_thetas):
    bit_transform = Bit(
        ae_1d_mmc_swesat_model,
        population_theta=ae_1d_mmc_swesat_thetas,
        mc_start_theta_approx=True,
        theta_estimation="NN"
    )

    assert bit_transform._start_theta.shape == (1, ), "Starting theta should have shape (1, )"

# def test_compute_multi_dimensional_bit_scores(bit: Bit, latent_variables):
#     torch.manual_seed(51)
#     start_theta_adjusted = torch.full((1, latent_variables), -1.2)
#     train_theta_adjusted = torch.randn(5, latent_variables)
#     theta_adjusted = torch.randn(5, latent_variables)
#     grid_start = torch.tensor([[-1] * latent_variables])
#     grid_end = torch.tensor([[1.0] * latent_variables])
#     inverted_scale = torch.ones((1, latent_variables))
#     inverted_scale[0, 1::2] = -1 # invert every other latent scale
#     grid_points = 10
    
#     bit_scores = bit._compute_multi_dimensional_bit_scores(
#         theta_adjusted,
#         start_theta_adjusted,
#         train_theta_adjusted,
#         grid_start,
#         grid_end,
#         inverted_scale,
#         grid_points
#     )

#     assert bit_scores.ndim == 2, "bit scores should have 2 dimensions"
#     assert bit_scores.size(1) == latent_variables, "bit scores should have size 1 in the second dimension"
#     assert bit_scores.size(0) == 5, "bit scores should have size 1 in the second dimension"
#     assert torch.all(bit_scores[theta_adjusted < start_theta_adjusted] == 0), "Smaller than start should be set to start"

# @pytest.mark.parametrize("guessing_probabilities", [None, [0.25, 0.5]])
# def test_bit_score_starting_theta(bit: Bit, guessing_probabilities, latent_variables):
#     items = [0, 1]  
#     train_theta = torch.randn(3, latent_variables)

#     mock_latent_scores = MagicMock(return_value=torch.randn(1, latent_variables))
#     with patch.object(bit.model, 'latent_scores', mock_latent_scores):
#         with patch.object(bit.model, 'item_theta_relationship_directions', return_value=torch.ones(len(items), latent_variables)):
#             starting_theta = bit.bit_score_starting_theta(
#                 theta_estimation="ML",
#                 start_all_incorrect=False,
#                 guessing_probabilities=guessing_probabilities,
#                 items=items,
#                 train_theta=train_theta
#             )
            
#             assert starting_theta.shape == (1, latent_variables), "Starting theta should have shape (1, latent_variables)"
#             if guessing_probabilities:
#                 assert len(guessing_probabilities) == len(items), "Guessing probabilities should match the number of items"
