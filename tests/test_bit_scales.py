from unittest.mock import MagicMock, patch
import torch
import pytest
from unittest.mock import patch
from irtorch.bit_scales import BitScales
from irtorch.estimation_algorithms import AE
from irtorch.models import BaseIRTModel

@pytest.fixture
def bit_scales(z_scores, latent_variables):
    # Create a mock instance of AEIRTNeuralNet
    item_categories = [2, 3]
    mock_algorithm = MagicMock(spec=AE)
    mock_algorithm.one_hot_encoded = False
    mock_algorithm.training_z_scores = z_scores.clone().detach()
    mock_algorithm.train_data = torch.tensor([[1, 2], [0, 0], [1, 2], [1, 1]]).float()

    # Mock z_scores method based on input
    def z_scores_mock(input_tensor):
        if torch.allclose(input_tensor, torch.tensor([[0.0, 0.0]])):
            # return min z scores (all 0 response)
            # a 2D tensor with one row where the values alternate between +3 and -3
            values = [3 if i % 2 == 0 else -3 for i in range(latent_variables)]
            return torch.tensor(values).unsqueeze(0).float()
        elif torch.allclose(input_tensor, torch.tensor([[1.0, 2.0]])):
            # return max z scores (all max response)
            # a 2D tensor with one row where the values alternate between +3 and -3
            values = [-3 if i % 2 == 0 else 3 for i in range(latent_variables)]
            return torch.tensor(values).unsqueeze(0).float()
        else:
            return torch.randn(input_tensor.shape[0], latent_variables)

    # Mock decoder forward method
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

    # Mock decoder log_likelihood method
    def log_likelihood_mock(replicated_data, replicated_logits, loss_reduction = "none"):    
        return torch.randn(replicated_data.shape[0], len(item_categories))

    mock_model = MagicMock(spec=BaseIRTModel)
    mock_model.mc_correct = None
    mock_model.latent_variables = latent_variables
    mock_model.item_categories = item_categories
    mock_model.modeled_item_responses = item_categories
    mock_model.side_effect = forward_mock
    mock_model.model_missing = False
    mock_model.log_likelihood = MagicMock(side_effect=log_likelihood_mock)
    mock_model.algorithm = MagicMock(spec=AE)
    mock_model.algorithm.z_scores = MagicMock(side_effect=z_scores_mock)
    mock_model.algorithm.training_z_scores = z_scores.clone().detach()
    mock_model.algorithm.train_data = torch.tensor([[1, 2], [0, 0], [1, 2], [1, 1]]).float()

    return BitScales(mock_model)

@pytest.mark.parametrize("one_dimensional", [True, False])
@pytest.mark.parametrize("bit_score_z_grid_method", ["NN", "ML"])
def test_bit_scores(bit_scales: BitScales, one_dimensional, latent_variables, bit_score_z_grid_method):
    z = torch.tensor([[0.8], [2.1], [-0.7], [-0.5]]).repeat(1, latent_variables)
    z = z[1::2] * -1 # invert every other scale
    start_z = torch.full((1, latent_variables), -0.2)

    def latent_scores_mock(*args, **kwargs):
        return torch.tensor([[0.8], [-0.7], [0.8]]).repeat(1, latent_variables)

    bit_scales.model.latent_scores = MagicMock(side_effect=latent_scores_mock)

    # with patch.object(BitScales, "latent_scores", return_value=torch.tensor([[0.8], [-0.7], [0.8]]).repeat(1, latent_variables)) as mock_latent_scores:
    inverted_scales = torch.ones((1, latent_variables))
    inverted_scales[0, 1::2] = -1 # invert every other scale
    with patch.object(BitScales, "_inverted_scales", return_value=inverted_scales):
        # with patch.object(IRTScorer, "_anti_invert_and_adjust_z_scores", return_value=(z, z, start_z)):
        grid_start = torch.full((latent_variables,), -0.6)
        grid_end = torch.full((latent_variables,), 2.0)
        with patch.object(BitScales, "_get_grid_boundaries", return_value=(grid_start, grid_end, torch.zeros((1, latent_variables), dtype=torch.bool))):
            if one_dimensional:
                with patch.object(BitScales, "_compute_1d_bit_scores", return_value=torch.tensor([[0.5], [0.5], [0.5], [0.5]])):
                    bit_scores, _ = bit_scales.bit_scores_from_z(z, start_z, one_dimensional=one_dimensional, z_estimation_method=bit_score_z_grid_method)
            else:
                with patch.object(BitScales, "_compute_multi_dimensional_bit_scores", return_value=torch.tensor([[0.5], [0.5], [0.5], [0.5]]).repeat(1, latent_variables)):
                    bit_scores, _ = bit_scales.bit_scores_from_z(z, start_z, one_dimensional=one_dimensional, z_estimation_method=bit_score_z_grid_method)

    # Assert the result
    if one_dimensional:
        assert bit_scores.shape == (4, 1)
    else:
        assert bit_scores.shape == (4, latent_variables)

    if bit_score_z_grid_method == "ML":
        bit_scales.model.latent_scores.assert_called_once_with(bit_scales.model.algorithm.train_data, z_estimation_method="ML", ml_map_device="cuda" if torch.cuda.is_available() else "cpu", lbfgs_learning_rate=0.3)

def test_compute_1d_bit_scores(bit_scales: BitScales, latent_variables):
    torch.manual_seed(4)
    start_z_adjusted = torch.full((1, latent_variables), -1.5)
    grid_start = torch.tensor([[-1.2] * latent_variables])
    grid_end = torch.tensor([[1.0] * latent_variables])
    z_adjusted = torch.randn(5, latent_variables)
    inverted_scale = torch.ones((1, latent_variables))
    inverted_scale[0, 1::2] = -1 # invert every other latent scale
    grid_points = 10

    bit_scores = bit_scales._compute_1d_bit_scores(
        z_adjusted,
        start_z_adjusted,
        grid_start,
        grid_end,
        inverted_scale,
        grid_points
    )

    assert bit_scores.ndim == 2, "bit scores should have 2 dimensions"
    assert bit_scores.size(1) == 1, "bit scores should have size 1 in the second dimension"
    assert bit_scores.size(0) == 5, "bit scores should have size 1 in the second dimension"
    assert torch.all(bit_scores[(z_adjusted < start_z_adjusted).all(dim=1)] == 0), "Smaller than start should be set to start"

def test_compute_multi_dimensional_bit_scores(bit_scales: BitScales, latent_variables):
    torch.manual_seed(51)
    start_z_adjusted = torch.full((1, latent_variables), -1.2)
    train_z_adjusted = torch.randn(5, latent_variables)
    z_adjusted = torch.randn(5, latent_variables)
    grid_start = torch.tensor([[-1] * latent_variables])
    grid_end = torch.tensor([[1.0] * latent_variables])
    inverted_scale = torch.ones((1, latent_variables))
    inverted_scale[0, 1::2] = -1 # invert every other latent scale
    grid_points = 10
    
    bit_scores = bit_scales._compute_multi_dimensional_bit_scores(
        z_adjusted,
        start_z_adjusted,
        train_z_adjusted,
        grid_start,
        grid_end,
        inverted_scale,
        grid_points
    )

    assert bit_scores.ndim == 2, "bit scores should have 2 dimensions"
    assert bit_scores.size(1) == latent_variables, "bit scores should have size 1 in the second dimension"
    assert bit_scores.size(0) == 5, "bit scores should have size 1 in the second dimension"
    assert torch.all(bit_scores[z_adjusted < start_z_adjusted] == 0), "Smaller than start should be set to start"