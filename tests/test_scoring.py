from unittest.mock import MagicMock, patch
import torch
from torch.nn.functional import softmax
import pytest
from irtorch.scoring import Scoring
from irtorch.estimation_algorithms import AE
from irtorch.models import BaseIRTModel

class ConcreteIRTModel(BaseIRTModel):
    """
    Concrete implementation of BaseIRTModel for testing purposes.
    """
    def forward(self, z):
        pass

    def probabilities_from_output(self, output: torch.Tensor):
        pass

@pytest.fixture
def base_irt_model(latent_variables):
    return ConcreteIRTModel(latent_variables, [2, 3])

@pytest.fixture
def irt_scorer(z_scores, latent_variables):
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
    mock_algorithm.z_scores = MagicMock(side_effect=z_scores_mock)

    return Scoring(mock_model, mock_algorithm)

from unittest.mock import patch
from irtorch.scoring import GaussianMixtureModel

@pytest.mark.parametrize("cv_n_components", [[1], [1, 2, 3]])
def test_cv_gaussian_mixture_model(irt_scorer: Scoring, cv_n_components):
    data = torch.randn(100, irt_scorer.model.latent_variables)

    with patch.object(GaussianMixtureModel, "fit") as mock_fit, patch.object(GaussianMixtureModel, "__call__") as mock_call:
        mock_fit.return_value = None
        mock_call.return_value = torch.tensor(1.0)

        gmm = irt_scorer._cv_gaussian_mixture_model(data, cv_n_components)

    assert isinstance(gmm, GaussianMixtureModel)
    assert gmm.n_components == cv_n_components[0]
    assert gmm.n_features == data.shape[1]
    mock_fit.assert_called()
    if len(cv_n_components) > 1:
        assert mock_call.call_count == len(cv_n_components) * 5  # 5-fold cross-validation
