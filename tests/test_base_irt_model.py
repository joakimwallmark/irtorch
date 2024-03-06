import pytest
import torch
from torch.nn.functional import softmax
from irtorch.models import BaseIRTModel
from unittest.mock import MagicMock, patch

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
    return ConcreteIRTModel(latent_variables, [3, 4])


def test_expected_item_sum_score(base_irt_model: BaseIRTModel):
    # Create a mock for item_probabilities() method
    def item_probabilities_mock(*args, **kwargs):
        return torch.tensor([
            [[0.1, 0.1, 0.8, 0.0], [0.1, 0.1, 0.1, 0.7]],
            [[0.3, 0.3, 0.4, 0.0], [0.3, 0.5, 0.1, 0.1]]
        ])
    base_irt_model.item_probabilities = MagicMock(
        side_effect=item_probabilities_mock
    )

    expected_item_scores = base_irt_model.expected_item_sum_score(torch.randn((2, 2)), return_item_scores=True)
    assert torch.allclose(expected_item_scores, torch.tensor([[1.7, 2.4], [1.1, 1.0]]))
    assert torch.allclose(expected_item_scores.sum(dim=1), base_irt_model.expected_item_sum_score(torch.randn((2, 2)), return_item_scores=False))

    base_irt_model.model_missing = True
    expected_item_scores = base_irt_model.expected_item_sum_score(torch.randn((2, 2)), return_item_scores=True)
    assert torch.allclose(expected_item_scores, torch.tensor([[0.8, 1.5], [0.4, 0.3]]))
    assert torch.allclose(expected_item_scores.sum(dim=1), base_irt_model.expected_item_sum_score(torch.randn((2, 2)), return_item_scores=False))

    base_irt_model.model_missing = False
    base_irt_model.mc_correct = [1, 2]
    expected_item_scores = base_irt_model.expected_item_sum_score(torch.randn((2, 2)), return_item_scores=True)
    assert torch.allclose(expected_item_scores, torch.tensor([[0.1, 0.1], [0.3, 0.5]]))
    assert torch.allclose(expected_item_scores.sum(dim=1), base_irt_model.expected_item_sum_score(torch.randn((2, 2)), return_item_scores=False))
    
    base_irt_model.model_missing = True
    base_irt_model.mc_correct = [0, 1] # should give the same as above even if in practice invalid mc_correct
    expected_item_scores = base_irt_model.expected_item_sum_score(torch.randn((2, 2)), return_item_scores=True)
    assert torch.allclose(expected_item_scores, torch.tensor([[0.1, 0.1], [0.3, 0.5]]))
    assert torch.allclose(expected_item_scores.sum(dim=1), base_irt_model.expected_item_sum_score(torch.randn((2, 2)), return_item_scores=False))

    # what if we just have 1 respondent?
    def item_probabilities_mock2(*args, **kwargs):
        return torch.tensor([[[0.1, 0.1, 0.8, 0.0], [0.1, 0.1, 0.2, 0.7]]])
    
    base_irt_model.item_probabilities = MagicMock(
        side_effect=item_probabilities_mock2
    )
    base_irt_model.mc_correct = None
    base_irt_model.model_missing = False
    expected_item_scores = base_irt_model.expected_item_sum_score(torch.randn((1, 2)), return_item_scores=True)
    assert torch.allclose(expected_item_scores, torch.tensor([[1.7, 2.6]]))
    base_irt_model.mc_correct = [3, 1]
    base_irt_model.model_missing = False
    expected_item_scores = base_irt_model.expected_item_sum_score(torch.randn((1, 2)), return_item_scores=True)
    assert torch.allclose(expected_item_scores, torch.tensor([[0.8, 0.1]]))
    base_irt_model.mc_correct = None
    base_irt_model.model_missing = True
    expected_item_scores = base_irt_model.expected_item_sum_score(torch.randn((1, 2)), return_item_scores=True)
    assert torch.allclose(expected_item_scores, torch.tensor([[0.8, 1.6]]))

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
