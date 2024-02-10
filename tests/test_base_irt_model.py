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

    assert torch.allclose(expected_item_scores, torch.tensor([[0.123599261,  0.061870147], [-0.107541688, -0.045147929]]))

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

    # Check the shape of the output
    assert information_matrices.shape == (2, 2, 2, 2)

    # Check the values of the output
    expected_output = torch.tensor([
        [[[ 4184.934082,  4786.798340], [ 4786.798340,  5478.406738]], [[19931.039062, 21267.791016], [21267.791016, 22694.289062]]],
        [[[646821568, 739235008], [739235008, 844860736]], [[2972079104, 3170238464], [3170238464, 3381610496]]]
    ])
    assert torch.allclose(information_matrices, expected_output)

    # Test with item=False
    information_matrices = base_irt_model.information(input_z, item=False)

    # Check the shape of the output
    assert information_matrices.shape == (2, 2, 2)

    # Check the values of the output
    expected_output = torch.tensor([
        [[24115.972656, 26054.589844], [26054.589844, 28172.695312]],
        [[3618900480, 3909473536], [3909473536, 4226471168]]
    ])
    assert torch.allclose(information_matrices, expected_output)
