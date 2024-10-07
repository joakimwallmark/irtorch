import pytest
import torch
from irtorch.rescale import Bit
from irtorch.models import BaseIRTModel, TwoParameterLogistic

class MockModel(BaseIRTModel):
    def __init__(self, latent_variables=1, item_categories=[2, 2, 2], mc_correct=None):
        super().__init__(latent_variables=latent_variables, item_categories=item_categories)
        self.mc_correct = mc_correct
        self.algorithm = None
        self.training_theta_scores = None

    def item_probabilities(self, theta_scores):
        # Simulate probabilities for binary items
        return torch.softmax(torch.randn(theta_scores.shape[0], self.items, self.max_item_responses))

    def probabilities_from_output(self, output):
        return output

    def expected_scores(self, thetas):
        return self.item_probabilities(thetas)

    def forward(self, thetas):
        return self.item_probabilities(thetas)

    def latent_scores(self, data, theta_estimation="ML", ml_map_device=None, lbfgs_learning_rate=None, rescale=True):
        return torch.randn(data.shape[0], self.latent_variables)


def test_bit_transform(latent_variables):
    model = TwoParameterLogistic(latent_variables=latent_variables, items=3)
    bit_scale = Bit(model)
    theta = torch.tensor([[0.0], [1.0], [-1.0]])
    bit_scores = bit_scale.transform(theta)
    assert bit_scores.shape == (3, latent_variables)
    assert torch.all(bit_scores >= 0)

    # Test with invalid grid_points
    with pytest.raises(ValueError):
        bit_scale.transform(theta, grid_points=0)

    # Test with items parameter
    bit_scores = bit_scale.transform(theta, items=[0])
    assert bit_scores.shape == (3, latent_variables)

    # Test with invalid items parameter
    with pytest.raises(ValueError):
        bit_scale.transform(theta, items='invalid')


def test_bit_transform_to_1D(latent_variables):
    model = TwoParameterLogistic(latent_variables=latent_variables, items=3)
    bit_scale = Bit(model)
    theta = torch.tensor([[0.0], [1.0], [-1.0]]).repeat(1, latent_variables)
    bit_scores = bit_scale.transform_to_1D(theta)
    assert bit_scores.shape == (3, 1)

    # Test with invalid grid_points
    with pytest.raises(ValueError):
        bit_scale.transform_to_1D(theta, grid_points=0)


def test_bit_gradients():
    model = TwoParameterLogistic(latent_variables=1, items=3)
    bit_scale = Bit(model)
    theta = torch.tensor([[0.0], [1.0], [-1.0]]).repeat(1, 1)
    gradients = bit_scale.gradients(theta)
    assert gradients.shape == (3, 1, 1)
    assert torch.all(gradients >= 0)


def test_bit__compute_inverted_scale_multiplier(latent_variables):
    model = TwoParameterLogistic(latent_variables=latent_variables, items=3)
    bit_scale = Bit(model)
    multiplier = bit_scale._get_inverted_scale_multiplier()
    assert multiplier.shape == (1, latent_variables)
    assert torch.all(multiplier == 1)


def test_bit__get_grid_boundaries(latent_variables):
    model = TwoParameterLogistic(latent_variables=latent_variables, items=3)
    bit_scale = Bit(model)
    train_theta_adjusted = torch.tensor([[-1.0], [0.0], [1.0]]).repeat(1, latent_variables)
    start_theta_adjusted = torch.full((1, latent_variables), -2.0)
    grid_start, grid_end = bit_scale._get_grid_boundaries(train_theta_adjusted, start_theta_adjusted)
    assert grid_start.shape == (1, latent_variables)
    assert grid_end.shape == (1, latent_variables)


def test_bit__compute_1d_bit_scores(latent_variables):
    model = TwoParameterLogistic(latent_variables=latent_variables, items=3)
    bit_scale = Bit(model)
    theta_adjusted = torch.tensor([[0.0], [1.0], [-1.0]]).repeat(1, latent_variables)
    start_theta_adjusted = torch.full((1, latent_variables), -3.0)
    grid_start = torch.full((1, latent_variables), -2.0)
    grid_end = torch.full((1, latent_variables), 2.0)
    invert_scale_multiplier = torch.ones((1, latent_variables))
    bit_scores = bit_scale._compute_1d_bit_scores(theta_adjusted, start_theta_adjusted, grid_start, grid_end, invert_scale_multiplier, 100)
    assert bit_scores.shape == (3, 1)


def test_bit__compute_multi_dimensional_bit_scores(latent_variables):
    model = TwoParameterLogistic(latent_variables=latent_variables, items=3)
    bit_scale = Bit(model)
    theta_adjusted = torch.tensor([[0.0], [1.0], [-1.0]]).repeat(1, latent_variables)
    start_theta_adjusted = torch.full((1, latent_variables), -3.0)
    grid_start = torch.full((1, latent_variables), -2.0)
    grid_end = torch.full((1, latent_variables), 2.0)
    median_thetas = torch.zeros((1, latent_variables))
    invert_scale_multiplier = torch.ones((1, latent_variables))
    bit_scores = bit_scale._compute_multi_dimensional_bit_scores(
        theta_adjusted,
        start_theta_adjusted,
        median_thetas,
        grid_start,
        grid_end,
        invert_scale_multiplier,
        100
    )
    assert bit_scores.shape == theta_adjusted.shape
