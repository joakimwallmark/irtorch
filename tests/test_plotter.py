import pytest
import torch
from irtorch.plotter import Plotter
from irtorch.models import BaseIRTModel
from irtorch.estimation_algorithms import MML
from unittest.mock import MagicMock

@pytest.fixture
def mock_model():
    item_categories = [2, 3, 4, 4, 2]
    def model_forward_mock(input_tensor: torch.Tensor):
        return torch.randn(input_tensor.shape[0], 5, 4)

    mock_model = MagicMock(spec = BaseIRTModel, side_effect=model_forward_mock)
    mock_model.latent_variables = 2
    mock_model.mc_correct = None
    mock_model.scale = []
    mock_model.items = 5
    mock_model.item_categories = item_categories
    mock_model.algorithm = MagicMock(spec = MML)
    mock_model.algorithm.training_history = {
        "train_loss": [0.9, 0.8, 0.7],
        "validation_loss": [0.85, 0.75, 0.65]
    }
    return mock_model

@pytest.fixture
def plotter_instance(mock_model):
    return Plotter(mock_model)

def test_plot_training_history_success(plotter_instance: Plotter):
    fig = plotter_instance.plot_training_history()
    assert fig is not None
    assert fig.data is not None
    assert fig.layout.title.text == "Training History"

def test_plot_training_history_not_trained(plotter_instance: Plotter):
    plotter_instance.model.algorithm.training_history = {"train_loss": [], "validation_loss": []}
    with pytest.raises(AttributeError, match="Model has not been trained yet"):
        plotter_instance.plot_training_history()

def test_plot_latent_score_distribution_success(plotter_instance: Plotter):
    scores_to_plot = torch.rand((100, 2))
    fig = plotter_instance.plot_latent_score_distribution(scores_to_plot=scores_to_plot)
    assert fig is not None
    assert fig.data is not None
    assert fig.layout.title.text == None

def test_plot_latent_score_distribution_invalid_dim(plotter_instance: Plotter):
    with pytest.raises(ValueError, match="Can only plot 1 or 2 latent variables."):
        plotter_instance.plot_latent_score_distribution(latent_variables_to_plot=(1, 2, 3))

def test_plot_item_entropy_success(plotter_instance: Plotter):
    plotter_instance.model.latent_variables = 1
    def probabilities_from_output(output: torch.Tensor) -> torch.Tensor:
        reshaped_output = output.reshape(-1, 4)
        return torch.nn.functional.softmax(reshaped_output, dim=1).reshape_as(output)
    
    plotter_instance.model.probabilities_from_output = MagicMock(side_effect=probabilities_from_output)
    fig = plotter_instance.plot_item_entropy(item=1)
    assert fig is not None
    assert fig.data is not None

def test_plot_item_entropy_invalid_latent_vars(plotter_instance: Plotter):
    with pytest.raises(TypeError, match="Cannot plot more than two latent variables in one plot."):
        plotter_instance.plot_item_entropy(item=1, latent_variables=(1, 2, 3))

def test_plot_item_latent_variable_relationships_success(plotter_instance: Plotter):
    relationships = torch.rand((5, 2))
    fig = plotter_instance.plot_item_latent_variable_relationships(relationships=relationships)
    assert fig is not None
    assert fig.data is not None
    assert fig.layout.title.text == "Relationships: Items vs. latent variables"

def test_plot_item_probabilities_success(plotter_instance: Plotter):
    plotter_instance.model.latent_variables = 1
    plotter_instance.model.item_probabilities = MagicMock(return_value=torch.rand((100, 5, 4)))
    plotter_instance.model.evaluate._min_max_theta_for_integration = MagicMock(return_value=(torch.tensor([-3]), torch.tensor([3])))
    fig = plotter_instance.plot_item_probabilities(item=1)
    assert fig is not None
    assert fig.data is not None
    assert "IRF - Item 1" in fig.layout.title.text

def test_plot_information_success(plotter_instance: Plotter):
    plotter_instance.model.latent_variables = 1
    plotter_instance.model.information = MagicMock(return_value=torch.rand((100,)))
    fig = plotter_instance.plot_information()
    assert fig is not None
    assert fig.data is not None

def test_plot_expected_sum_score_success(plotter_instance: Plotter):
    plotter_instance.model.latent_variables = 1
    plotter_instance.model.expected_scores = MagicMock(return_value=torch.rand((100,)))
    fig = plotter_instance.plot_expected_sum_score()
    assert fig is not None
    assert fig.data is not None
    assert "Expected sum score" in fig.layout.title.text
