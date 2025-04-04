"""Tests for the Plotter class in irtorch.plotter module."""

import os
import sys

import pytest
import torch
from unittest.mock import MagicMock

from irtorch.plotter import Plotter
from irtorch.rescale.scale import Scale
from irtorch.models import BaseIRTModel
from irtorch.estimation_algorithms import MML

# Add the current directory to the path so we can import the patch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class DummyScale(Scale):
    """A simple scale transformation for testing purposes.

    This scale adds 1 to the input tensor and subtracts 1 from the output tensor.
    """
    def transform(self, theta: torch.Tensor) -> torch.Tensor:
        """Transform the input tensor by adding 1."""
        return theta + 1

    def inverse(self, transformed_theta: torch.Tensor) -> torch.Tensor:
        """Inverse transform by subtracting 1."""
        return transformed_theta - 1

    def jacobian(self, theta: torch.Tensor) -> torch.Tensor:
        """Return the Jacobian matrix for the transformation.

        Returns an identity matrix for each row since the transformation is a simple shift.
        """
        n, dims = theta.shape
        return torch.eye(dims).unsqueeze(0).repeat(n, 1, 1)

@pytest.fixture
def mock_model():
    """Create a mock IRT model for testing.

    Returns:
        MagicMock: A mock IRT model with necessary attributes and methods for testing.
    """
    item_categories = [2, 3, 4, 4, 2]
    def model_forward_mock(input_tensor: torch.Tensor):
        return torch.randn(input_tensor.shape[0], 5, 4)

    model = MagicMock(spec=BaseIRTModel, side_effect=model_forward_mock)
    model.latent_variables = 2
    model.mc_correct = None
    model.scale = []
    model.items = 5
    model.item_categories = item_categories
    model.algorithm = MagicMock(spec=MML)
    model.algorithm.training_history = {
        "train_loss": [0.9, 0.8, 0.7],
        "validation_loss": [0.85, 0.75, 0.65]
    }
    model.scale = [DummyScale(invertible=True)]
    model.transform_theta = MagicMock(side_effect=model.scale[0].transform)
    model.evaluate = MagicMock()
    model.evaluate._min_max_theta_for_integration = MagicMock(return_value=(torch.tensor([-3]), torch.tensor([3])))
    return model

@pytest.fixture
def plotter(mock_model):
    """Create a Plotter instance for testing.

    Args:
        mock_model: A mock IRT model fixture.

    Returns:
        Plotter: A Plotter instance initialized with the mock model.
    """
    return Plotter(mock_model)

class TestPlotter:
    """Test suite for the Plotter class."""

    def test_training_history(self, plotter):
        """Test that training history plot is created successfully."""
        fig = plotter.training_history()
        assert fig is not None
        assert fig.data is not None
        assert fig.layout.title.text == "Training History"

    def test_training_history_not_trained(self, plotter):
        """Test that an error is raised when plotting training history for an untrained model."""
        plotter.model.algorithm.training_history = {"train_loss": [], "validation_loss": []}
        with pytest.raises(AttributeError, match="Model has not been trained yet"):
            plotter.training_history()

    def test_latent_score_distribution(self, plotter):
        """Test that latent score distribution plot is created successfully."""
        scores_to_plot = torch.rand((100, 2))
        fig = plotter.latent_score_distribution(scores_to_plot=scores_to_plot)
        assert fig is not None
        assert fig.data is not None

    def test_latent_score_distribution_with_population_data(self, plotter):
        """Test latent_score_distribution with population_data parameter."""
        plotter.model.latent_scores = MagicMock(return_value=torch.rand(100, 2))
        plotter.model.algorithm.train_data = torch.rand(100, 5)
        population_data = torch.rand(50, 5)
        fig = plotter.latent_score_distribution(population_data=population_data)
        assert fig is not None
        assert fig.data is not None
        # Verify that latent_scores was called with the population_data
        plotter.model.latent_scores.assert_called_once()
        args, kwargs = plotter.model.latent_scores.call_args
        assert 'data' in kwargs
        assert torch.equal(kwargs['data'], population_data)

    def test_latent_score_distribution_with_custom_labels(self, plotter):
        """Test latent_score_distribution with custom labels."""
        scores_to_plot = torch.rand((100, 2))
        custom_title = "Custom Distribution Title"
        custom_x_label = "Custom X Label"
        custom_y_label = "Custom Y Label"
        fig = plotter.latent_score_distribution(
            scores_to_plot=scores_to_plot,
            title=custom_title,
            x_label=custom_x_label,
            y_label=custom_y_label
        )
        assert fig is not None
        assert fig.data is not None
        assert fig.layout.title.text == custom_title
        assert fig.layout.xaxis.title.text == custom_x_label
        # For 2D distributions, the y-axis label might be different than what we provided
        # because it's a histogram with a count on the y-axis

    def test_latent_score_distribution_1d(self, plotter):
        """Test latent_score_distribution with 1D latent variables."""
        scores_to_plot = torch.rand((100, 1))
        fig = plotter.latent_score_distribution(
            scores_to_plot=scores_to_plot,
            latent_variables=(1,)
        )
        assert fig is not None
        assert fig.data is not None
        # 1D distribution should have a histogram
        assert any(trace.type == 'histogram' for trace in fig.data)

    def test_latent_score_distribution_invalid(self, plotter):
        """Test that an error is raised when plotting latent score distribution with invalid dimensions."""
        with pytest.raises(ValueError, match="Can only plot 1 or 2 latent variables."):
            plotter.latent_score_distribution(latent_variables=(1, 2, 3))

    def test_latent_score_distribution_too_many_dimensions(self, plotter):
        """Test that an error is raised when plotting latent score distribution with too many dimensions."""
        plotter.model.latent_variables = 1
        with pytest.raises(ValueError, match="Cannot plot 2 dimensions"):
            plotter.latent_score_distribution(latent_variables=(1, 2))

    def test_item_entropy(self, plotter):
        """Test that item entropy plot is created successfully."""
        plotter.model.latent_variables = 1
        def probabilities_from_output(output: torch.Tensor) -> torch.Tensor:
            reshaped_output = output.reshape(-1, 4)
            return torch.nn.functional.softmax(reshaped_output, dim=1).reshape_as(output)

        plotter.model.probabilities_from_output = MagicMock(side_effect=probabilities_from_output)
        fig = plotter.item_entropy(item=1)
        assert fig is not None
        assert fig.data is not None

    def test_item_entropy_2d(self, plotter):
        """Test item entropy with 2D latent variables."""
        plotter.model.latent_variables = 2
        def probabilities_from_output(output: torch.Tensor) -> torch.Tensor:
            reshaped_output = output.reshape(-1, 4)
            return torch.nn.functional.softmax(reshaped_output, dim=1).reshape_as(output)

        plotter.model.probabilities_from_output = MagicMock(side_effect=probabilities_from_output)
        fig = plotter.item_entropy(item=1, latent_variables=(1, 2))
        assert fig is not None
        assert fig.data is not None
        # 2D entropy should be a surface plot
        assert any(trace.type == 'surface' for trace in fig.data)

    def test_item_entropy_with_custom_params(self, plotter):
        """Test item entropy with custom parameters."""
        plotter.model.latent_variables = 1
        def probabilities_from_output(output: torch.Tensor) -> torch.Tensor:
            reshaped_output = output.reshape(-1, 4)
            return torch.nn.functional.softmax(reshaped_output, dim=1).reshape_as(output)

        plotter.model.probabilities_from_output = MagicMock(side_effect=probabilities_from_output)
        custom_title = "Custom Entropy Title"
        custom_x_label = "Custom X Label"
        custom_y_label = "Custom Y Label"
        custom_color = "red"
        fig = plotter.item_entropy(
            item=1,
            title=custom_title,
            x_label=custom_x_label,
            y_label=custom_y_label,
            color=custom_color,
            theta_range=(-2, 2),
            steps=50
        )
        assert fig is not None
        assert fig.data is not None
        assert fig.layout.title.text == custom_title
        assert fig.layout.xaxis.title.text == custom_x_label
        assert fig.layout.yaxis.title.text == custom_y_label

    def test_item_entropy_invalid_latent_vars(self, plotter):
        """Test that an error is raised when plotting item entropy with invalid latent variables."""
        with pytest.raises(TypeError, match="Cannot plot more than two latent variables in one plot."):
            plotter.item_entropy(item=1, latent_variables=(1, 2, 3))

    def test_item_entropy_invalid_theta_range(self, plotter):
        """Test that an error is raised when plotting item entropy with invalid theta range."""
        with pytest.raises(TypeError, match="theta_range needs to have a length of 2."):
            plotter.item_entropy(item=1, theta_range=(-2, 0, 2))

    def test_item_latent_variable_relationships(self, plotter):
        """Test that item latent variable relationships plot is created successfully."""
        relationships = torch.rand((5, 2))
        fig = plotter.item_latent_variable_relationships(relationships=relationships)
        assert fig is not None
        assert fig.data is not None
        assert fig.layout.title.text == "Relationships: Items vs. latent variables"

    def test_item_probabilities(self, plotter):
        """Test that item probabilities plot is created successfully."""
        plotter.model.latent_variables = 1
        plotter.model.item_probabilities = MagicMock(return_value=torch.rand((100, 5, 4)))
        fig = plotter.item_probabilities(item=1)
        assert fig is not None
        assert fig.data is not None
        assert "IRF - Item 1" in fig.layout.title.text

    def test_item_probabilities_2d(self, plotter):
        """Test item probabilities with 2D latent variables."""
        plotter.model.latent_variables = 2
        plotter.model.item_probabilities = MagicMock(return_value=torch.rand((625, 5, 4)))
        fig = plotter.item_probabilities(item=1, latent_variables=(1, 2))
        assert fig is not None
        assert fig.data is not None
        # 2D probabilities should have surface plots
        assert any(trace.type == 'surface' for trace in fig.data)

    def test_item_probabilities_grayscale(self, plotter):
        """Test item probabilities with grayscale."""
        plotter.model.latent_variables = 1
        plotter.model.item_probabilities = MagicMock(return_value=torch.rand((100, 5, 4)))
        fig = plotter.item_probabilities(item=1, grayscale=True)
        assert fig is not None
        assert fig.data is not None

    def test_item_probabilities_with_custom_params(self, plotter):
        """Test item probabilities with custom parameters."""
        plotter.model.latent_variables = 1
        plotter.model.item_probabilities = MagicMock(return_value=torch.rand((100, 5, 4)))
        custom_title = "Custom Probabilities Title"
        custom_x_label = "Custom X Label"
        custom_y_label = "Custom Y Label"
        fig = plotter.item_probabilities(
            item=1,
            title=custom_title,
            x_label=custom_x_label,
            y_label=custom_y_label,
            theta_range=(-2, 2),
            steps=50
        )
        assert fig is not None
        assert fig.data is not None
        assert fig.layout.title.text == custom_title
        assert fig.layout.xaxis.title.text == custom_x_label
        assert fig.layout.yaxis.title.text == custom_y_label

    def test_item_probabilities_invalid_latent_vars(self, plotter):
        """Test that an error is raised when plotting item probabilities with invalid latent variables."""
        with pytest.raises(TypeError, match="Cannot plot more than two latent variables in one plot."):
            plotter.item_probabilities(item=1, latent_variables=(1, 2, 3))

    def test_information(self, plotter):
        """Test that information plot is created successfully."""
        plotter.model.latent_variables = 1
        plotter.model.information = MagicMock(return_value=torch.rand((100,)))
        fig = plotter.information()
        assert fig is not None
        assert fig.data is not None

    def test_information_with_items(self, plotter):
        """Test information plot with specific items."""
        plotter.model.latent_variables = 1
        plotter.model.information = MagicMock(return_value=torch.rand((100, 5)))
        fig = plotter.information(items=[1, 3])
        assert fig is not None
        assert fig.data is not None
        # Verify that information was called with item=True
        plotter.model.information.assert_called_once()
        _, kwargs = plotter.model.information.call_args
        assert kwargs.get('item') is True

    def test_information_2d(self, plotter):
        """Test information plot with 2D latent variables."""
        plotter.model.latent_variables = 2
        plotter.model.information = MagicMock(return_value=torch.rand((324,)))
        fig = plotter.information(latent_variables=(1, 2), degrees=[45, 45])
        assert fig is not None
        assert fig.data is not None
        # 2D information should be a surface plot
        assert any(trace.type == 'surface' for trace in fig.data)

    def test_information_with_custom_params(self, plotter):
        """Test information plot with custom parameters."""
        plotter.model.latent_variables = 1
        # Make sure the information method returns a tensor of the right shape
        # The shape should match the number of steps (50)
        plotter.model.information = MagicMock(return_value=torch.rand(50))
        custom_title = "Custom Information Title"
        custom_x_label = "Custom X Label"
        custom_y_label = "Custom Y Label"
        custom_color = "blue"
        fig = plotter.information(
            title=custom_title,
            x_label=custom_x_label,
            y_label=custom_y_label,
            color=custom_color,
            theta_range=(-2, 2),
            steps=50
        )
        assert fig is not None
        assert fig.data is not None
        assert fig.layout.title.text == custom_title
        assert fig.layout.xaxis.title.text == custom_x_label
        assert fig.layout.yaxis.title.text == custom_y_label

    def test_information_invalid_latent_vars(self, plotter):
        """Test that an error is raised when plotting information with invalid latent variables."""
        with pytest.raises(TypeError, match="Cannot plot more than two latent variables in one plot."):
            plotter.information(latent_variables=(1, 2, 3))

    def test_expected_sum_score(self, plotter):
        """Test that expected sum score plot is created successfully."""
        plotter.model.latent_variables = 1
        plotter.model.expected_scores = MagicMock(return_value=torch.rand((100,)))
        fig = plotter.expected_sum_score()
        assert fig is not None
        assert fig.data is not None
        assert "Expected sum score" in fig.layout.title.text

    def test_expected_sum_score_with_items(self, plotter):
        """Test expected sum score plot with specific items."""
        plotter.model.latent_variables = 1
        plotter.model.expected_scores = MagicMock(return_value=torch.rand((100, 5)))
        fig = plotter.expected_sum_score(items=[1, 3])
        assert fig is not None
        assert fig.data is not None
        # Verify that expected_scores was called with return_item_scores=True
        plotter.model.expected_scores.assert_called_once()
        _, kwargs = plotter.model.expected_scores.call_args
        assert kwargs.get('return_item_scores') is True

    def test_expected_sum_score_2d(self, plotter):
        """Test expected sum score plot with 2D latent variables."""
        plotter.model.latent_variables = 2
        plotter.model.expected_scores = MagicMock(return_value=torch.rand((324,)))
        fig = plotter.expected_sum_score(latent_variables=(1, 2))
        assert fig is not None
        assert fig.data is not None
        # 2D expected sum score should be a surface plot
        assert any(trace.type == 'surface' for trace in fig.data)

    def test_expected_sum_score_with_custom_params(self, plotter):
        """Test expected sum score plot with custom parameters."""
        plotter.model.latent_variables = 1
        # Make sure the expected_scores method returns a tensor of the right shape
        # The shape should match the number of steps (50)
        plotter.model.expected_scores = MagicMock(return_value=torch.rand(50))
        custom_title = "Custom Sum Score Title"
        custom_x_label = "Custom X Label"
        custom_y_label = "Custom Y Label"
        custom_color = "green"
        fig = plotter.expected_sum_score(
            title=custom_title,
            x_label=custom_x_label,
            y_label=custom_y_label,
            color=custom_color,
            theta_range=(-2, 2),
            steps=50
        )
        assert fig is not None
        assert fig.data is not None
        assert fig.layout.title.text == custom_title
        assert fig.layout.xaxis.title.text == custom_x_label
        assert fig.layout.yaxis.title.text == custom_y_label

    def test_expected_sum_score_invalid_latent_vars(self, plotter):
        """Test that an error is raised when plotting expected sum score with invalid latent variables."""
        with pytest.raises(TypeError, match="Cannot plot more than two latent variables in one plot."):
            plotter.expected_sum_score(latent_variables=(1, 2, 3))

    def test_scale_transformations(self, plotter):
        """Test that scale transformations plot is created successfully."""
        plotter.model.transform_theta = MagicMock(return_value=torch.rand(100, 2))
        fig = plotter.scale_transformations(input_latent_variable=1, steps=10)
        assert fig is not None
        assert hasattr(fig, "data")

    def test_scale_transformations_with_custom_params(self, plotter):
        """Test scale transformations with custom parameters."""
        plotter.model.transform_theta = MagicMock(return_value=torch.rand(100, 2))
        custom_title = "Custom Scale Title"
        custom_x_label = "Custom X Label"
        custom_y_label = "Custom Y Label"
        custom_color = "purple"
        fig = plotter.scale_transformations(
            input_latent_variable=1,
            title=custom_title,
            x_label=custom_x_label,
            y_label=custom_y_label,
            color=custom_color,
            input_theta_range=(-2, 2),
            steps=50
        )
        assert fig is not None
        assert hasattr(fig, "data")
        assert fig.layout.title.text == custom_title
        assert fig.layout.xaxis.title.text == custom_x_label
        assert fig.layout.yaxis.title.text == custom_y_label

    def test_scale_transformations_no_scale(self, plotter):
        """Test that an error is raised when plotting scale transformations with no scale."""
        plotter.model.scale = []
        with pytest.raises(ValueError, match="No scale transformations available."):
            plotter.scale_transformations(input_latent_variable=1)

    def test_scale_transformations_invalid_latent_var(self, plotter):
        """Test that an error is raised when plotting scale transformations with invalid latent variable."""
        plotter.model.latent_variables = 1
        with pytest.raises(TypeError, match="Cannot plot latent variable 2 with a 1-dimensional model."):
            plotter.scale_transformations(input_latent_variable=2)

    def test_log_likelihood(self, plotter):
        """Test the log_likelihood method."""
        plotter.model.latent_variables = 1
        plotter.model.evaluate.log_likelihood = MagicMock(return_value=torch.rand(100, 5))
        data = torch.randint(0, 2, (1, 5)).float()
        fig = plotter.log_likelihood(data=data)
        assert fig is not None
        assert fig.data is not None

    def test_log_likelihood_with_sum_score(self, plotter):
        """Test the log_likelihood method with expected_sum_score=True."""
        plotter.model.latent_variables = 1
        plotter.model.evaluate.log_likelihood = MagicMock(return_value=torch.rand(100, 5))
        plotter.model.expected_scores = MagicMock(return_value=torch.rand(100))
        data = torch.randint(0, 2, (1, 5)).float()
        fig = plotter.log_likelihood(data=data, expected_sum_score=True)
        assert fig is not None
        assert fig.data is not None
