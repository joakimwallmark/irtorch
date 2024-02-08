from unittest.mock import MagicMock
import torch
import pytest
from matplotlib.figure import Figure
from irtorch.estimation_algorithms import BaseIRTAlgorithm
from irtorch.irt_plotter import IRTScorer, IRTPlotter, IRTEvaluator
from irtorch.models import BaseIRTModel


@pytest.fixture
def irt_plotter(latent_variables):
    item_categories = [2, 3]
    mock_algorithm = MagicMock(spec=BaseIRTAlgorithm)
    mock_algorithm.train_data = torch.cat(
        [
            torch.randint(0, item_cat, (30, 1))
            for item_cat in item_categories
        ],
        dim=1,
    )

    mock_model = MagicMock(spec=BaseIRTModel)
    mock_model.item_categories = item_categories
    mock_model.modeled_item_responses = item_categories

    # Mock item_probabilities method based on input
    def item_probabilities_mock(input_tensor):
        # Create a list of 2D tensors
        logits = torch.randn(input_tensor.shape[0], len(item_categories), max(item_categories))
        for item_id, item_cats in enumerate(item_categories):
            logits[:, item_id, item_cats:] = -torch.inf
        return torch.softmax(logits, dim = 2)

    mock_model.item_probabilities = MagicMock(side_effect=item_probabilities_mock)

    def z_scores_mock(input_tensor):
        z_scores = torch.randn(input_tensor.shape[0], latent_variables)
        return z_scores

    mock_algorithm.z_scores = MagicMock(side_effect=z_scores_mock)

    mock_scorer = MagicMock(spec=IRTScorer)

    # Mock entropy_distance method based on input
    def entropy_distance_mock(input_tensor, steps, entropy_method):
        entropy_distances = torch.randn(input_tensor.shape[0]).abs() * 10
        return entropy_distances.unsqueeze(1)

    mock_scorer.entropy_distance = MagicMock(side_effect=entropy_distance_mock)

    mock_evaluator = MagicMock(spec=IRTEvaluator)

    return IRTPlotter(mock_model, mock_algorithm, mock_evaluator, mock_scorer)


# def test_plot_training_history(irt_plotter: IRTPlotter):
#     # Check for AttributeError
#     with pytest.raises(AttributeError):
#         irt_plotter.plot_training_history()

#     # Create a mock training history
#     training_history = {
#         "train_loss": [1, 2, 3],
#         "validation_loss": [2, 3, 4],
#     }
#     irt_plotter.algorithm.training_history = training_history

#     # Test plotting all available measures
#     figure, axs = irt_plotter.plot_training_history()
#     assert isinstance(figure, Figure)
#     assert len(axs) == 1  # Should be 3 subplots

#     measures = {
#         "Loss function": {
#             "train": "train_loss",
#             "validation": "validation_loss",
#         }
#     }

#     for ax, (_, measure) in zip(axs, measures.items()):
#         expected_lines = 0
#         if (
#             measure.get("train")
#             and len(irt_plotter.algorithm.training_history.get(measure["train"], []))
#             > 0
#         ):
#             expected_lines += 1
#         if (
#             len(irt_plotter.algorithm.training_history.get(measure["validation"], []))
#             > 0
#         ):
#             expected_lines += 1
#         assert len(ax.lines) == expected_lines

#     # Test plotting specific measures
#     figure, axs = irt_plotter.plot_training_history(["Loss function"])
#     assert len(axs) == 1  # Only one measure is being plotted
#     train_line, validation_line = axs[0].lines
#     assert (train_line.get_ydata() == training_history["train_loss"]).all()
#     assert (validation_line.get_ydata() == training_history["validation_loss"]).all()
