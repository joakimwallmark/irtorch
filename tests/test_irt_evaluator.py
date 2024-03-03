from unittest.mock import MagicMock, patch
import torch
import pytest
from irtorch.irt_evaluator import IRTEvaluator, IRTScorer
from irtorch.estimation_algorithms import BaseIRTAlgorithm
from irtorch.models import BaseIRTModel
from irtorch.quantile_mv_normal import QuantileMVNormal


@pytest.fixture
def irt_evaluator(latent_variables):
    # Create a mock instance of AEIRTNeuralNet
    item_categories = [2, 3]
    mock_algorithm = MagicMock(spec=BaseIRTAlgorithm)
    mock_algorithm.one_hot_encoded = False
    mock_algorithm.train_data = torch.cat(
        [
            torch.randint(0, item_cat, (30, 1))
            for item_cat in item_categories
        ],
        dim=1,
    )
    mock_algorithm.training_z_scores = torch.randn(30, latent_variables)

    # Mock fix_missing_values method 
    def fix_missing_values_mock(input_tensor):
        return input_tensor

    mock_algorithm.fix_missing_values = MagicMock(side_effect=fix_missing_values_mock)

    def model_forward_mock(input_tensor: torch.Tensor):
        logits = [
            torch.randn(input_tensor.shape[0], category)
            for category in item_categories
        ]
        return torch.cat(logits, dim=1)

    # Mock item_probabilities method based on input
    def item_probabilities_mock(input_tensor):
        logits = torch.randn(input_tensor.shape[0], len(item_categories), max(item_categories))
        for item_id, item_cats in enumerate(item_categories):
            logits[:, item_id, item_cats:] = -torch.inf
        return torch.softmax(logits, dim = 2)
    
    mock_model = MagicMock(spec=BaseIRTModel, side_effect=model_forward_mock)
    mock_model.item_probabilities = MagicMock(side_effect=item_probabilities_mock)
    mock_model.item_categories = item_categories
    mock_model.modeled_item_responses = item_categories
    mock_model.model_missing = False
    mock_model.mc_correct = None

    # Mock z_scores method based on input
    def z_scores_mock(input_tensor):
        z_scores = torch.randn(input_tensor.shape[0], latent_variables)
        return z_scores

    mock_algorithm.z_scores = MagicMock(side_effect=z_scores_mock)
    # Create a mock instance of IRTScorer
    mock_scorer = MagicMock(spec=IRTScorer)

    # Mock z_scores method based on input
    def latent_scores(data, **kwargs):
        z_scores = torch.randn(data.shape[0], latent_variables)
        return z_scores

    # Mock bit_score_distance method based on input
    def bit_scores_from_z_mock(z, **kwargs):
        return torch.randn(z.shape[0], z.shape[1]).abs() * 10, torch.randn(1, z.shape[1])

    # Mock min_max_z_for_integration_mock
    def min_max_z_for_integration_mock(z):
        z_min = z.min(dim=0)[0]
        z_max = z.max(dim=0)[0]
        z_stds = z.std(dim=0)
        return z_min - z_stds, z_max + z_stds

    # Mock latent density pdf function
    def pdf_mock(z):
        return torch.rand(z.shape[0])
    mock_scorer.latent_scores = MagicMock(side_effect=latent_scores)
    mock_scorer.bit_scores_from_z = MagicMock(side_effect=bit_scores_from_z_mock)
    mock_scorer.min_max_z_for_integration = MagicMock(
        side_effect=min_max_z_for_integration_mock
    )
    mock_scorer.latent_density = QuantileMVNormal()
    mock_scorer.latent_density.pdf = MagicMock(side_effect=pdf_mock)
    return IRTEvaluator(mock_model, mock_algorithm, mock_scorer)


# add test for the _evaluate_data_z_input method
def test_evaluate_data_z_input(irt_evaluator: IRTEvaluator):
    # Create some synthetic test data
    data = torch.cat(
        [
            torch.randint(0, item_cat, (30, 1))
            for item_cat in irt_evaluator.model.item_categories
        ],
        dim=1,
    )

    # Call the method with data and z as None
    result_data, result_z = irt_evaluator._evaluate_data_z_input(data, None, 'NN')

    # Check the shape of the output data and z
    assert result_data.shape == data.shape
    assert result_z.shape == irt_evaluator.algorithm.training_z_scores.shape

    # Call the method with data as None
    result_data, result_z = irt_evaluator._evaluate_data_z_input(None, None, 'NN')

    # Check the shape of the output data and z
    assert result_data.shape == irt_evaluator.algorithm.train_data.shape
    assert result_z.shape == irt_evaluator.algorithm.training_z_scores.shape

def test_probabilities_from_grouped_z(irt_evaluator: IRTEvaluator, latent_variables):
    # Create some synthetic z scores
    grouped_z = (
        torch.randn(10, latent_variables),
        torch.randn(10, latent_variables),
        torch.randn(10, latent_variables),
    )

    # Call the method
    probabilities = irt_evaluator._grouped_z_probabilities(grouped_z)
    assert probabilities.shape == (3, 2, 3)
    assert torch.allclose(probabilities.sum(dim=2), torch.ones(probabilities.shape[0], probabilities.shape[1]), atol=1e-7)

def test_probabilities_from_grouped_data(irt_evaluator: IRTEvaluator):
    # Create synthetic test data groups
    grouped_data = (
        torch.cat(
            [
                torch.randint(0, item_cat, (5, 1))
                for item_cat in irt_evaluator.model.item_categories
            ],
            dim=1,
        ),
        torch.cat(
            [
                torch.randint(0, item_cat, (5, 1))
                for item_cat in irt_evaluator.model.item_categories
            ],
            dim=1,
        ),
        torch.cat(
            [
                torch.randint(0, item_cat, (5, 1))
                for item_cat in irt_evaluator.model.item_categories
            ],
            dim=1,
        ),
    )

    # Call the method
    probabilities = irt_evaluator._grouped_data_probabilities(grouped_data)

    assert probabilities.shape == torch.Size([3, 2, 3])
    assert torch.allclose(probabilities.sum(dim=2), torch.ones_like(probabilities[:, :, 0]), atol=1e-7)

@pytest.mark.parametrize("scale", ["bit", "z"])
def test_latent_group_probabilities(irt_evaluator: IRTEvaluator, scale):
    # Create some synthetic test data
    data = torch.cat(
        [
            torch.randint(0, item_cat, (30, 1))
            for item_cat in irt_evaluator.model.item_categories
        ],
        dim=1,
    )

    groups = 3
    (
        grouped_data_probabilities,
        grouped_model_probabilities,
        group_averages,
    ) = irt_evaluator.latent_group_probabilities(
        groups=groups, data=data, scale=scale, latent_variable=1
    )

    # Check the number of groups
    assert grouped_data_probabilities.shape == grouped_model_probabilities.shape
    assert grouped_data_probabilities.shape == torch.Size([3, 2, 3])
    assert torch.allclose(grouped_data_probabilities.sum(dim=2), torch.ones_like(grouped_data_probabilities[:, :, 0]), atol=1e-7)
    assert torch.allclose(grouped_model_probabilities.sum(dim=2), torch.ones_like(grouped_model_probabilities[:, :, 0]), atol=1e-7)
    assert group_averages.shape == (groups,)

    if scale == "bit":
        # check if bit score was called
        irt_evaluator.scorer.bit_scores_from_z.assert_called_once()

def test_group_fit_residuals(irt_evaluator: IRTEvaluator):
    data = torch.cat(
        [
            torch.randint(0, item_cat, (30, 1))
            for item_cat in irt_evaluator.model.item_categories
        ],
        dim=1,
    )

    residuals, mid_points = irt_evaluator.group_fit_residuals(data=data, groups=3, latent_variable=1)

    assert residuals.shape == (3, 2, 3)
    assert mid_points.shape == (3,)
    assert irt_evaluator.model.item_probabilities.call_count == 3

@pytest.mark.parametrize(
    "latent_density_method", ["data", "encoder sampling", "qmvn", "gmm"]
)
def test_sum_score_probabilities(irt_evaluator: IRTEvaluator, latent_density_method):
    if latent_density_method == "encoder sampling":
        with pytest.raises(ValueError):
            total_score_probs = irt_evaluator.sum_score_probabilities(
                latent_density_method=latent_density_method
            )
    else:
        total_score_probs = irt_evaluator.sum_score_probabilities(
            latent_density_method=latent_density_method, trapezoidal_segments=5
        )
        # We should have 4 scores since we have
        # mock_neuralnet.decoder.modeled_item_responses = [2, 3]
        assert total_score_probs.shape == (4,)
        # They should add up to 1
        assert torch.isclose(total_score_probs.sum(), torch.tensor(1.0), atol=1e-6)

def test_log_likelihood(irt_evaluator: IRTEvaluator):
    # Create synthetic test data
    data = torch.cat(
        [
            torch.randint(0, item_cat, (30, 1))
            for item_cat in irt_evaluator.model.item_categories
        ],
        dim=1,
    )

    # Mock log_likelihood method
    def log_likelihood_mock(data, output, loss_reduction):
        t1 = torch.tensor([-0.5, -1.0]).repeat(data.shape[0] // 2)
        t2 = torch.tensor([-0.1, -0.2]).repeat(data.shape[0] - (data.shape[0] // 2))
        return torch.cat([t1, t2])

    irt_evaluator.model.log_likelihood = MagicMock(side_effect=log_likelihood_mock)

    # latent_group_fit
    # Test with default parameters
    group_mean_likelihoods = irt_evaluator.group_fit_log_likelihood(data=data)
    assert group_mean_likelihoods.shape == (10,)  # Default number of groups is 10
    assert group_mean_likelihoods.dtype == torch.float32
    assert torch.allclose(group_mean_likelihoods[:5], torch.full((5,), -1.5)) # first half of groups
    assert torch.allclose(group_mean_likelihoods[5:], torch.full((5,), -0.3)) # second half of groups
    # Test with specified parameters
    group_mean_likelihoods = irt_evaluator.group_fit_log_likelihood(groups=5, latent_variable=1, data=data, z_estimation_method="ML")
    assert group_mean_likelihoods.shape == (5,)  # We specified 5 groups
    assert group_mean_likelihoods.dtype == torch.float32
    assert torch.allclose(group_mean_likelihoods[:2], torch.full((2,), -1.5)) # first half of groups
    assert torch.allclose(group_mean_likelihoods[2], torch.full((1,), -0.9)) # first half of groups
    assert torch.allclose(group_mean_likelihoods[3:], torch.full((2,), -0.3)) # second half of groups

    # respondent level
    # Test with default parameters
    person_likelihoods = irt_evaluator.log_likelihood(data=data, reduction="sum", level="respondent")
    assert person_likelihoods.shape == (30,)  # We have 30 respondents
    assert person_likelihoods.dtype == torch.float32
    assert torch.allclose(person_likelihoods[:15], torch.full((15,), -1.5)) # first half of respondents
    assert torch.allclose(person_likelihoods[15:], torch.full((15,), -0.3)) # second half of respondents

    # item level
    # Test with default parameters
    item_likelihoods = irt_evaluator.log_likelihood(data=data, reduction="mean", level="item")
    assert item_likelihoods.shape == (2,)  # We have 2 items
    assert item_likelihoods.dtype == torch.float32
    assert torch.isclose(item_likelihoods[0], torch.tensor(-0.3))
    assert torch.isclose(item_likelihoods[1], torch.tensor(-0.6))


def test_accuracy(irt_evaluator: IRTEvaluator):
    # Create synthetic test data
    data = torch.cat(
        [
            torch.randint(0, item_cat, (30, 1))
            for item_cat in irt_evaluator.model.item_categories
        ],
        dim=1,
    )

    # Mock probabilities
    def item_probabilities_mock(z):
        # static 3D tensor with dimensions (respondents, items, item categories)
        t1 = torch.tensor(([0.55, 0.45, 0.0], [0.2, 0.35, 0.45])).unsqueeze(0).repeat(z.shape[0] // 2, 1, 1)
        t2 = torch.tensor(([0.15, 0.85, 0.0], [0.6, 0.05, 0.35])).unsqueeze(0).repeat(z.shape[0] - (z.shape[0] // 2), 1, 1)
        return torch.cat([t1, t2])

    irt_evaluator.model.item_probabilities = MagicMock(side_effect=item_probabilities_mock)

    # overall
    accuracy = irt_evaluator.accuracy(data=data, level="all")
    assert accuracy.shape == ()  # We have 30 respondents
    assert accuracy.dtype == torch.float32
    assert torch.all((accuracy >= 0) & (accuracy <= 1)), "All values are not between 0 and 1"
    
    # respondent level
    accuracy = irt_evaluator.accuracy(data=data, level="respondent")
    assert accuracy.shape == (30,)  # We have 30 respondents
    assert accuracy.dtype == torch.float32
    assert torch.all((accuracy >= 0) & (accuracy <= 1)), "All values are not between 0 and 1"

    # item level
    item_likelihoods = irt_evaluator.accuracy(data=data, level="item")
    assert item_likelihoods.shape == (2,)  # We have 2 items
    assert item_likelihoods.dtype == torch.float32
    assert torch.all((accuracy >= 0) & (accuracy <= 1)), "All values are not between 0 and 1"


def test_residuals(irt_evaluator: IRTEvaluator):
    # Create synthetic test data
    data = torch.tensor(
        (
            [0, 2],
            [1, 1],
            [1, 0],
            [0, 1],
            [0, 2],
        )
    ).float()

    # Mock probabilities
    def item_probabilities_mock(z):
        # static 3D tensor with dimensions (respondents, items, item categories)
        t1 = torch.tensor(([0.55, 0.45, 0.0], [0.2, 0.35, 0.45])).unsqueeze(0).repeat(z.shape[0] // 2, 1, 1)
        t2 = torch.tensor(([0.15, 0.85, 0.0], [0.6, 0.05, 0.35])).unsqueeze(0).repeat(z.shape[0] - (z.shape[0] // 2), 1, 1)
        return torch.cat([t1, t2])

    irt_evaluator.model.item_probabilities = MagicMock(side_effect=item_probabilities_mock)
    # For scored tests, we actual vs model expected scores


    # For MC tests, we have residuals for each MC option
    irt_evaluator.model.mc_correct = [0, 2]
    # non-averaged
    residuals = irt_evaluator.residuals(data=data)
    assert residuals.shape == (5, 2)  # We have 30 respondents
    assert torch.allclose(residuals, torch.tensor([
        [0.4500, 0.5500],
        [0.5500, 0.6500],
        [0.1500, 0.4000],
        [0.8500, 0.9500],
        [0.8500, 0.6500]]
    )), "Residuals are not correct"

    # overall
    residuals = irt_evaluator.residuals(data=data, average_over="everything")
    assert residuals.shape == ()  # We have 30 respondents
    assert torch.isclose(residuals, torch.tensor(0.6050)), "Residuals are not correct"

    # respondent level
    residuals = irt_evaluator.residuals(data=data, average_over="items")
    assert residuals.shape == (5,)  # We have 5 respondents
    assert torch.allclose(residuals, torch.tensor([0.5000, 0.6000, 0.2750, 0.9000, 0.7500])), "Residuals are not correct"

    # item level
    residuals = irt_evaluator.residuals(data=data, average_over="respondents")
    assert residuals.shape == (2,)  # We have 2 items
    assert torch.allclose(residuals, torch.tensor([0.5700, 0.6400])), "Residuals are not correct"
