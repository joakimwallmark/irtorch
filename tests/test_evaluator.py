from unittest.mock import MagicMock, patch
import torch
import numpy as np
import pandas as pd
import pytest
from irtorch.models import BaseIRTModel
from irtorch.evaluator import Evaluator
from irtorch.estimation_algorithms import AE
from irtorch.quantile_mv_normal import QuantileMVNormal
from irtorch.torch_modules.gaussian_mixture_model import GaussianMixtureModel

@pytest.fixture
def algorithm(latent_variables):
    item_categories = [2, 3]
    mock_algorithm = MagicMock(spec=AE)
    mock_algorithm.one_hot_encoded = False
    mock_algorithm.train_data = torch.cat(
        [
            torch.randint(0, item_cat, (30, 1))
            for item_cat in item_categories
        ],
        dim=1,
    )
    mock_algorithm.training_theta_scores = torch.randn(30, latent_variables)
    # Mock theta_scores method based on input
    def theta_scores_mock(input_tensor):
        theta_scores = torch.randn(input_tensor.shape[0], latent_variables)
        return theta_scores

    mock_algorithm.theta_scores = MagicMock(side_effect=theta_scores_mock)
    return mock_algorithm

@pytest.fixture
def irt_model(latent_variables, algorithm: AE):
    item_categories = [2, 3]
    items = 2

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
    
    # Mock theta_scores method based on input
    def latent_scores(data, **kwargs):
        theta_scores = torch.randn(data.shape[0], latent_variables)
        return theta_scores
    
    def transform_theta_mock(theta):
        return torch.randn(theta.shape[0],theta.shape[1]).abs() * 10

    mock_model = MagicMock(spec=BaseIRTModel, side_effect=model_forward_mock)
    mock_model.algorithm = algorithm
    mock_model.scale = [5]
    mock_model.transform_theta = MagicMock(side_effect=transform_theta_mock)
    mock_model.item_probabilities = MagicMock(side_effect=item_probabilities_mock)
    mock_model.latent_scores = MagicMock(side_effect=latent_scores)
    mock_model.item_categories = item_categories
    mock_model.items = items
    mock_model.mc_correct = None
    mock_model.latent_variables = latent_variables

    return mock_model

@pytest.fixture
def evaluation(irt_model: BaseIRTModel):
    def pdf_mock(theta):
        return torch.rand(theta.shape[0])

    evaluation = Evaluator(irt_model)
    evaluation.latent_density = QuantileMVNormal()
    evaluation.latent_density.pdf = MagicMock(side_effect=pdf_mock)

    return evaluation

def test_mutual_information_difference(evaluation: Evaluator):
    data = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 2.0]])
    
    def probabilities_from_output_mock(input_tensor):
        # tensor that matches the shape of the data tensor
        return torch.tensor([
            [[0.55, 0.45, 0.0], [0.20, 0.35, 0.45]],
            [[0.35, 0.65, 0.0], [0.10, 0.40, 0.50]],
            [[0.25, 0.75, 0.0], [0.10, 0.20, 0.70]],
            [[0.15, 0.85, 0.0], [0.05, 0.05, 0.90]],
        ])

    evaluation.model.probabilities_from_output = MagicMock(side_effect=probabilities_from_output_mock)
    
    mid, amid, _ = evaluation.mutual_information_difference(data=data)
    assert mid.equals(amid), "Mid and Amid are not equal"
    expected_mid = pd.DataFrame([
        [0.8042, 0.3031],
        [0.3031, 0.9862]
    ])
    assert np.allclose(mid.values, expected_mid.values, atol=1e-4), "Mid is not correct"

def test__evaluate_data_theta_input(evaluation: Evaluator):
    # Create some synthetic test data
    data = torch.cat(
        [
            torch.randint(0, item_cat, (30, 1))
            for item_cat in evaluation.model.item_categories
        ],
        dim=1,
    ).float()
    data[5:7, 1] = torch.nan

    # Call the method with data and theta as None
    result_data, result_theta, missing_mask = evaluation._evaluate_data_theta_input(data, None, theta_estimation="NN")

    # Check the shape of the output data and theta
    assert result_data.shape == data.shape
    assert result_theta.shape == evaluation.model.algorithm.training_theta_scores.shape
    assert missing_mask.shape == data.shape
    assert missing_mask[5:7, 1].all()

    # Call the method with data as None
    result_data, result_theta, missing_mask = evaluation._evaluate_data_theta_input(None, None, theta_estimation="NN")

    # Check the shape of the output data and theta
    assert result_data.shape == evaluation.model.algorithm.train_data.shape
    assert result_theta.shape == evaluation.model.algorithm.training_theta_scores.shape
    assert missing_mask.shape == data.shape
    assert (~missing_mask).all()

def test_probabilities_from_grouped_theta(evaluation: Evaluator, latent_variables):
    # Create some synthetic theta scores
    grouped_theta = (
        torch.randn(10, latent_variables),
        torch.randn(10, latent_variables),
        torch.randn(10, latent_variables),
    )

    # Call the method
    probabilities = evaluation._grouped_theta_probabilities(grouped_theta)
    assert probabilities.shape == (3, 2, 3)
    assert torch.allclose(probabilities.sum(dim=2), torch.ones(probabilities.shape[0], probabilities.shape[1]), atol=1e-7)

def test_probabilities_from_grouped_data(evaluation: Evaluator):
    # Create synthetic test data groups
    grouped_data = (
        torch.cat(
            [
                torch.randint(0, item_cat, (5, 1))
                for item_cat in evaluation.model.item_categories
            ],
            dim=1,
        ),
        torch.cat(
            [
                torch.randint(0, item_cat, (5, 1))
                for item_cat in evaluation.model.item_categories
            ],
            dim=1,
        ),
        torch.cat(
            [
                torch.randint(0, item_cat, (5, 1))
                for item_cat in evaluation.model.item_categories
            ],
            dim=1,
        ),
    )

    # Call the method
    probabilities = evaluation._grouped_data_probabilities(grouped_data)

    assert probabilities.shape == torch.Size([3, 2, 3])
    assert torch.allclose(probabilities.sum(dim=2), torch.ones_like(probabilities[:, :, 0]), atol=1e-7)

@pytest.mark.parametrize("rescale", [True, False])
def test_latent_group_probabilities(evaluation: Evaluator, rescale):
    # Create some synthetic test data
    data = torch.cat(
        [
            torch.randint(0, item_cat, (30, 1))
            for item_cat in evaluation.model.item_categories
        ],
        dim=1,
    ).float()

    groups = 3
    (
        grouped_data_probabilities,
        grouped_model_probabilities,
        group_averages,
    ) = evaluation.latent_group_probabilities(groups=groups, data=data, rescale=rescale, latent_variable=1)

    # Check the number of groups
    assert grouped_data_probabilities.shape == grouped_model_probabilities.shape
    assert grouped_data_probabilities.shape == torch.Size([3, 2, 3])
    assert torch.allclose(grouped_data_probabilities.sum(dim=2), torch.ones_like(grouped_data_probabilities[:, :, 0]), atol=1e-7)
    assert torch.allclose(grouped_model_probabilities.sum(dim=2), torch.ones_like(grouped_model_probabilities[:, :, 0]), atol=1e-7)
    assert group_averages.shape == (groups,)

    if rescale:
        evaluation.model.transform_theta.assert_called_once()

def test_group_fit_residuals(evaluation: Evaluator):
    data = torch.cat(
        [
            torch.randint(0, item_cat, (30, 1))
            for item_cat in evaluation.model.item_categories
        ],
        dim=1,
    )

    residuals, mid_points = evaluation.group_fit_residuals(data=data, groups=3, latent_variable=1)

    assert residuals.shape == (3, 2, 3)
    assert mid_points.shape == (3,)
    assert evaluation.model.item_probabilities.call_count == 3

@pytest.mark.parametrize(
    "latent_density_method", ["data", "encoder sampling", "qmvn", "gmm"]
)
def test_sum_score_probabilities(evaluation: Evaluator, latent_density_method):
    # mock _cv_gaussian_mixture_model method
    def _cv_gaussian_mixture_model_mock(data, cv_n_components):
        return GaussianMixtureModel(n_components=cv_n_components[0], n_features=data.shape[1])
    
    evaluation._cv_gaussian_mixture_model = MagicMock(side_effect=_cv_gaussian_mixture_model_mock)
    
    if latent_density_method == "encoder sampling":
        with pytest.raises(ValueError):
            total_score_probs = evaluation.sum_score_probabilities(
                latent_density_method=latent_density_method
            )
    else:
        total_score_probs = evaluation.sum_score_probabilities(
            latent_density_method=latent_density_method, trapezoidal_segments=5
        )
        # We should have 4 scores since we have item_categories = [2, 3]
        assert total_score_probs.shape == (4,)
        # They should add up to 1
        assert torch.isclose(total_score_probs.sum(), torch.tensor(1.0), atol=1e-6)

def test_log_likelihood(evaluation: Evaluator):
    # Create synthetic test data
    data = torch.cat(
        [
            torch.randint(0, item_cat, (30, 1))
            for item_cat in evaluation.model.item_categories
        ],
        dim=1,
    )

    # Mock log_likelihood method
    def log_likelihood_mock(data, output, missing_mask, loss_reduction):
        t1 = torch.tensor([-0.5, -1.0]).repeat(data.shape[0] // 2)
        t2 = torch.tensor([-0.1, -0.2]).repeat(data.shape[0] - (data.shape[0] // 2))
        return torch.cat([t1, t2])

    evaluation.model.log_likelihood = MagicMock(side_effect=log_likelihood_mock)

    # latent_group_fit
    # Test with default parameters
    group_mean_likelihoods = evaluation.group_fit_log_likelihood(data=data)
    assert group_mean_likelihoods.shape == (10,)  # Default number of groups is 10
    assert group_mean_likelihoods.dtype == torch.float32
    assert torch.allclose(group_mean_likelihoods[:5], torch.full((5,), -1.5)) # first half of groups
    assert torch.allclose(group_mean_likelihoods[5:], torch.full((5,), -0.3)) # second half of groups
    # Test with specified parameters
    group_mean_likelihoods = evaluation.group_fit_log_likelihood(groups=5, latent_variable=1, data=data, theta_estimation="ML")
    assert group_mean_likelihoods.shape == (5,)  # We specified 5 groups
    assert group_mean_likelihoods.dtype == torch.float32
    assert torch.allclose(group_mean_likelihoods[:2], torch.full((2,), -1.5)) # first half of groups
    assert torch.allclose(group_mean_likelihoods[2], torch.full((1,), -0.9)) # first half of groups
    assert torch.allclose(group_mean_likelihoods[3:], torch.full((2,), -0.3)) # second half of groups

    # respondent level
    # Test with default parameters
    person_likelihoods = evaluation.log_likelihood(data=data, reduction="sum", level="respondent")
    assert person_likelihoods.shape == (30,)  # We have 30 respondents
    assert person_likelihoods.dtype == torch.float32
    assert torch.allclose(person_likelihoods[:15], torch.full((15,), -1.5)) # first half of respondents
    assert torch.allclose(person_likelihoods[15:], torch.full((15,), -0.3)) # second half of respondents

    # item level
    # Test with default parameters
    item_likelihoods = evaluation.log_likelihood(data=data, reduction="mean", level="item")
    assert item_likelihoods.shape == (2,)  # We have 2 items
    assert item_likelihoods.dtype == torch.float32
    assert torch.isclose(item_likelihoods[0], torch.tensor(-0.3))
    assert torch.isclose(item_likelihoods[1], torch.tensor(-0.6))


def test_accuracy(evaluation: Evaluator):
    # Create synthetic test data
    data = torch.cat(
        [
            torch.randint(0, item_cat, (30, 1))
            for item_cat in evaluation.model.item_categories
        ],
        dim=1,
    ).float()
    data[5:7, 1] = torch.nan

    # Mock probabilities
    def item_probabilities_mock(theta):
        # static 3D tensor with dimensions (respondents, items, item categories)
        t1 = torch.tensor(([0.55, 0.45, 0.0], [0.2, 0.35, 0.45])).unsqueeze(0).repeat(theta.shape[0] // 2, 1, 1)
        t2 = torch.tensor(([0.15, 0.85, 0.0], [0.6, 0.05, 0.35])).unsqueeze(0).repeat(theta.shape[0] - (theta.shape[0] // 2), 1, 1)
        return torch.cat([t1, t2])

    evaluation.model.item_probabilities = MagicMock(side_effect=item_probabilities_mock)

    # overall
    accuracy = evaluation.accuracy(data=data, level="all")
    assert accuracy.shape == ()
    assert accuracy.dtype == torch.float32
    assert torch.all((accuracy >= 0) & (accuracy <= 1)), "All values are not between 0 and 1"
    
    # respondent level
    accuracy = evaluation.accuracy(data=data, level="respondent")
    assert accuracy.shape == (30,)  # We have 30 respondents
    assert accuracy.dtype == torch.float32
    assert torch.all((accuracy >= 0) & (accuracy <= 1)), "All values are not between 0 and 1"

    # item level
    item_likelihoods = evaluation.accuracy(data=data, level="item")
    assert item_likelihoods.shape == (2,)  # We have 2 items
    assert item_likelihoods.dtype == torch.float32
    assert torch.all((accuracy >= 0) & (accuracy <= 1)), "All values are not between 0 and 1"

def test_precisions(evaluation: Evaluator):
    # Create synthetic test data
    data = torch.cat(
        [
            torch.randint(0, item_cat, (30, 1))
            for item_cat in evaluation.model.item_categories
        ],
        dim=1,
    ).float()
    # Introduce missing responses for testing (e.g., missing responses in the second item)
    data[5:7, 1] = torch.nan

    # Mock probabilities
    def item_probabilities_mock(theta):
        # static 3D tensor with dimensions (respondents, items, item categories)
        t1 = torch.tensor(([0.55, 0.45, 0.0], [0.2, 0.35, 0.45])).unsqueeze(0).repeat(theta.shape[0] // 2, 1, 1)
        t2 = torch.tensor(([0.15, 0.85, 0.0], [0.6, 0.05, 0.35])).unsqueeze(0).repeat(theta.shape[0] - (theta.shape[0] // 2), 1, 1)
        return torch.cat([t1, t2])
    
    evaluation.model.item_probabilities = MagicMock(side_effect=item_probabilities_mock)

    df = evaluation.predictions(data=data)

    assert isinstance(df, pd.DataFrame), "predictions should return a pandas DataFrame"

    # The number of rows should equal the number of items (in this case 2)
    # and the DataFrame should have 6 columns for the metrics.
    expected_rows = len(evaluation.model.item_categories)
    expected_columns = ["precision", "recall", "f1", "w_precision", "w_recall", "w_f1"]
    assert df.shape == (expected_rows, len(expected_columns)), f"Expected DataFrame shape ({expected_rows}, {len(expected_columns)}), got {df.shape}"

    # Check that all expected columns are present.
    for col in expected_columns:
        assert col in df.columns, f"Missing expected column: {col}"
        # Ensure that all metric values are within [0, 1]
        assert df[col].between(0, 1).all(), f"Some values in {col} are outside the range [0, 1]"


def test_residuals(evaluation: Evaluator):
    # Create synthetic test data
    data = torch.tensor(
        (
            [0., 2.],
            [1., 1.],
            [1., 0.],
            [0., -1.],
            [0., torch.nan],
        )
    )

    # Mock probabilities
    def item_probabilities_mock(theta):
        # static 3D tensor with dimensions (respondents, items, item categories)
        t1 = torch.tensor(([0.55, 0.45, 0.0], [0.2, 0.35, 0.45])).unsqueeze(0).repeat(theta.shape[0] // 2, 1, 1)
        t2 = torch.tensor(([0.15, 0.85, 0.0], [0.6, 0.05, 0.35])).unsqueeze(0).repeat(theta.shape[0] - (theta.shape[0] // 2), 1, 1)
        return torch.cat([t1, t2])

    evaluation.model.item_probabilities = MagicMock(side_effect=item_probabilities_mock)
    # For scored tests, we actual vs model expected scores


    # For MC tests, we have residuals for each MC option
    evaluation.model.mc_correct = [0, 2]
    # non-averaged
    residuals = evaluation.residuals(data=data)
    assert residuals.shape == (5, 2)
    assert torch.allclose(residuals, torch.tensor([
        [0.4500, 0.5500],
        [0.5500, 0.6500],
        [0.1500, 0.4000],
        [0.8500, torch.nan],
        [0.8500, torch.nan]],
    ), equal_nan=True), "Residuals are not correct"

    # overall
    residuals = evaluation.residuals(data=data, average_over="everything")
    assert residuals.shape == ()  # We have 30 respondents
    assert torch.isclose(residuals, torch.tensor(0.556249976)), "Residuals are not correct"

    # respondent level
    residuals = evaluation.residuals(data=data, average_over="items")
    assert residuals.shape == (5,)  # We have 5 respondents
    assert torch.allclose(residuals, torch.tensor([0.5000, 0.6000, 0.2750, 0.8500, 0.8500])), "Residuals are not correct"

    # item level
    residuals = evaluation.residuals(data=data, average_over="respondents")
    assert residuals.shape == (2,)  # We have 2 items
    assert torch.allclose(residuals, torch.tensor([0.5700, 0.53333336114])), "Residuals are not correct"

def test_infit_outfit(evaluation: Evaluator):
    data = torch.tensor([[0, 2], [1, 1]]).float()
    theta = torch.randn(2, evaluation.model.latent_variables)

    def item_probabilities_mock(theta):
        return torch.tensor([[[0.55, 0.45, 0.0], [0.2, 0.35, 0.45]], [[0.15, 0.85, 0.0], [0.6, 0.05, 0.35]]])

    
    evaluation.model.item_probabilities = MagicMock(side_effect=item_probabilities_mock)
    
    def expected_scores_mock(theta, **kwargs):
        return torch.tensor(([0.45, 0.35 + 2 * 0.45], [0.85, 0.05+2 * 0.35]))

    evaluation.model.expected_scores = MagicMock(side_effect=expected_scores_mock)

    infit, outfit = evaluation.infit_outfit(data, theta, level = "item")
    assert torch.allclose(infit, torch.tensor([0.6000000, 0.4237288])), "Infit is incorrect"
    assert torch.allclose(outfit, torch.tensor([0.4973261, 0.5139347])), "Outfit is incorrect"

    infit, outfit = evaluation.infit_outfit(data, theta, level = "respondent")
    assert torch.allclose(infit, torch.tensor([0.9161677, 0.0837438])), "Infit is incorrect"
    assert torch.allclose(outfit, torch.tensor([0.8878143, 0.1234465])), "Outfit is incorrect"

    def expected_scores_mock_mc(theta, **kwargs):
        return torch.tensor(([0.55, 0.45], [0.15, 0.35]))

    evaluation.model.expected_scores = MagicMock(side_effect=expected_scores_mock_mc)

    evaluation.model.mc_correct = [0, 2]
    infit, outfit = evaluation.infit_outfit(data, theta, level = "item")
    assert torch.allclose(infit, torch.tensor([0.6000000, 0.8947369])), "Infit is incorrect"
    assert torch.allclose(outfit, torch.tensor([0.4973262, 0.8803419])), "Outfit is incorrect"

    infit, outfit = evaluation.infit_outfit(data, theta, level = "respondent")
    assert torch.allclose(infit, torch.tensor([1.0202020, 0.4084507])), "Infit is incorrect"
    assert torch.allclose(outfit, torch.tensor([1.0202019, 0.3574660])), "Outfit is incorrect"

    # With missing data
    data[0, 1] = torch.nan
    infit, outfit = evaluation.infit_outfit(data, theta, level = "item")
    assert torch.allclose(infit, torch.tensor([0.6000000, 0.53846150])), "Infit is incorrect"
    assert torch.allclose(outfit, torch.tensor([0.4973262, 0.53846150])), "Outfit is incorrect"
    infit, outfit = evaluation.infit_outfit(data, theta, level = "respondent")
    assert torch.allclose(infit, torch.tensor([0.8181817, 0.4084507]), equal_nan=True), "Infit is incorrect"
    assert torch.allclose(outfit, torch.tensor([0.8181817, 0.3574660]), equal_nan=True), "Outfit is incorrect"

def test_q3(evaluation: Evaluator):
    data = torch.cat(
        [
            torch.randint(0, item_cat, (5, 1))
            for item_cat in evaluation.model.item_categories
        ],
        dim=1,
    ).float()
    data[1:2, 1] = torch.nan

    def residuals_mock(data, theta, average_over, **kwargs):
        residuals = torch.zeros_like(data)
        residuals[:, 0] = 0.5
        residuals[0, 0] = 0
        residuals[:, 1] = torch.arange(data.size()[0]) / data.size()[0]
        residuals[torch.isnan(data)] = torch.nan
        return residuals
    
    evaluation.residuals = MagicMock(side_effect=residuals_mock)

    q3_matrix, _ = evaluation.q3(data=data)

    assert q3_matrix.shape == (len(evaluation.model.item_categories), len(evaluation.model.item_categories)), \
        "q3_matrix does not have the expected shape"
    assert np.allclose(q3_matrix.iloc[0, 1], q3_matrix.iloc[1, 0], atol=1e-3), \
        "q3_matrix is not symmetric"
    assert np.isclose(q3_matrix.iloc[0, 1], 0.878, atol=1e-3), \
        "q3_matrix[0, 1] is not close to the expected value"

def test__observed_item_score_proportions(evaluation: Evaluator):
    data = torch.tensor(
        [
            [1., 2.],
            [1., 0.],
            [torch.nan, 2.],
            [0., 2.],
        ],
    )

    observed_proportions = evaluation._observed_item_score_proportions(data)
    assert torch.allclose(
        observed_proportions,
        torch.tensor([
            [0.3333333, 0.6666667, 0.00],
            [0.2500000, 0.0000000, 0.75]
        ])
    )

@pytest.mark.parametrize("cv_n_components", [[1], [1, 2, 3]])
def test__cv_gaussian_mixture_model(evaluation: Evaluator, cv_n_components):
    data = torch.randn(100, evaluation.model.latent_variables)

    with patch.object(GaussianMixtureModel, "fit") as mock_fit, patch.object(GaussianMixtureModel, "__call__") as mock_call:
        mock_fit.return_value = None
        mock_call.return_value = torch.tensor(1.0)

        gmm = evaluation._cv_gaussian_mixture_model(data, cv_n_components)

    assert isinstance(gmm, GaussianMixtureModel)
    assert gmm.n_components == cv_n_components[0]
    assert gmm.n_features == data.shape[1]
    mock_fit.assert_called()
    if len(cv_n_components) > 1:
        assert mock_call.call_count == len(cv_n_components) * 5  # 5-fold cross-validation