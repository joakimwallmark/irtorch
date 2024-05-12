import pytest
from unittest.mock import patch
from utils import initialize_fit
import torch
from torch.distributions import MultivariateNormal
from irtorch.estimation_algorithms import MMLIRT
from irtorch.models import MonotoneNN


# The @pytest.fixture decorator is used to create fixture methods.
# This method is called once per test method that uses it, and its return value is passed to the test method as an argument.
# pytest.fixture with params creates two fixtures, one for the CPU and one for the GPU.
# The ids parameter is used to give the tests meaningful names
class TestMMLIRT:
    @pytest.fixture()
    def algorithm(self, device, latent_variables, item_categories):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("GPU is not available.")

        model = MonotoneNN(
            latent_variables = latent_variables,
            item_categories = item_categories,
            hidden_dim = [3]
        )
        algorithm = MMLIRT(model=model)

        algorithm.imputation_method = "zero"

        initialize_fit(algorithm)
        return algorithm

    def test_forward(self, algorithm: MMLIRT, test_data):
        output = algorithm.forward(test_data)
        assert output.shape == (120, algorithm.model.max_item_responses * algorithm.model.items)

    def test__train_step(self, algorithm: MMLIRT, test_data: torch.Tensor, device):
        previous_loss = float("inf")
        latent_grid = torch.linspace(-3, 3, 5).view(-1, 1)
        latent_grid = latent_grid.expand(-1, algorithm.model.latent_variables).contiguous().to(device)
        normal_dist = MultivariateNormal(
            loc=torch.zeros(algorithm.model.latent_variables).to(device),
            covariance_matrix=torch.eye(algorithm.model.latent_variables).to(device)
        )
        if latent_grid.size(1) > 1:
            columns = [latent_grid[:, i] for i in range(latent_grid.size(1))]
            latent_combos = torch.cartesian_prod(*columns)
        else:
            latent_combos = latent_grid

        log_weights = normal_dist.log_prob(latent_combos)
        latent_combos_rep = latent_combos.repeat_interleave(test_data.size(0), dim=0)
        train_data_rep = test_data.repeat(latent_combos.size(0), 1)
        log_weights_rep = log_weights.repeat_interleave(test_data.size(0), dim=0)


        algorithm.to(device)
        for _ in range(2):
            loss = algorithm._train_step(
                train_data_rep.to(device),
                latent_combos_rep,
                log_weights_rep,
                latent_combos.size(0)
            )
            assert loss <= previous_loss  # Loss should decrease
            previous_loss = loss

    def test__impute_missing_zero(self, algorithm: MMLIRT):
        a, b = 5, 5
        data = torch.full((a, b), 5)
        no_missing_mask = torch.full((a, b), 0)
        missing_mask = torch.tensor(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
            ]
        )

        algorithm.imputation_method = "zero"
        imputed_data = algorithm._impute_missing(data, missing_mask)
        assert torch.equal(
            imputed_data,
            torch.tensor(
                [
                    [5, 5, 0, 5, 5],
                    [5, 0, 5, 5, 5],
                    [5, 5, 5, 5, 0],
                    [5, 5, 5, 5, 5],
                    [0, 0, 5, 5, 5],
                ]
            ),
        )
        imputed_data = algorithm._impute_missing(data, no_missing_mask)
        assert torch.equal(imputed_data, data)

    # The following is a test for the fit function of the MMLIRTNeuralNet class
    def test_fit(self, algorithm: MMLIRT, test_data):
        # Mock the inner functions that would be called during training
        with patch.object(
            algorithm, "_train_step", return_value=torch.tensor(0.5)
        ) as mocked_train_step:
            # Call fit function
            algorithm.fit(
                train_data=test_data[0:100],
                max_epochs=5
            )

            # Check if inner functions are called
            assert mocked_train_step.call_count == 5
