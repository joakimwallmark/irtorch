import pytest
from unittest.mock import patch
import torch
from torch.distributions import MultivariateNormal
from irtorch.irt_dataset import PytorchIRTDataset
from irtorch.estimation_algorithms import MML
from irtorch.models import MonotoneNN


# The @pytest.fixture decorator is used to create fixture methods.
# This method is called once per test method that uses it, and its return value is passed to the test method as an argument.
# pytest.fixture with params creates two fixtures, one for the CPU and one for the GPU.
# The ids parameter is used to give the tests meaningful names
class TestMML:
    @pytest.fixture()
    def irt_model(self, latent_variables, item_categories):
        model = MonotoneNN(
            latent_variables = latent_variables,
            item_categories = item_categories,
            hidden_dim = [3]
        )
        return model

    @pytest.fixture()
    def algorithm(self, data_loaders):
        algorithm = MML()
        algorithm.imputation = "zero"
        # Set DataLoaders from the fixture
        algorithm.data_loader = data_loaders
        return algorithm

    def test__train_step(self, algorithm: MML, irt_model: MonotoneNN, test_data: torch.Tensor):
        algorithm.optimizer = torch.optim.Adam(
            list(irt_model.parameters()), lr=0.01, amsgrad=True
        )
        previous_loss = float("inf")
        latent_grid = torch.linspace(-3, 3, 5).view(-1, 1)
        latent_grid = latent_grid.expand(-1, irt_model.latent_variables).contiguous()
        normal_dist = MultivariateNormal(
            loc=torch.zeros(irt_model.latent_variables),
            covariance_matrix=torch.eye(irt_model.latent_variables)
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
        irt_data_rep = PytorchIRTDataset(train_data_rep)

        algorithm
        for _ in range(2):
            loss = algorithm._train_step(
                irt_model,
                irt_data_rep,
                latent_combos_rep,
                log_weights_rep,
                latent_combos.size(0)
            )
            assert loss <= previous_loss  # Loss should decrease
            previous_loss = loss

    def test_fit(self, algorithm: MML, irt_model: MonotoneNN, test_data):
        # Mock the inner functions that would be called during training
        with patch.object(
            algorithm, "_train_step", return_value=torch.tensor(0.5)
        ) as mocked_train_step:
            # Call fit function
            algorithm.fit(
                model=irt_model,
                train_data=test_data[0:100],
                max_epochs=5
            )

            # Check if inner functions are called
            assert mocked_train_step.call_count == 5

    def test__quasi_mc(self, algorithm: MML):
        algorithm.covariance_matrix = torch.eye(1)
        points, log_weights = algorithm._quasi_mc(5, 1)
        assert points.size() == (5, 1)
        assert log_weights.size() == (5,)

        algorithm.covariance_matrix = torch.eye(2)
        points, log_weights = algorithm._quasi_mc(5, 2)
        assert points.size() == (25, 2)
        assert log_weights.size() == (25,)
