import pytest
from unittest.mock import patch
from utils import initialize_fit
import torch
from irtorch.estimation_algorithms import AE
from irtorch.models import MonotoneNN


# The @pytest.fixture decorator is used to create fixture methods.
# This method is called once per test method that uses it, and its return value is passed to the test method as an argument.
# pytest.fixture with params creates two fixtures, one for the CPU and one for the GPU.
# The ids parameter is used to give the tests meaningful names
class TestAEIRT:
    @pytest.fixture()
    def algorithm(self, device, latent_variables, item_categories, data_loaders):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("GPU is not available.")

        model = MonotoneNN(
            latent_variables = latent_variables,
            item_categories = item_categories,
            hidden_dim = [3]
        )
        algorithm = AE(
            model=model,
            hidden_layers_encoder=[20],  # 1 hidden layer with 20 neurons
            nonlinear_encoder=torch.nn.ELU()
        )

        algorithm.imputation_method = "zero"
        # Set DataLoaders from the fixture
        algorithm.data_loader, algorithm.validation_data_loader = data_loaders

        initialize_fit(algorithm)
        return algorithm

    def test_forward(self, algorithm: AE, test_data):
        output = algorithm.forward(test_data)
        assert output.shape == (120, algorithm.model.max_item_responses * algorithm.model.items)

    def test_latent_scores(self, algorithm: AE, test_data):
        output = algorithm.z_scores(test_data.to(next(algorithm.parameters()).device))
        assert output.shape == (120, algorithm.model.latent_variables)

    def test__train_step(self, algorithm: AE):
        previous_loss = float("inf")
        for epoch in range(2):
            loss = algorithm._train_step(epoch)
            assert loss <= previous_loss  # Loss should decrease
            previous_loss = loss

    def test__validation_step(self, algorithm: AE):
        log_likelihood = algorithm._validation_step()
        assert isinstance(log_likelihood, float)
        assert log_likelihood > 0

    def test__impute_missing_zero(self, algorithm: AE):
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

    # The following is a test for the fit function of the AEIRTNeuralNet class
    def test_fit(self, algorithm: AE, test_data):
        # Mock the inner functions that would be called during training
        with patch.object(
            algorithm, "_train_step", return_value=torch.tensor(0.5)
        ) as mocked_train_step, patch.object(
            algorithm, "_validation_step", return_value=torch.tensor(0.5)
        ) as mocked_validation_step:
            # Call fit function
            algorithm.fit(
                train_data=test_data[0:100],
                validation_data=test_data[100:120],
                max_epochs=5
            )

            # Check if inner functions are called
            assert mocked_train_step.call_count == 5
            assert mocked_validation_step.call_count == 5
