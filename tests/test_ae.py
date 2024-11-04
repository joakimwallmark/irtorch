import pytest
from unittest.mock import patch
import torch
from irtorch.estimation_algorithms import AE
from irtorch.models import MonotoneNN, BaseIRTModel, GeneralizedPartialCredit
from irtorch.estimation_algorithms.encoders import StandardEncoder


# The @pytest.fixture decorator is used to create fixture methods.
# This method is called once per test method that uses it, and its return value is passed to the test method as an argument.
# pytest.fixture with params creates two fixtures, one for the CPU and one for the GPU.
# The ids parameter is used to give the tests meaningful names
class TestAE:
    @pytest.fixture()
    def irt_model(self, latent_variables, item_categories):
        model = GeneralizedPartialCredit(
            latent_variables = latent_variables,
            item_categories = item_categories
        )
        return model

    @pytest.fixture()
    def algorithm(self, irt_model: MonotoneNN, data_loaders, latent_variables, item_categories):
        algorithm = AE()
        algorithm.imputation_method = "zero"
        algorithm.one_hot_encoded = False
        # Set DataLoaders from the fixture
        algorithm.data_loader = data_loaders

        algorithm.encoder = StandardEncoder(
            input_dim=len(item_categories),
            latent_variables=latent_variables,
            hidden_dim=[2 * sum(irt_model.item_categories)]
        )

        return algorithm

    def test_theta_scores(self, algorithm: AE, irt_model: BaseIRTModel, test_data):
        output = algorithm.theta_scores(test_data)
        assert output.shape == (120, irt_model.latent_variables)

    def test__train_step(self, algorithm: AE, irt_model: BaseIRTModel):
        algorithm.optimizer = torch.optim.Adam(
            list(algorithm.encoder.parameters()) + list(irt_model.parameters()), lr=0.1, amsgrad=True
        )
        first_loss = float("inf")
        for epoch in range(5):
            loss = algorithm._train_step(irt_model, epoch)
            if first_loss == float("inf"):
                first_loss = loss
        
        assert loss <= first_loss, "Loss should decrease"

    def test_fit(self, algorithm: AE, irt_model: BaseIRTModel, test_data):
        # Mock the inner functions that would be called during training
        with patch.object(
            algorithm, "_train_step", return_value=torch.tensor(0.5)
        ) as mocked_train_step:
            algorithm.fit(
                model=irt_model,
                train_data=test_data[0:100],
                max_epochs=5
            )

            # Check if inner functions are called
            assert mocked_train_step.call_count == 5
