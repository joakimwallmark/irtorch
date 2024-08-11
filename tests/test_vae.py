import pytest
import torch
from irtorch.estimation_algorithms.vae import VAE, VariationalEncoder
from irtorch.models import MonotoneNN, BaseIRTModel


# The @pytest.fixture decorator is used to create fixture methods.
# This method is called once per test method that uses it, and its return value is passed to the test method as an argument.
# pytest.fixture with params creates two fixtures, one for the CPU and one for the GPU.
# The ids parameter is used to give the tests meaningful names
class TestVAE:
    @pytest.fixture()
    def irt_model(self, latent_variables, item_categories):
        model = MonotoneNN(
            latent_variables = latent_variables,
            item_categories = item_categories,
            hidden_dim = [3]
        )
        return model

    @pytest.fixture()
    def algorithm(self, irt_model: BaseIRTModel, data_loaders, latent_variables, item_categories):
        algorithm = VAE()
        algorithm.imputation_method = "zero"
        algorithm.one_hot_encoded = False
        # Set DataLoaders from the fixture
        algorithm.data_loader, algorithm.validation_data_loader = data_loaders

        algorithm.encoder = VariationalEncoder(
            input_dim=len(item_categories),
            latent_variables=latent_variables,
            hidden_dim=[2 * sum(irt_model.modeled_item_responses)]
        )
        return algorithm

    @pytest.fixture()
    def irt_model_small(self, latent_variables, item_categories_small):
        model = MonotoneNN(
            latent_variables = latent_variables,
            item_categories = item_categories_small,
            hidden_dim = [3]
        )
        return model

    @pytest.fixture()
    def algorithm_small_data(self, irt_model: BaseIRTModel, data_loaders_small, latent_variables, item_categories_small):
        algorithm = VAE()
        algorithm.imputation_method = "zero"
        algorithm.one_hot_encoded = False
        # Set DataLoaders from the fixture
        algorithm.data_loader, algorithm.validation_data_loader = data_loaders_small

        algorithm.encoder = VariationalEncoder(
            input_dim=len(item_categories_small),
            latent_variables=latent_variables,
            hidden_dim=[2 * sum(irt_model.modeled_item_responses)]
        )
        return algorithm

    def test_theta_scores(self, algorithm: VAE, irt_model: BaseIRTModel, test_data):
        output = algorithm.theta_scores(test_data)
        assert output.shape == (120, irt_model.latent_variables)

    @pytest.mark.parametrize("iw_samples", [1, 3])
    def test__loss_function(
        self,
        algorithm_small_data: VAE,
        irt_model_small: BaseIRTModel,
        test_data,
        iw_samples,
        latent_variables,
    ):
        algorithm_small_data.iw_samples = iw_samples
        logits = torch.tensor(
            [
                [0.8221, -0.3415, -torch.inf, 0.3751, -0.3118, 0.0909],
                [-0.6602, 0.0387, -torch.inf, -0.4591, 0.3182, 0.0278],
            ],
        ).repeat(iw_samples, 1)
        means = torch.tensor([[0], [1]]).repeat(
            iw_samples, latent_variables
        )
        logvars = torch.tensor([[0], [0.5]]).repeat(
            iw_samples, latent_variables
        )
        std = torch.exp(0.5 * logvars)
        theta_samples = means + torch.randn_like(std) * std
        loss = algorithm_small_data._loss_function(
            irt_model_small, test_data[0:2, 0:2], torch.zeros_like(test_data[0:2, 0:2]), logits, theta_samples, means, logvars
        )
        assert loss > 0
