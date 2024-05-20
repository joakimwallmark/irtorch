import pytest
import torch
from utils import initialize_fit
from irtorch.estimation_algorithms.vae import VAE
from irtorch.models import MonotoneNN


# The @pytest.fixture decorator is used to create fixture methods.
# This method is called once per test method that uses it, and its return value is passed to the test method as an argument.
# pytest.fixture with params creates two fixtures, one for the CPU and one for the GPU.
# The ids parameter is used to give the tests meaningful names
class TestVAIRT:
    @pytest.fixture()
    def algorithm(self, device, latent_variables, item_categories, data_loaders):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("GPU is not available.")

        model = MonotoneNN(
            latent_variables = latent_variables,
            item_categories = item_categories,
            hidden_dim = [3]
        )
        algorithm = VAE(
            model=model,
            hidden_layers_encoder=[20],  # 1 hidden layer with 20 neurons
            nonlinear_encoder=torch.nn.ELU()
        )

        # Set DataLoaders from the fixture
        algorithm.data_loader, algorithm.validation_data_loader = data_loaders

        initialize_fit(algorithm)
        return algorithm

    @pytest.fixture()
    def algorithm_small_data(
        self, latent_variables, item_categories_small, data_loaders_small
    ):
        # same weights and biases every time
        torch.manual_seed(0)
        model = MonotoneNN(
            latent_variables = latent_variables,
            item_categories = item_categories_small,
            hidden_dim = [3]
        )
        algorithm = VAE(
            model=model,
            hidden_layers_encoder=[20],  # 1 hidden layer with 20 neurons
            nonlinear_encoder=torch.nn.ELU()
        )

        # Set DataLoaders from the fixture
        algorithm.data_loader, algorithm.validation_data_loader = data_loaders_small

        initialize_fit(algorithm)
        return algorithm

    def test_forward(self, algorithm: VAE, test_data):
        algorithm.iw_samples = 1
        output = algorithm(test_data)
        assert len(output) == 4
        assert output[0].shape == (
            algorithm.iw_samples * 120,
            max(algorithm.model.modeled_item_responses)*len(algorithm.model.modeled_item_responses),
        )
        assert output[1].shape == (
            algorithm.iw_samples * 120,
            algorithm.model.latent_variables,
        )
        assert output[2].shape == (
            algorithm.iw_samples * 120,
            algorithm.model.latent_variables,
        )
        assert output[3].shape == (
            algorithm.iw_samples * 120,
            algorithm.model.latent_variables,
        )
        algorithm.iw_samples = 10
        output = algorithm(test_data)
        assert len(output) == 4
        assert output[0].shape == (
            algorithm.iw_samples * 120,
            max(algorithm.model.modeled_item_responses)*len(algorithm.model.modeled_item_responses),
        )
        assert output[1].shape == (
            algorithm.iw_samples * 120,
            algorithm.model.latent_variables,
        )
        assert output[2].shape == (
            algorithm.iw_samples * 120,
            algorithm.model.latent_variables,
        )
        assert output[3].shape == (
            algorithm.iw_samples * 120,
            algorithm.model.latent_variables,
        )

    def test_latent_scores(self, algorithm: VAE, test_data):
        output = algorithm.z_scores(test_data)
        assert output.shape == (120, algorithm.model.latent_variables)

    def test__impute_missing_with_prior(self, algorithm: VAE):
        a, b = 5, 5
        data = torch.full((a, b), 5).float()
        missing_mask = torch.tensor(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
            ]
        )

        algorithm.imputation_method = "prior"
        imputed_data = algorithm._impute_missing_with_prior(data, missing_mask)
        assert torch.equal(
            torch.full((20,), 5).float(),
            imputed_data[(1 - missing_mask).bool()],
        )
        for replaced in torch.not_equal(
            torch.full((5,), 5).float(),
            imputed_data[missing_mask.bool()],
        ):
            assert replaced

    def test__mean_scores(self, algorithm: VAE):
        logits = torch.tensor(
            [
                [
                    0.1586,
                    -0.3347,
                    -0.1472,
                    0.0962,
                    -0.2445,
                    -0.1546,
                    -0.0588,
                    0.0421,
                    -0.2624,
                    -0.1405,
                    0.0943,
                    0.0735,
                    -0.1177,
                    -0.0686,
                    0.0465,
                    -0.0359,
                ]
            ]
        )

        means = algorithm._mean_scores(logits)
        assert torch.allclose(
            means,
            torch.tensor([0.3791, 0.9709, 1.0655, 1.6510, 1.5445]),
            atol=1e-4,
        )

    @pytest.mark.parametrize("iw_samples", [1, 3])
    def test__loss_function(
        self,
        algorithm_small_data: VAE,
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
        z_samples = means + torch.randn_like(std) * std
        loss = algorithm_small_data._loss_function(
            test_data[0:2, 0:2], logits, z_samples, means, logvars
        )
        assert loss > 0
