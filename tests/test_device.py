import pytest
import torch
from irtorch.models import GeneralizedPartialCredit
from irtorch.estimation_algorithms import MML, AE, VAE, JML

# Models are CPU-resident: algorithms move them to `device` for computation and back afterwards.
FIT_KWARGS = {
    MML: {"max_epochs": 3},
    AE: {"max_epochs": 3, "batch_size": 60},
    VAE: {"max_epochs": 3, "batch_size": 60},
    JML: {"max_epochs": 3, "batch_size": 60},
}


@pytest.mark.parametrize("algorithm", FIT_KWARGS.keys(), ids=lambda alg: alg.__name__)
def test_fit_and_score_on_device(algorithm, device, test_data, item_categories):
    torch.manual_seed(0)
    model = GeneralizedPartialCredit(item_categories=item_categories)
    model.fit(train_data=test_data, algorithm=algorithm(), device=device, **FIT_KWARGS[algorithm])

    assert all(param.device.type == "cpu" for param in model.parameters())
    assert all(torch.isfinite(param).all() for param in model.parameters())

    for theta_estimation in ["ML", "MAP"]:
        theta = model.latent_scores(test_data[:10], theta_estimation=theta_estimation, device=device)
        assert theta.shape == (10, 1)
        assert theta.device.type == "cpu"
        assert torch.isfinite(theta).all()
    assert all(param.device.type == "cpu" for param in model.parameters())
