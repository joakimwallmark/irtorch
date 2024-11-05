import os
import random
import pytest
import torch
import numpy as np
import irtorch
from irtorch.models import MonotoneNN, GradedResponse, GeneralizedPartialCredit, BaseIRTModel
from irtorch.estimation_algorithms import AE, VAE, MML
from irtorch.irt_dataset import PytorchIRTDataset

@pytest.fixture(
    scope="module",
    params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
    ids=["cpu", "gpu"] if torch.cuda.is_available() else ["cpu"],
)
def device(request):
    return request.param

@pytest.fixture(scope="module")
def item_categories():
    return [2, 3, 3, 4, 4]

@pytest.fixture(scope="module")
def item_categories_small():
    return [2, 3]

@pytest.fixture(scope="module")
def test_data():
    test_data = torch.load("tests/datasets/test_data.pt", weights_only=False)
    return test_data[0:120, 8:13]

@pytest.fixture(scope="module", params=[1, 2, 3], ids=["dim1", "dim2", "dim3"])
def latent_variables(request):
    return request.param

@pytest.fixture(scope="module")
def theta_scores(latent_variables):
    repeated_tensor = torch.randn(4, latent_variables)

    return repeated_tensor

@pytest.fixture(scope="module")
def data_loaders(test_data):
    # reproducability as per https://pytorch.org/docs/stable/notes/randomness.html
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    data_loader = torch.utils.data.DataLoader(
        PytorchIRTDataset(data=test_data[0:100]),
        batch_size=32,
        shuffle=True,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return data_loader

@pytest.fixture(scope="module")
def data_loaders_small(test_data):
    # reproducability as per https://pytorch.org/docs/stable/notes/randomness.html
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    data_loader = torch.utils.data.DataLoader(
        PytorchIRTDataset(data=test_data[0:2]),
        batch_size=32,
        shuffle=True,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return data_loader
