import os
import random
import pytest
import torch
import numpy as np
from irtorch.irt import IRT
from irtorch._internal_utils import PytorchIRTDataset
from irtorch.utils import get_item_categories

@pytest.fixture(
    scope="module",
    params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
    ids=["cpu", "gpu"] if torch.cuda.is_available() else ["cpu"],
)
def device(request):
    return request.param

# This fixture runs once per module, regardless of how many tests invoke it...
@pytest.fixture(scope="module", params=["VAE", "AE"])
def fitting_algorithm(request):
    return request.param

@pytest.fixture(scope="module", params=["binary", "polytomous", "mc"])
def data_type(request):
    return request.param

@pytest.fixture(scope="module")
def data(data_type):
    if data_type == "binary":
        return torch.load("tests/datasets/test_data_bin.pt")
    if data_type == "polytomous":
        return torch.load("tests/datasets/test_data_poly.pt")
    if data_type == "mc":
        return torch.load("tests/datasets/test_data_mc.pt")

@pytest.fixture(scope="module")
def model(device, latent_variables, data, data_type, fitting_algorithm):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU is not available.")
    one_hot_encoded=False
    correct_cat=None
    if (data_type == "mc"):
        one_hot_encoded=True
        with open("tests/datasets/mc_correct.txt", "r") as file:
            correct_cat = file.read().replace("\n", "")
        correct_cat = [int(num) for num in correct_cat]
        n_cats = [4, 4, 5, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 5, 4, 5, 4, 5]
    else:
        n_cats = get_item_categories(data)

    torch.manual_seed(125)
    model = IRT(
        estimation_algorithm=fitting_algorithm,
        latent_variables=latent_variables,
        item_categories=n_cats,
        one_hot_encoded=one_hot_encoded,
        mc_correct=correct_cat
    )
    # check if file exists
    file_path = f"tests/models/{fitting_algorithm}_latent_variables{latent_variables}_{data_type}_{device}.pt"
    if os.path.isfile(file_path):
        model.load_model(f"tests/models/{fitting_algorithm}_latent_variables{latent_variables}_{data_type}_{device}.pt")
    else:
        model.fit(
            train_data=data,
            device=device
        )
        model.save_model(f"tests/models/{fitting_algorithm}_latent_variables{latent_variables}_{data_type}_{device}.pt")

    return model

@pytest.fixture(scope="module")
def item_categories():
    return [2, 3, 3, 4, 4]

@pytest.fixture(scope="module")
def item_categories_small():
    return [2, 3]

@pytest.fixture(scope="module")
def item_categories_binary():
    return [2] * 5

@pytest.fixture(scope="module")
def test_data():
    test_data = torch.load("tests/datasets/test_data.pt")
    return test_data[0:120, 8:13]

@pytest.fixture(scope="module", params=[1, 2, 3], ids=["dim1", "dim2", "dim3"])
def latent_variables(request):
    return request.param

@pytest.fixture(scope="module")
def z_scores(latent_variables):
    repeated_tensor = torch.randn(4, latent_variables)
    # repeated_tensor = torch.tensor([[1.0], [2.0], [1.5]]).repeat(1, latent_variables)
    # for i in range(latent_variables):
    #     repeated_tensor[:, i] = repeated_tensor[:, i] ** (1+ (i * 0.5))  # Offset each column by a predictable amount

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

    validation_data_loader = torch.utils.data.DataLoader(
        PytorchIRTDataset(data=test_data[100:120]),
        batch_size=32,
        shuffle=True,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return data_loader, validation_data_loader


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

    validation_data_loader = torch.utils.data.DataLoader(
        PytorchIRTDataset(data=test_data[2:4]),
        batch_size=32,
        shuffle=True,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return data_loader, validation_data_loader
