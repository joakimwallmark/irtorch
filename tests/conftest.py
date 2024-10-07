import os
import random
import pytest
import torch
import numpy as np
import irtorch
from irtorch.models import MonotoneNN, GradedResponse, GeneralizedPartialCredit, BaseIRTModel
from irtorch.estimation_algorithms import AE, VAE, MML
from irtorch.irt_dataset import PytorchIRTDataset

# These fixture runs once per module, regardless of how many tests invoke it...
@pytest.fixture(scope="module")
def mml_1d_gpc_natmat_model() -> BaseIRTModel:
    data = irtorch.load_dataset.swedish_national_mathematics_1()
    torch.manual_seed(125)
    irt_model = GeneralizedPartialCredit(
        data=data,
        latent_variables=1,
    )
    file_path = f"tests/fitted_models/mml_1d_gpc_natmat_model.pt"
    if os.path.isfile(file_path):
        irt_model.load_model(file_path)
    else:
        irt_model.fit(
            train_data=data,
            algorithm=MML(),
        )
        irt_model.save_model(file_path)

    return irt_model

@pytest.fixture(scope="module")
def mml_1d_gpc_natmat_thetas(mml_1d_gpc_natmat_model: BaseIRTModel):
    data = irtorch.load_dataset.swedish_national_mathematics_1()
    file_path = f"tests/fitted_models/mml_1d_gpc_natmat_thetas.pt"
    if os.path.isfile(file_path):
        return torch.load(file_path)
    else:
        thetas = mml_1d_gpc_natmat_model.latent_scores(data=data)
        torch.save(thetas, file_path)
        return thetas

@pytest.fixture(scope="module")
def ae_1d_mmc_swesat_model():
    data_sat, correct_responses = irtorch.load_dataset.swedish_sat_verbal()
    torch.manual_seed(125)
    irt_model = MonotoneNN(
        data=data_sat,
        latent_variables=1,
        mc_correct=correct_responses,
    )
    file_path = f"tests/fitted_models/ae_1d_mmc_swesat_model.pt"
    if os.path.isfile(file_path):
        irt_model.load_model(file_path)
    else:
        irt_model.fit(
            train_data=data_sat,
            algorithm=AE(),
        )
        irt_model.save_model(file_path)

    return irt_model

# @pytest.fixture(scope="module")
# def aaaa(ae_1d_mmc_swesat_model: BaseIRTModel) -> torch.Tensor:
@pytest.fixture(scope="module")
def ae_1d_mmc_swesat_thetas(ae_1d_mmc_swesat_model: BaseIRTModel) -> torch.Tensor:
    data_sat, _ = irtorch.load_dataset.swedish_sat_verbal()
    file_path = f"tests/fitted_models/ae_1d_mmc_swesat_thetas.pt"
    if os.path.isfile(file_path):
        return torch.load(file_path, weights_only=False)
    else:
        thetas = ae_1d_mmc_swesat_model.latent_scores(data=data_sat)
        torch.save(thetas, file_path)
        return thetas

@pytest.fixture(scope="module")
def vae_5d_graded_big_five_model():
    data = irtorch.load_dataset.big_five()[0]
    torch.manual_seed(125)
    irt_model = GradedResponse(
        data=data,
        latent_variables=5,
    )
    file_path = f"tests/fitted_models/vae_5d_graded_big_five_model.pt"
    if os.path.isfile(file_path):
        irt_model.load_model(file_path)
    else:
        irt_model.fit(
            train_data=data,
            algorithm=VAE(),
        )
        irt_model.save_model(file_path)

    return irt_model

@pytest.fixture(scope="module")
def vae_5d_graded_big_five_thetas(vae_5d_graded_big_five_model: BaseIRTModel):
    data = irtorch.load_dataset.big_five()[0]
    file_path = f"tests/fitted_models/vae_5d_graded_big_five_thetas.pt"
    if os.path.isfile(file_path):
        return torch.load(file_path, weights_only=False)
    else:
        thetas = vae_5d_graded_big_five_model.latent_scores(data=data, theta_estimation = "NN")
        torch.save(thetas, file_path)
        return thetas

def ae_1d_mmc_swesat_model_thetas(ae_1d_mmc_swesat_model: BaseIRTModel):
    data = irtorch.load_dataset.swedish_national_mathematics_1()
    file_path = f"tests/fitted_models/ae_1d_mmc_swesat_model_thetas.pt"
    if os.path.isfile(file_path):
        return torch.load(file_path, weights_only=False)
    else:
        thetas = mml_1d_gpc_natmat_model.latent_scores(data=data)
        torch.save(thetas, file_path)
        return thetas

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
