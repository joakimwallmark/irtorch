import pytest
import torch
from irtorch.models import BaseIRTModel
from irtorch.estimation_algorithms import JML

class DummyIRTModel(BaseIRTModel):
    def __init__(self, latent_variables=1):
        super().__init__(latent_variables, [2] * 10)

    def forward(self, theta):
        return theta

    def log_likelihood(self, responses, model_out, missing_mask=None, loss_reduction="sum"):
        return torch.sum(model_out)

@pytest.fixture
def setup_jml():
    model = DummyIRTModel()
    train_data = torch.randint(0, 2, (100, 10)).float()
    jml = JML()
    return jml, model, train_data

def test_fit(setup_jml):
    jml, model, train_data = setup_jml
    jml.fit(model, train_data, max_epochs=10, batch_size=10)
    assert len(jml.training_history["train_loss"]) > 0

def test_fit_with_start_thetas(setup_jml):
    jml, model, train_data = setup_jml
    start_thetas = torch.randn(100, 1)
    jml.fit(model, train_data, start_thetas=start_thetas, max_epochs=10, batch_size=10)
    assert len(jml.training_history["train_loss"]) > 0

def test_fit_invalid_start_thetas(setup_jml):
    jml, model, train_data = setup_jml
    start_thetas = torch.randn(50, 1)
    with pytest.raises(ValueError):
        jml.fit(model, train_data, start_thetas=start_thetas, max_epochs=10, batch_size=10)

def test_fit_invalid_latent_variables(setup_jml):
    jml, model, train_data = setup_jml
    start_thetas = torch.randn(100, 2)
    with pytest.raises(ValueError):
        jml.fit(model, train_data, start_thetas=start_thetas, max_epochs=10, batch_size=10)
