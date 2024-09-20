import pytest
import torch
from irtorch.models import ModelMix, TwoParameterLogistic, GeneralizedPartialCredit

@pytest.fixture
def model_mix() -> ModelMix:
    two_pl = TwoParameterLogistic(latent_variables=2, items=2)
    gpc = GeneralizedPartialCredit(latent_variables=2, item_categories=[3])
    return ModelMix([two_pl, gpc])

def test_forward_loss(model_mix: ModelMix):
    optimizer = torch.optim.Adam(
        [{"params": model_mix.parameters()}], lr=0.02, amsgrad=True
    )
    original_2pl_weights = model_mix.models[0].weight_param.clone()
    original_2pl_bias = model_mix.models[0].bias_param.clone()
    original_gpc_weights = model_mix.models[1].weight_param.clone()
    original_gpc_bias = model_mix.models[1].bias_param.clone()

    theta = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.5], [0.8, 0.6]])
    data = torch.tensor([[0, 1, 0], [0, 1, 1], [1, 0, 2], [1, 1, 2]]).float()

    for _ in range(2): # update two times with the same data
        optimizer.zero_grad()
        output = model_mix.forward(theta)
        assert output.shape == (4, 7), "Incorrect output shape"

        loss = -model_mix.log_likelihood(data=data, output=output)
        # Compute gradients
        loss.backward()
        optimizer.step()

    assert torch.all(original_2pl_weights != model_mix.models[0].weight_param), "Free weights should have changed"
    assert torch.all(original_2pl_bias != model_mix.models[0].bias_param), "Free bias should have changed"
    assert torch.all(original_gpc_weights != model_mix.models[1].weight_param), "Free weights should have changed"
    assert torch.all(original_gpc_bias != model_mix.models[1].bias_param), "Free bias should have changed"

def test_probabilities_from_output(model_mix: ModelMix):
    output = torch.tensor([
        [0, 0, -0.4, 0.2, -0.2, 0.0, 0.5],
        [0, 0, -0.4, 0.2, -0.2, 0.0, 0.5],
        [0, 0, -0.4, 0.2, -0.2, 0.0, 0.5],
        [0, 0, -0.4, 0.2, -0.2, 0.0, 0.5]
    ])
    probs = model_mix.probabilities_from_output(output)
    assert probs.shape == (4, 3, 3), "Incorrect shape for probabilities"
    assert torch.all(probs[:, :2, 2] == 0.0), "Incorrect probabilities for 2PL"

def test_log_likelihood(model_mix: ModelMix):
    data = torch.tensor([[0, 1, 0], [0, 1, 1], [1, 0, 2], [1, 1, 2]]).float()
    output = torch.tensor([
        [0, 0, -0.4, 0.2, -0.2, 0.0, 0.5],
        [0, 0, -0.4, 0.2, -0.2, 0.0, 0.5],
        [0, 0, -0.4, 0.2, -0.2, 0.0, 0.5],
        [0, 0, -0.4, 0.2, -0.2, 0.0, 0.5]
    ])
    result = model_mix.log_likelihood(data, output, loss_reduction="none")
    assert result.shape == (12, ), "Incorrect shape for log_likelihood"
    assert torch.isclose(result.sum(), torch.tensor(-9.296221733)), "Incorrect log_likelihood sum"
    assert torch.equal(result[0], result[2]) and torch.equal(result[0], result[4]), "all first item likelihoods should be equal"
    assert torch.equal(result[1], result[3]), "same responses should be equal"
    assert torch.isclose(result[11], torch.tensor(-0.743420)), "incorrect item likelihood"

    data = torch.tensor([[0, 1, 0], [0, torch.nan, 1], [1, 0, torch.nan], [1, 1, 2]]).float()
    missing_mask = torch.isnan(data)
    output = torch.tensor([
        [0, 0, -0.4, 0.2, -0.2, 0.0, 0.5],
        [0, 0, -0.4, 0.2, -0.2, 0.0, 0.5],
        [0, 0, -0.4, 0.2, -0.2, 0.0, 0.5],
        [0, 0, -0.4, 0.2, -0.2, 0.0, 0.5]
    ])
    result = model_mix.log_likelihood(data, output, missing_mask, loss_reduction="none")
    assert result.shape == (12, ), "Incorrect shape for log_likelihood"
    assert torch.isclose(result.nansum(), torch.tensor(-8.115312576)), "Incorrect log_likelihood sum"
    assert torch.equal(result[0], result[2]) and torch.equal(result[0], result[4]), "all first item likelihoods should be equal"
    assert torch.isnan(result[3]), "missing response should be nan"
    assert torch.isnan(result[10]), "missing response should be nan"
    assert torch.isclose(result[11], torch.tensor(-0.743420)), "incorrect item likelihood"
