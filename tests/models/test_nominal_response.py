import pytest
import torch
import pandas as pd
from irtorch.models import NominalResponse

def test_forward():
    # With reference category
    model = NominalResponse(
        latent_variables = 2,
        item_categories=[2, 3, 3],
        item_theta_relationships=torch.tensor([[True, True], [True, True], [False, True]]),
        reference_category=True
    )
    original_weights = model.weight_param.clone()
    original_bias = model.bias_param.clone()

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}], lr=0.02, amsgrad=True
    )
    theta = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.5], [0.8, 0.6]])
    data = torch.tensor([[0, 1, 0], [0, 1, 1], [1, 2, 1], [1, 2, 2]]).float()
    for _ in range(2): # update two times with the same data
        optimizer.zero_grad()
        output = model.forward(theta)
        assert output.shape == (4, 9), "Incorrect output shape"

        loss = -model.log_likelihood(data=data, output=output)
        # Compute gradients
        loss.backward()
        optimizer.step()

    assert model.weight_param.shape == torch.Size([8]), "Incorrect weight shape"
    assert model.bias_param.shape == torch.Size([5]), "Incorrect weight shape"
    assert torch.all(original_weights != model.weight_param), "Free weights should have changed"
    assert torch.all(original_bias != model.bias_param), "Free bias should have changed"

    # Without reference category
    model = NominalResponse(
        latent_variables = 2,
        item_categories=[2, 3, 3],
        item_theta_relationships=torch.tensor([[True, True], [True, True], [False, True]]),
        reference_category=False
    )
    original_weights = model.weight_param.clone()
    original_bias = model.bias_param.clone()

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}], lr=0.02, amsgrad=True
    )
    theta = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.5], [0.8, 0.6]])
    data = torch.tensor([[0, 1, 0], [0, 1, 1], [1, 2, 1], [1, 2, 2]]).float()
    for _ in range(2): # update two times with the same data
        optimizer.zero_grad()
        output = model.forward(theta)
        assert output.shape == (4, 9), "Incorrect output shape"

        loss = -model.log_likelihood(data=data, output=output)
        # Compute gradients
        loss.backward()
        optimizer.step()

    assert model.weight_param.shape == torch.Size([13]), "Incorrect weight shape"
    assert model.bias_param.shape == torch.Size([8]), "Incorrect weight shape"
    assert torch.all(original_weights != model.weight_param), "Free weights should have changed"
    assert torch.all(original_bias != model.bias_param), "Free bias should have changed"

def test_item_parameters(latent_variables):
    model = NominalResponse(
        latent_variables = latent_variables,
        item_categories=[2, 3, 4],
        item_theta_relationships=torch.tensor([[True], [True], [True]]).repeat(1, latent_variables)
    )
    parameters = model.item_parameters(irt_format=False)

    assert isinstance(parameters, pd.DataFrame), f"Output should be a DataFrame"
    assert parameters.shape == (model.items, (model.latent_variables + 1) * model.max_item_responses), f"Incorrect shape of parameters DataFrame"

    parameters_irt = model.item_parameters(irt_format=True)

    assert isinstance(parameters_irt, pd.DataFrame), f"Output should be a DataFrame"
    assert parameters_irt.shape == (model.items, (model.latent_variables + 1) * model.max_item_responses), f"Incorrect shape of parameters DataFrame"

    model = NominalResponse(
        latent_variables = latent_variables,
        item_categories=[2, 3, 4],
        item_theta_relationships=torch.tensor([[True], [True], [True]]).repeat(1, latent_variables),
        reference_category=True
    )
    parameters = model.item_parameters(irt_format=False)

    assert isinstance(parameters, pd.DataFrame), f"Output should be a DataFrame"
    assert parameters.shape == (model.items, (model.latent_variables + 1) * model.max_item_responses), f"Incorrect shape of parameters DataFrame"

    parameters_irt = model.item_parameters(irt_format=True)

    assert isinstance(parameters_irt, pd.DataFrame), f"Output should be a DataFrame"
    assert parameters_irt.shape == (model.items, (model.latent_variables + 1) * model.max_item_responses), f"Incorrect shape of parameters DataFrame"
