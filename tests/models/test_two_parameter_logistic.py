import pytest
import torch
import pandas as pd
from irtorch.models import TwoParameterLogistic

def test_forward():
    model = TwoParameterLogistic(
        latent_variables = 2,
        items = 3,
        item_theta_relationships=torch.tensor([[True, True], [True, True], [False, True]])
    )
    original_weights = model.weight_param.clone()
    original_bias = model.bias_param.clone()

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}], lr=0.02, amsgrad=True
    )
    theta = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.5], [0.5, 0.6]])
    data = torch.tensor([[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 1]]).float()
    for _ in range(2): # update two times with the same data
        optimizer.zero_grad()
        output = model.forward(theta)
        assert output.shape == (4, 6), "Incorrect output shape"

        loss = -model.log_likelihood(data=data, output=output)
        # Compute gradients
        loss.backward()
        optimizer.step()

    assert torch.all(original_weights != model.weight_param), "Free weights should have changed"
    assert torch.all(original_bias != model.bias_param), "Free bias should have changed"

def test_item_parameters(latent_variables):
    model = TwoParameterLogistic(
        latent_variables = latent_variables,
        items = 3,
        item_theta_relationships=torch.tensor([[True], [True], [True]]).repeat(1, latent_variables)
    )
    parameters = model.item_parameters(irt_format=False)

    assert isinstance(parameters, pd.DataFrame), "Output should be a DataFrame"
    assert parameters.shape == (model.items, model.latent_variables + 1), "Incorrect shape of parameters DataFrame"

    if latent_variables == 1:
        parameters_irt = model.item_parameters(irt_format=True)
        assert isinstance(parameters_irt, pd.DataFrame), "Output should be a DataFrame"
        assert parameters_irt.shape == (model.items, model.latent_variables + 1), "Incorrect shape of parameters DataFrame"
    else:
        with pytest.raises(ValueError):
            model.item_parameters(irt_format=True)

def test_item_theta_relationship_directions():
    model = TwoParameterLogistic(
        latent_variables = 2,
        items = 3,
        item_theta_relationships=torch.tensor([[True, True], [True, True], [False, True]])
    )
    model.weight_param.data = torch.tensor([0.5, 0.6, -0.8, -1, 2])

    directions = model.item_theta_relationship_directions()
    assert directions.shape == (3, 2), "Incorrect directions shape"
    assert torch.all(directions[0, :] == 1), "Incorrect directions"
    assert torch.all(directions[1, :] == -1), "Incorrect directions"
    assert directions[2, 0] == 0, "Incorrect directions"
    assert directions[2, 1] == 1, "Incorrect directions"
