import pytest
import torch
import pandas as pd
from irtorch.models import OneParameterLogistic

def test_forward():
    model = OneParameterLogistic(
        items = 3,
    )
    original_bias = model.bias_param.clone()

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}], lr=0.02, amsgrad=True
    )
    theta = torch.tensor([[0.1], [0.3], [0.5], [0.6]])
    data = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1]]).float()
    for _ in range(2): # update two times with the same data
        optimizer.zero_grad()
        output = model.forward(theta)
        assert output.shape == (4, 6), "Incorrect output shape"

        loss = -model.log_likelihood(data=data, output=output)
        # Compute gradients
        loss.backward()
        optimizer.step()

    assert torch.all(original_bias != model.bias_param), "Free bias should have changed"

def test_item_parameters():
    model = OneParameterLogistic(
        items = 3,
    )
    parameters = model.item_parameters()

    assert isinstance(parameters, pd.DataFrame), "Output should be a DataFrame"
    assert parameters.shape == (model.items, 1), "Incorrect shape of parameters DataFrame"

def test_item_theta_relationship_directions():
    model = OneParameterLogistic(
        items = 3,
    )
    directions = model.item_theta_relationship_directions()
    assert directions.shape == (3, 1), "Incorrect directions shape"
    assert torch.all(directions == 1), "Incorrect directions"
