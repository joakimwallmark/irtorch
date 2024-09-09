import pytest
import torch
import pandas as pd
from irtorch.models import GradedResponse

def test_forward():
    model = GradedResponse(
        latent_variables = 2,
        item_categories=[2, 3, 3],
        item_theta_relationships=torch.tensor([[True, True], [True, True], [False, True]])
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

    assert torch.all(original_weights != model.weight_param), "Free weights should have changed"
    assert torch.all(original_bias != model.bias_param), "Free bias should have changed"

def test_log_likelihood():
    model = GradedResponse(
        latent_variables = 2,
        item_categories=[2, 3, 3],
        item_theta_relationships=torch.tensor([[True, True], [True, True], [False, True]])
    )
    theta = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.5], [0.8, 0.6]])
    data = torch.tensor([[0., 1., 0.], [0., 1., 1.], [1, 2, torch.nan], [1, 2, 2]])
    output = model.forward(theta)
    ll = model.log_likelihood(data=data, output=output, missing_mask=data.isnan(), loss_reduction="none")
    assert ll.shape == (12,), "Incorrect log likelihood shape"
    assert torch.isnan(ll[8]), "Log likelihood for missing data should be nan"
    
def test_probabilities_from_output():
    model = GradedResponse(
        latent_variables = 2,
        item_categories=[2, 3, 4],
        item_theta_relationships=torch.tensor([[True, True], [True, True], [False, True]])
    )

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}], lr=0.02, amsgrad=True
    )
    theta = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.5], [0.8, 0.6]])
    data = torch.tensor([[0, 1, 0], [0, 1, 1], [1, 2, 1], [1, 2, 3]]).float()
    for _ in range(2): # update two times with the same data
        optimizer.zero_grad()
        output = model.forward(theta)
        assert output.shape == (4, 12), "Incorrect output shape"

        loss = -model.log_likelihood(data=data, output=output)
        # Compute gradients
        loss.backward()
        optimizer.step()

    probabilities = model.probabilities_from_output(model(theta))
    assert probabilities.shape == (4, 3, 4), "Incorrect probabilities shape"
    assert torch.all(probabilities[:, 0, 2:4] == 0.0), "probabilities for missing categories should be 0"
    assert torch.all(probabilities[:, 1, 3:4] == 0.0), "probabilities for missing categories should be 0"
    assert torch.allclose(probabilities.sum(dim=2), torch.ones(probabilities.shape[0], probabilities.shape[1])), "probabilities for missing categories should be 0"

def test_item_parameters(latent_variables):
    model = GradedResponse(
        latent_variables = latent_variables,
        item_categories=[2, 3, 4],
        item_theta_relationships=torch.tensor([[True], [True], [True]]).repeat(1, latent_variables)
    )
    parameters = model.item_parameters(irt_format=False)

    assert isinstance(parameters, pd.DataFrame), "Output should be a DataFrame"
    assert parameters.shape == (model.items, model.latent_variables + model.max_item_responses), "Incorrect shape of parameters DataFrame"

    parameters_irt = model.item_parameters(irt_format=True)

    assert isinstance(parameters_irt, pd.DataFrame), "Output should be a DataFrame"
    if model.latent_variables == 1:
        assert parameters_irt.shape == (model.items, model.latent_variables + model.max_item_responses - 1), "Incorrect shape of parameters DataFrame"
    else:
        assert parameters_irt.shape == (model.items, model.latent_variables + model.max_item_responses), "Incorrect shape of parameters DataFrame"

def test_item_theta_relationship_directions():
    model = GradedResponse(
        latent_variables = 2,
        item_categories=[2, 3, 4],
        item_theta_relationships=torch.tensor([[True, True], [True, True], [False, True]])
    )
    model.weight_param.data = torch.tensor([0.5, 0.6, -0.8, -1, 2])

    directions = model.item_theta_relationship_directions()
    assert directions.shape == (3, 2), "Incorrect directions shape"
    assert torch.all(directions[0, :] == 1), "Incorrect directions"
    assert torch.all(directions[1, :] == -1), "Incorrect directions"
    assert directions[2, 0] == 0, "Incorrect directions"
    assert directions[2, 1] == 1, "Incorrect directions"
