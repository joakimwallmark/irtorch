import pytest
import torch
import pandas as pd
from irtorch.models import Parametric

def test_1pl_forward():
    model = Parametric(
        latent_variables = 2,
        item_categories=[2, 2, 2],
        model = "1PL",
        item_z_relationships=torch.tensor([[True, True], [True, True], [False, True]])
    )
    original_weights = model.weight_param.clone()
    original_bias = model.bias_param.clone()

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}], lr=0.02, amsgrad=True
    )
    z = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.5], [0.5, 0.6]])
    data = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1]]).float()
    for _ in range(2): # update two times with the same data
        optimizer.zero_grad()
        output = model.forward(z)
        assert output.shape == (4, 6), "Incorrect output shape"

        loss = -model.log_likelihood(data=data, output=output)
        # Compute gradients
        loss.backward()
        optimizer.step()

    assert torch.all(original_weights != model.weight_param), "Free weights should have changed"
    assert torch.all(original_bias != model.bias_param), "Free bias should have changed"


def test_2pl_forward():
    model = Parametric(
        latent_variables = 2,
        model = "2PL",
        item_categories=[2, 2, 2],
        item_z_relationships=torch.tensor([[True, True], [True, True], [False, True]])
    )
    original_weights = model.weight_param.clone()
    original_bias = model.bias_param.clone()

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}], lr=0.02, amsgrad=True
    )
    z = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.5], [0.5, 0.6]])
    data = torch.tensor([[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 1]]).float()
    for _ in range(2): # update two times with the same data
        optimizer.zero_grad()
        output = model.forward(z)
        assert output.shape == (4, 6), "Incorrect output shape"

        loss = -model.log_likelihood(data=data, output=output)
        # Compute gradients
        loss.backward()
        optimizer.step()

    assert torch.all(original_weights != model.weight_param), "Free weights should have changed"
    assert torch.all(original_bias != model.bias_param), "Free bias should have changed"

def test_gpc_forward():
    model = Parametric(
        latent_variables = 2,
        model = "GPC",
        item_categories=[2, 3, 3],
        item_z_relationships=torch.tensor([[True, True], [True, True], [False, True]])
    )
    original_weights = model.weight_param.clone()
    original_bias = model.bias_param.clone()

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}], lr=0.02, amsgrad=True
    )
    z = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.5], [0.8, 0.6]])
    data = torch.tensor([[0, 1, 0], [0, 1, 1], [1, 2, 1], [1, 2, 2]]).float()
    for _ in range(2): # update two times with the same data
        optimizer.zero_grad()
        output = model.forward(z)
        assert output.shape == (4, 9), "Incorrect output shape"

        loss = -model.log_likelihood(data=data, output=output)
        # Compute gradients
        loss.backward()
        optimizer.step()

    assert torch.all(original_weights != model.weight_param), "Free weights should have changed"
    assert torch.all(original_bias != model.bias_param), "Free bias should have changed"

def test_nominal_forward():
    # With reference category
    model = Parametric(
        latent_variables = 2,
        model = "nominal",
        item_categories=[2, 3, 3],
        item_z_relationships=torch.tensor([[True, True], [True, True], [False, True]]),
        reference_category=True
    )
    original_weights = model.weight_param.clone()
    original_bias = model.bias_param.clone()

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}], lr=0.02, amsgrad=True
    )
    z = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.5], [0.8, 0.6]])
    data = torch.tensor([[0, 1, 0], [0, 1, 1], [1, 2, 1], [1, 2, 2]]).float()
    for _ in range(2): # update two times with the same data
        optimizer.zero_grad()
        output = model.forward(z)
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
    model = Parametric(
        latent_variables = 2,
        model = "nominal",
        item_categories=[2, 3, 3],
        item_z_relationships=torch.tensor([[True, True], [True, True], [False, True]]),
        reference_category=False
    )
    original_weights = model.weight_param.clone()
    original_bias = model.bias_param.clone()

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}], lr=0.02, amsgrad=True
    )
    z = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.5], [0.8, 0.6]])
    data = torch.tensor([[0, 1, 0], [0, 1, 1], [1, 2, 1], [1, 2, 2]]).float()
    for _ in range(2): # update two times with the same data
        optimizer.zero_grad()
        output = model.forward(z)
        assert output.shape == (4, 9), "Incorrect output shape"

        loss = -model.log_likelihood(data=data, output=output)
        # Compute gradients
        loss.backward()
        optimizer.step()

    assert model.weight_param.shape == torch.Size([13]), "Incorrect weight shape"
    assert model.bias_param.shape == torch.Size([8]), "Incorrect weight shape"
    assert torch.all(original_weights != model.weight_param), "Free weights should have changed"
    assert torch.all(original_bias != model.bias_param), "Free bias should have changed"
    
def test_probabilities_from_output():
    model = Parametric(
        latent_variables = 2,
        model = "GPC",
        item_categories=[2, 3, 4],
        item_z_relationships=torch.tensor([[True, True], [True, True], [False, True]])
    )

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}], lr=0.02, amsgrad=True
    )
    z = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.5], [0.8, 0.6]])
    data = torch.tensor([[0, 1, 0], [0, 1, 1], [1, 2, 1], [1, 2, 3]]).float()
    for _ in range(2): # update two times with the same data
        optimizer.zero_grad()
        output = model.forward(z)
        assert output.shape == (4, 12), "Incorrect output shape"

        loss = -model.log_likelihood(data=data, output=output)
        # Compute gradients
        loss.backward()
        optimizer.step()

    probabilities = model.probabilities_from_output(model(z))
    assert probabilities.shape == (4, 3, 4), "Incorrect probabilities shape"
    assert torch.all(probabilities[:, 0, 2:4] == 0.0), "probabilities for missing categories should be 0"
    assert torch.all(probabilities[:, 1, 3:4] == 0.0), "probabilities for missing categories should be 0"
    assert torch.allclose(probabilities.sum(dim=2), torch.ones(probabilities.shape[0], probabilities.shape[1])), "probabilities for missing categories should be 0"

def test_item_parameters(latent_variables):
    for model_type in ["nominal", "1PL", "2PL", "GPC"]:
        if model_type in ["GPC", "nominal"]:
            item_categories = [2, 3, 4]
        else:
            item_categories = [2, 2, 2]
        model = Parametric(
            latent_variables = latent_variables,
            model = model_type,
            item_categories=item_categories,
            item_z_relationships=torch.tensor([[True], [True], [True]]).repeat(1, latent_variables)
        )
        parameters = model.item_parameters(IRT_format=False)

        assert isinstance(parameters, pd.DataFrame), f"Output should be a DataFrame for {model_type} model"
        if model_type == "nominal":
            assert parameters.shape == (model.items, (model.latent_variables + 1) * model.max_item_responses), f"Incorrect shape of parameters DataFrame for {model_type} model"
        else:
            assert parameters.shape == (model.items, model.latent_variables + model.max_item_responses), f"Incorrect shape of parameters DataFrame for {model_type} model"

        parameters_irt = model.item_parameters(IRT_format=True)

        assert isinstance(parameters_irt, pd.DataFrame), f"Output should be a DataFrame for {model_type} model"
        if model_type == "nominal":
            assert parameters_irt.shape == (model.items, (model.latent_variables + 1) * model.max_item_responses), f"Incorrect shape of parameters DataFrame for {model_type} model"
        elif model.latent_variables == 1:
            assert parameters_irt.shape == (model.items, model.latent_variables + model.max_item_responses - 1), f"Incorrect shape of parameters DataFrame for {model_type} model"
        else:
            assert parameters_irt.shape == (model.items, model.latent_variables + model.max_item_responses), f"Incorrect shape of parameters DataFrame for {model_type} model"

def test_item_z_relationship_directions():
    model = Parametric(
        latent_variables = 2,
        model = "GPC",
        item_categories=[2, 3, 4],
        item_z_relationships=torch.tensor([[True, True], [True, True], [False, True]])
    )
    model.weight_param.data = torch.tensor([0.5, 0.6, -0.8, -1, 2])

    directions = model.item_z_relationship_directions()
    assert directions.shape == (3, 2), "Incorrect directions shape"
    assert torch.all(directions[0, :] == 1), "Incorrect directions"
    assert torch.all(directions[1, :] == -1), "Incorrect directions"
    assert directions[2, 0] == 0, "Incorrect directions"
    assert directions[2, 1] == 1, "Incorrect directions"

    model = Parametric(
        latent_variables = 2,
        model = "nominal",
        item_categories=[2, 3, 3],
        item_z_relationships=torch.tensor([[True, True], [True, True], [False, True]]),
        mc_correct=[2, 2, 3],
        reference_category=True
    )

    model.weight_param.data = torch.tensor([1.0, -1.0, -1.0, -1.0, -1.0, 3.0, 1.0, 1.0])

    directions = model.item_z_relationship_directions()
    assert directions.shape == (3, 2), "Incorrect directions shape"
    assert directions[0, 0] == 1, "Incorrect directions"
    assert directions[0, 1] == -1, "Incorrect directions"
    assert directions[1, 0] == -1, "Incorrect directions"
    assert directions[1, 1] == -1, "Incorrect directions"
    assert directions[2, 0] == 0, "Incorrect directions"
    assert directions[2, 1] == 1, "Incorrect directions"

    # Without reference category
    model = Parametric(
        latent_variables = 2,
        model = "nominal",
        item_categories=[2, 3, 3],
        item_z_relationships=torch.tensor([[True, True], [True, True], [False, True]]),
        mc_correct=[2, 2, 3],
        reference_category=False
    )

    model.weight_param.data = torch.tensor([0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, -1.0, -1.0, 3.0, 0.0, 1.0, 1.0])

    directions = model.item_z_relationship_directions()
    assert directions.shape == (3, 2), "Incorrect directions shape"
    assert directions[0, 0] == 1, "Incorrect directions"
    assert directions[0, 1] == -1, "Incorrect directions"
    assert directions[1, 0] == -1, "Incorrect directions"
    assert directions[1, 1] == -1, "Incorrect directions"
    assert directions[2, 0] == 0, "Incorrect directions"
    assert directions[2, 1] == 1, "Incorrect directions"
