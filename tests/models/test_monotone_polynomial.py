import pytest
import torch
from irtorch.models import MonotonePolynomial
from irtorch.irt_dataset import PytorchIRTDataset

def test_log_likelihood():
    model = MonotonePolynomial(
        latent_variables = 2,
        item_categories=[3, 4]
    )

    data = torch.tensor([[0., 2.], [1., 3.], [2., 3.]])
    output = torch.tensor([
        [0, 0, 0, -torch.inf, -0.4, 0.2, -0.2, 0.5],
        [0, 0, 0, -torch.inf, -0.4, 0.2, -0.2, 0.5],
        [0, 0, 0, -torch.inf, -0.4, 0.2, -0.2, 0.5]
    ])
    result = model.log_likelihood(data, output, loss_reduction="none")
    assert result.shape == (6, ), "Incorrect shape for log_likelihood"
    assert torch.isclose(result.sum(), torch.tensor(-6.912685394)), "Incorrect log_likelihood sum"
    assert torch.equal(result[0], result[2]) and torch.equal(result[0], result[4]), "all first item likelihoods should be equal"
    assert torch.equal(result[3], result[5]), "same responses should be equal"
    assert torch.isclose(result[1], torch.tensor(-1.6722826)), "incorrect item likelihood"

    data = torch.tensor([[0., torch.nan], [1., 3.], [2., 3.]])
    data_irt = PytorchIRTDataset(data)
    output = torch.tensor([
        [0, 0, 0, -torch.inf, torch.nan, torch.nan, torch.nan, torch.nan],
        [0, 0, 0, -torch.inf, -0.4, 0.2, -0.2, 0.5],
        [0, 0, 0, -torch.inf, -0.4, 0.2, -0.2, 0.5]
    ])
    result = model.log_likelihood(data_irt.data, output, data_irt.mask, loss_reduction="none")
    assert result.shape == (6, ), "Incorrect shape for log_likelihood"
    assert torch.isclose(result.nansum(), torch.tensor(-5.240402698516)), "Incorrect log_likelihood sum"
    assert torch.equal(result[0], result[2]) and torch.equal(result[0], result[4]), "all first item likelihoods should be equal"
    assert torch.equal(result[3], result[5]), "same responses should be equal"
    assert torch.isnan(result[1]), "incorrect missing response should be nan"

@pytest.mark.parametrize("separate", ["items", "categories"])
def test_forward_ordered(separate):
    item_theta_relationships=torch.tensor([[1, 1], [1, 1], [0, 1]], dtype=torch.bool)
    model = MonotonePolynomial(
        latent_variables = 2,
        item_categories=[2, 3, 3],
        degree=3,
        separate=separate,
        item_theta_relationships=item_theta_relationships,
        negative_latent_variable_item_relationships=True,
    )

    original_parameter_dictionary = {k: v.clone() for k, v in model.named_parameters()}

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}], lr=0.02, amsgrad=True
    )
    theta = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.5], [0.8, 0.6]])
    data = torch.tensor([[0, 1, 0], [1, 0, 1], [1, 2, 1], [1, 2, 2]]).float()
    for _ in range(100): # update many times, otherwise tau may not change
        optimizer.zero_grad()
        output = model.forward(theta)
        assert output.shape == (4, 9), "Incorrect output shape"

        loss = -model.log_likelihood(data=data, output=output)
        loss.backward()
        optimizer.step()
        
    # Assert that the parameters have updated
    changed_indices = item_theta_relationships.transpose(0, 1)
    if separate == "categories":
        changed_indices = changed_indices.repeat_interleave(3, dim=1)
        changed_indices[:, 2] = False
    for name, param in model.named_parameters():
        original_parameters = original_parameter_dictionary[name]
        if name in ["mono_poly.omega", "mono_poly.alpha", "mono_poly.tau"]:
            assert torch.all(original_parameters[:, changed_indices] != param[:, changed_indices]), f"Parameters for {name} should have changed"
        elif name == "mono_poly.directions":
            assert torch.all(original_parameters[item_theta_relationships.transpose(0, 1)] != param[item_theta_relationships.transpose(0, 1)]), f"Parameters for {name} should have changed"
        else:
            assert torch.all(original_parameters != param), f"Parameters for {name} should have changed"

def test_probabilities_from_output():
    model = MonotonePolynomial(
        latent_variables = 2,
        item_categories=[2, 3, 4],
        item_theta_relationships=torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.bool),
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
        loss.backward()
        optimizer.step()

    probabilities = model.probabilities_from_output(model(theta))
    assert probabilities.shape == (4, 3, 4), "Incorrect probabilities shape"
    assert torch.all(probabilities[:, 0, 2:4] == 0.0), "probabilities for missing categories should be 0"
    assert torch.all(probabilities[:, 1, 3:4] == 0.0), "probabilities for missing categories should be 0"
    assert torch.allclose(probabilities.sum(dim=2), torch.ones(probabilities.shape[0], probabilities.shape[1])), "probabilities should sum to 1"

def test_item_theta_relationship_directions():
    torch.manual_seed(0)
    model = MonotonePolynomial(
        latent_variables = 2,
        item_categories=[2, 3, 4],
        item_theta_relationships=torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.bool),
    )

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}], lr=0.02, amsgrad=True
    )
    theta = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.5], [0.8, 0.6]])
    data = torch.tensor([[0, 2, 0], [0, 1, 1], [1, 0, 1], [1, 0, 3]]).float()
    for _ in range(2): # update two times with the same data
        optimizer.zero_grad()
        output = model.forward(theta)
        loss = -model.log_likelihood(data=data, output=output)
        loss.backward()
        optimizer.step()

    item_theta_relationship_directions = model.item_theta_relationship_directions()
    assert item_theta_relationship_directions.shape == (3, 2), "Incorrect item_theta_relationship_directions shape"
    assert torch.all(item_theta_relationship_directions[0, :] == 1), "item_theta_relationship_directions should be 1 for the first item"
    assert torch.all(item_theta_relationship_directions[1, :] == -1), "item_theta_relationship_directions should be -1 the second item"
    assert item_theta_relationship_directions[2, 0] == 1, "item_theta_relationship_directions should be 1 the third and first latent variable"
    assert item_theta_relationship_directions[2, 1] == 0, "item_theta_relationship_directions should be 0 the third and second latent variable"
