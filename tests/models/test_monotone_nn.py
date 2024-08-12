import pytest
import torch
from irtorch.models import MonotoneNN

def test_mc_correct_output_idx():
    model = MonotoneNN(
        latent_variables = 2,
        hidden_dim=[6],
        item_categories=[3, 4],
        use_bounded_activation=True,
        mc_correct=[1, 0]
    )

    assert torch.equal(model.mc_correct_output_idx, torch.tensor([False, True,  False, False,  True, False, False, False]))


def test_log_likelihood():
    model = MonotoneNN(
        latent_variables = 2,
        hidden_dim=[6],
        item_categories=[3, 4],
        use_bounded_activation=True
    )

    data = torch.tensor([[0, 2], [1, 3], [2, 3]]).float()
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


def test_split_activation():
    model = MonotoneNN(2, [2, 2, 2], [6], use_bounded_activation=True)

    input_tensor = torch.cat((torch.ones(2, 3), -torch.ones(2, 3)), dim=1).requires_grad_()
    output_tensor_true = model.split_activation(input_tensor)
    output_tensor_true.sum().backward()
    assert output_tensor_true.shape == input_tensor.shape, "Output shape is incorrect with use_bounded_activation=True"
    assert input_tensor.grad is not None, "Gradients are not being tracked"
    assert torch.allclose(output_tensor_true, torch.tensor([
        [ 1.0000,  0.6321,  1.0000, -0.6321, -1.0000, -1.0000],
        [ 1.0000,  0.6321,  1.0000, -0.6321, -1.0000, -1.0000]
    ]), atol=1e-4), "Incorrect output values"

    # Test when use_bounded_activation is False
    input_tensor = torch.cat((torch.ones(2, 3), -torch.ones(2, 3)), dim=1).requires_grad_()
    model.use_bounded_activation=False
    output_tensor_false = model.split_activation(input_tensor)
    output_tensor_false.sum().backward()
    assert output_tensor_false.shape == input_tensor.shape, "Output shape is incorrect with use_bounded_activation=False"
    assert input_tensor.grad is not None, "Gradients are not being tracked"
    assert torch.allclose(output_tensor_false, torch.tensor([
        [ 1.0000,  0.6321,  1.0000, -1.0000, -0.6321, -1.0000],
        [ 1.0000,  0.6321,  1.0000, -1.0000, -0.6321, -1.0000]
    ]), atol=1e-4), "Incorrect output values"

@pytest.mark.parametrize("separate", ["items", "categories"])
def test_forward_ordered(separate):
    hidden_dim = [3, 6]
    model = MonotoneNN(
        latent_variables = 2,
        item_categories=[2, 3, 3],
        hidden_dim=hidden_dim,
        mc_correct=None,
        separate=separate,
        item_theta_relationships=torch.tensor([[1, 1], [1, 1], [0, 1]], dtype=torch.bool),
        negative_latent_variable_item_relationships=True,
        use_bounded_activation=True
    )

    original_parameter_dictionary = {k: v.clone() for k, v in model.named_parameters()}

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}], lr=0.02, amsgrad=True
    )
    theta = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.5], [0.8, 0.6]])
    data = torch.tensor([[0, 1, 0], [1, 1, 1], [1, 2, 1], [1, 2, 2]]).float()
    for _ in range(2): # update two times with the same data
        optimizer.zero_grad()
        output = model.forward(theta)
        assert output.shape == (4, 9), "Incorrect output shape"

        loss = -model.log_likelihood(data=data, output=output)
        loss.backward()
        optimizer.step()
        
    # Assert that the parameters have updated
    for name, param in model.named_parameters():
        original_parameters = original_parameter_dictionary[name]
        assert torch.all(original_parameters != param), f"Parameters for {name} should have changed"

@pytest.mark.parametrize("separate", ["items", "categories"])
def test_forward_mc(separate):
    hidden_dim = [3, 6]
    model = MonotoneNN(
        latent_variables = 2,
        item_categories=[2, 3, 3],
        hidden_dim=hidden_dim,
        mc_correct=[1, 0, 2],
        separate=separate,
        item_theta_relationships=torch.tensor([[1, 1], [1, 1], [0, 1]], dtype=torch.bool),
        use_bounded_activation=True
    )

    original_parameter_dictionary = {k: v.clone() for k, v in model.named_parameters()}

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}], lr=0.02, amsgrad=True
    )
    theta = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.5], [0.8, 0.6]])
    data = torch.tensor([[0, 1, 0], [1, 1, 1], [1, 2, 1], [1, 2, 2]]).float()
    for _ in range(2): # update two times with the same data
        optimizer.zero_grad()
        output = model.forward(theta)
        assert output.shape == (4, 9), "Incorrect output shape"

        loss = -model.log_likelihood(data=data, output=output)
        loss.backward()
        optimizer.step()
        
    # Assert that the parameters have updated
    for name, param in model.named_parameters():
        original_parameters = original_parameter_dictionary[name]
        assert torch.all(original_parameters != param), f"Parameters for {name} should have changed"


def test_probabilities_from_output():
    model = MonotoneNN(
        latent_variables = 2,
        item_categories=[2, 3, 4],
        hidden_dim=[6],
        item_theta_relationships=torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.bool),
        use_bounded_activation=True
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
    model = MonotoneNN(
        latent_variables = 2,
        item_categories=[2, 3, 4],
        hidden_dim=[6],
        item_theta_relationships=torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.bool),
        use_bounded_activation=True
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
