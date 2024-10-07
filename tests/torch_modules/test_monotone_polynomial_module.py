import pytest
import torch
import torch.nn.functional as F
from irtorch.torch_modules import MonotonePolynomialModule

def test_MonotonePolynomialModule_init():
    degree = 5
    in_features=2
    out_features=3
    model = MonotonePolynomialModule(degree, in_features, out_features, intercept=True)
    assert isinstance(model, MonotonePolynomialModule)
    assert model.k == 2
    assert model.intercept.shape == (out_features,)
    assert model.omega.shape == (1, in_features, out_features)
    assert model.alpha.shape == (model.k, in_features, out_features)
    assert model.tau.shape == (model.k, in_features, out_features)

    with pytest.raises(ValueError, match="shared_directions must be greater than 0."):
        MonotonePolynomialModule(degree, shared_directions=0, negative_relationships=True)

    with pytest.raises(ValueError, match="out_features must be divisible by shared_directions."):
        MonotonePolynomialModule(degree, out_features=5, negative_relationships=True, shared_directions=2)

    degree = 4
    with pytest.raises(ValueError, match="Degree must be an uneven number."):
        MonotonePolynomialModule(degree)


def test_MonotonePolynomialModule_forward():
    degree = 5
    relationship_matrix=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    model = MonotonePolynomialModule(degree, in_features=2, out_features=3, relationship_matrix=relationship_matrix, shared_directions=3)
    x = torch.randn(5, 2)  # Sample input tensor
    output = model.forward(x)
    assert output.shape == (5, 3), "Incorrect output shape"

    # Testing with specific input to check for monotonicity constraints
    original_parameter_dictionary = {k: v.clone() for k, v in model.named_parameters()}
    x_test = torch.tensor([[1.0, 1.2], [2.0, 1.5], [3.0, 2.8]], requires_grad=True)
    output_test = model(x_test)
    assert output_test.shape == (3, 3), "Incorrect output shape"

    # Ensure the output increases for increasing input (monotonic constraint)
    assert torch.all(output_test[1:] >= output_test[:-1]), "Output is not monotonic"

    # Perform a backward pass to ensure gradients are computed
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss = output_test.sum()
    loss.backward()
    optimizer.step()

    # Ensure gradients are not None
    for param in model.parameters():
        assert param.grad is not None, "Gradients not computed for all parameters"

    # Assert that the parameters have updated
    for name, param in model.named_parameters():
        original_parameters = original_parameter_dictionary[name]
        if name in ["omega", "alpha", "tau"]:
            assert torch.all(original_parameters[:, relationship_matrix] != param[:, relationship_matrix]), f"Parameters for {name} should have changed"
        elif name == "directions":
            assert torch.all(original_parameters[relationship_matrix] != param[relationship_matrix]), f"Parameters for {name} should have changed"
        else:
            assert torch.all(original_parameters != param), f"Parameters for {name} should have changed"

def test_MonotonePolynomialModule_get_polynomial_coefficients():
    degree = 3
    in_features = 2
    out_features = 3
    intercept = True
    negative_relationships = True
    shared_directions = 1
    relationship_matrix = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)

    model = MonotonePolynomialModule(
        degree,
        in_features=in_features,
        out_features=out_features,
        intercept=intercept,
        relationship_matrix=relationship_matrix,
        negative_relationships=negative_relationships,
        shared_directions=shared_directions
    )

    # Initialize the parameters for testing
    model.omega.data = torch.ones(1, in_features, out_features)
    model.alpha.data = torch.ones(model.k, in_features, out_features)
    model.tau.data = torch.zeros(model.k, in_features, out_features)
    model.intercept.data = torch.ones(out_features)
    model.directions.data = torch.ones(in_features, out_features // shared_directions)
    
    coefficients = model.get_polynomial_coefficients()
    
    # Expected behavior for the initialized values
    sp_tau = F.softplus(model.tau)
    b = F.softplus(model.omega)
    for i in range(model.k):
        matrix = torch.zeros((2*(i+1)+1, 2*(i+1)-1, in_features, out_features))
        range_indices = torch.arange(2*(i+1)-1)
        matrix[range_indices, range_indices, :, :] = 1
        matrix[range_indices + 1, range_indices, :, :] = -2 * model.alpha[i]
        matrix[range_indices + 2, range_indices, :, :] = model.alpha[i] ** 2 + sp_tau[i]
        b = torch.einsum('abio,bio->aio', matrix, b) / (i + 1)

    if negative_relationships:
        b.multiply_(model.directions.repeat_interleave(shared_directions, dim=1))

    # remove relationship between some items and latent variables
    if relationship_matrix is not None:
        b[:, ~relationship_matrix] = 0.0

    expected_coefficients = b, model.intercept

    assert torch.allclose(coefficients[0], expected_coefficients[0])
    assert torch.allclose(coefficients[1], expected_coefficients[1])
