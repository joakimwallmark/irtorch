import pytest
import torch
from irtorch.torch_modules import NegationLayer, SoftplusLinear, MonotonicPolynomial

def test_MonotonicPolynomial_init():
    # Test if the MonotonicPolynomial initializes correctly with a valid degree.
    degree = 3
    model = MonotonicPolynomial(degree)
    assert isinstance(model, MonotonicPolynomial)
    assert model.k == 1
    assert model.intercept.shape == (1,)
    assert model.omega.shape == (1,)
    assert model.alphas.shape == (model.k,)
    assert model.taus.shape == (model.k,)
    degree = 4
    # Test if the MonotonicPolynomial raises a ValueError for an even degree.
    with pytest.raises(ValueError, match="Degree must be an uneven number."):
        MonotonicPolynomial(degree)

def test_MonotonicPolynomial_forward():
    """Test the forward method with a sample input."""
    degree = 3
    model = MonotonicPolynomial(degree)
    x = torch.randn(5, 1)  # Sample input tensor
    output = model.forward(x)
    assert output.shape == (5, 1)

    # Testing with specific input to check for monotonicity constraints
    x_test = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    output_test = model(x_test)
    
    # Check if the output shape is as expected
    assert output_test.shape == (3, 1), "Incorrect output shape"

    # Ensure the output increases for increasing input (monotonic constraint)
    assert torch.all(output_test[1:] >= output_test[:-1]), "Output is not monotonic"

    # Perform a backward pass to ensure gradients are computed
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss = output_test.sum()
    loss.backward()

    # Ensure gradients are not None
    for param in model.parameters():
        assert param.grad is not None, "Gradients not computed for all parameters"

    # Ensure the model parameters are updated
    original_params = [param.clone() for param in model.parameters()]
    optimizer.step()
    for original, updated in zip(original_params, model.parameters()):
        assert not torch.equal(original, updated), "Model parameters did not update"


def test_SoftplusLinear_forward():
    zero_outputs=torch.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1])
    sp_linear = SoftplusLinear(
        in_features=2,
        out_features=9,
        zero_outputs=zero_outputs,
    )

    original_weights = sp_linear.raw_weight_param.clone()
    original_bias = sp_linear.bias_param.clone()

    # Each row increases in all inputs, so the output should increase over rows
    optimizer = torch.optim.Adam(
        [{"params": sp_linear.parameters()}], lr=0.02, amsgrad=True
    )
    for _ in range(2):
        optimizer.zero_grad()
        result = sp_linear(torch.tensor([[-1.0, 0.1], [0.1, 1.0], [1.0, 2.0]]))
        assert result.shape == (3, 9), "Incorrect output shape"
        assert torch.all((result[1:] - result[:-1])[:, sp_linear.free_bias.bool()] > 0), "Not all updated columns are strictly increasing over rows."
        assert torch.all(result[:, zero_outputs] == 0.0)
        
        result.sum().backward()
        assert torch.all(sp_linear.bias_param.grad != 0) , "Not all bias gradients are computed."
        assert torch.all(sp_linear.raw_weight_param.grad != 0) , "Not all weight gradients are computed."
        optimizer.step()

    assert torch.all(original_weights != sp_linear.raw_weight_param), "Free weights should have changed"
    assert torch.all(original_bias != sp_linear.bias_param), "Free weights should have changed"

def test_SoftplusLinear_forward_with_separate_groups():
    zero_inputs=torch.tensor([0, 0, 0, 0, 1, 0])
    zero_outputs=torch.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1])
    sp_linear = SoftplusLinear(
        in_features=6,
        out_features=9,
        separate_inputs=torch.tensor([1, 3, 2]),
        separate_outputs=torch.tensor([2, 2, 5]),
        zero_inputs=zero_inputs,
        zero_outputs=zero_outputs,
    )

    original_weights = sp_linear.raw_weight_param.clone()
    original_bias = sp_linear.bias_param.clone()

    # Each row increases in all inputs, so the output should increase over rows
    optimizer = torch.optim.Adam(
        [{"params": sp_linear.parameters()}], lr=0.02, amsgrad=True
    )
    for _ in range(2):
        optimizer.zero_grad()
        result = sp_linear(torch.tensor([[-1.0, -0.4, 0.3, 0.6, 1.0, 1.1],
                                         [0.1, 0.2, 0.4, 0.7, 1.2, 1.3],
                                         [0.4, 0.3, 4.2, 6.1, 8.0, 8.5]]))
        assert result.shape == (3, 9), "Incorrect output shape"
        assert torch.all((result[1:] - result[:-1])[:, sp_linear.free_bias.bool()] > 0), "Not all updated columns are strictly increasing over rows."
        assert torch.all(result[:, zero_outputs] == 0.0)
        
        result.sum().backward()
        assert torch.all(sp_linear.bias_param.grad != 0) , "Not all bias gradients are computed."
        assert torch.all(sp_linear.raw_weight_param.grad != 0) , "Not all weight gradients are computed."
        optimizer.step()

    assert torch.all(original_weights != sp_linear.raw_weight_param), "Free weights should have changed"
    assert torch.all(original_bias != sp_linear.bias_param), "Free weights should have changed"


def test_NegationLayer_forward():
    neg_layer = NegationLayer(
        item_theta_relationships=torch.tensor([1, 0, 1]),
        inputs_per_items=3,
        zero_outputs=torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 1]).bool(),
    )
    with torch.no_grad():
        neg_layer.weight_param[0].fill_(2)
        neg_layer.weight_param[1].fill_(-1)

    # neg_layer.weights
    input = torch.tensor([[2.0] * 9, [1.0] * 9, [-1.0] * 9])
    input[:, 8] = 0.0
    input[:, 2:6] = 0.0
    result = neg_layer(input)

    result.sum().backward()
    assert torch.all(neg_layer.weight_param.grad != 0) , "Not all weight gradients are computed."
    assert torch.equal(result, torch.tensor([
        [ 4.,  4., 0., 0.,  0.,  0., -2., -2., 0.],
        [ 2.,  2., 0., 0.,  0.,  0.,  -1.,  -1., 0.],
        [-2., -2., 0., 0.,  0.,  0.,  1.,  1., 0.]
    ]))

def test_NegationLayer_all_item_weights():
    neg_layer = NegationLayer(
        item_theta_relationships=torch.tensor([1, 0, 1]),
        inputs_per_items=3,
        zero_outputs=torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 1]).bool(),
    )
    with torch.no_grad():
        neg_layer.weight_param[0].fill_(2)
        neg_layer.weight_param[1].fill_(-1)
        
    assert neg_layer.all_item_weights().equal(torch.tensor([ 2.,  0., -1.]))


def test_separate_weights():
    separate_inputs = torch.tensor([3, 2])
    separate_outputs = torch.tensor([3, 2])
    model = SoftplusLinear(5, 5)
    result = model.separate_weights(model.in_features, model.out_features, separate_inputs, separate_outputs)
    assert result.shape == (model.out_features, model.in_features)
    assert torch.all(result[0:3, 0:3] == True)
    assert torch.all(result[3:5, 3:5] == True)
    assert torch.all(result[0:3, 3:5] == False)
    assert torch.all(result[3:5, 0:3] == False)