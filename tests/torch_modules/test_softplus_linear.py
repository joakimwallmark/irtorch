import pytest
import torch
from irtorch.torch_modules import SoftplusLinear

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
        result = sp_linear(torch.tensor([
            [-1.0, -0.4, 0.3, 0.6, 1.0, 1.1],
            [0.1, 0.2, 0.4, 0.7, 1.2, 1.3],
            [0.4, 0.3, 4.2, 6.1, 8.0, 8.5]
        ]))
        assert result.shape == (3, 9), "Incorrect output shape"
        assert torch.all((result[1:] - result[:-1])[:, sp_linear.free_bias.bool()] > 0), "Not all updated columns are strictly increasing over rows."
        assert torch.all(result[:, zero_outputs] == 0.0)
        
        result.sum().backward()
        assert torch.all(sp_linear.bias_param.grad != 0) , "Not all bias gradients are computed."
        assert torch.all(sp_linear.raw_weight_param.grad != 0) , "Not all weight gradients are computed."
        optimizer.step()

    assert torch.all(original_weights != sp_linear.raw_weight_param), "Free weights should have changed"
    assert torch.all(original_bias != sp_linear.bias_param), "Free weights should have changed"


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