import pytest
import torch
from irtorch.torch_modules import NegationLayer

def test_NegationLayer_forward():
    neg_layer = NegationLayer(
        item_theta_relationships=torch.tensor([1, 0, 1]),
        inputs_per_item=3,
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
        inputs_per_item=3,
        zero_outputs=torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 1]).bool(),
    )
    with torch.no_grad():
        neg_layer.weight_param[0].fill_(2)
        neg_layer.weight_param[1].fill_(-1)
        
    assert neg_layer.all_item_weights().equal(torch.tensor([ 2.,  0., -1.]))
