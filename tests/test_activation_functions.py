import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck
from irtorch.activation_functions import ConvexELU, BoundedELU

def test_ConvexELU_forward():
    # Test the forward pass
    input = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    alpha = 0.5
    result = ConvexELU.apply(input, alpha)
    expected = torch.tensor([-1.0, 0.0, 0.31606], requires_grad=True)  # Expected output
    assert torch.allclose(result, expected), "Forward pass is incorrect"

def test_ConvexELU_backward():
    # Test the backward pass using gradcheck
    input = torch.randn(6, dtype=torch.double, requires_grad=True)
    alpha = 0.5
    assert gradcheck(ConvexELU.apply, (input, alpha)), "Backward pass (gradient) failed"

def test_ConvexELU_backward_double_backward():
    # Test the double backward pass using gradgradcheck
    # Required for second order optimization methods
    input = torch.randn(6, dtype=torch.double, requires_grad=True)
    alpha = 0.5
    assert gradgradcheck(ConvexELU.apply, (input, alpha)), "Double backward pass (gradient of gradient) failed"

def test_BoundedELU_forward():
    # Test the forward pass
    input = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    alpha = 0.5
    result = BoundedELU.apply(input, alpha)
    expected = torch.tensor([-1.31606, -0.5, 0.0, 0.5, 1.31606], requires_grad=True)  # Expected output
    assert torch.allclose(result, expected), "Forward pass is incorrect"

def test_BoundedELU_backward():
    # Test the backward pass using gradcheck
    input = torch.randn(3, dtype=torch.double, requires_grad=True)
    alpha = 0.5
    assert gradcheck(BoundedELU.apply, (input, alpha)), "Backward pass (gradient) failed"

def test_BoundedELU_backward_double_backward():
    # Test the double backward pass using gradgradcheck
    # Required for second order optimization methods
    input = torch.randn(3, dtype=torch.double, requires_grad=True)
    alpha = 0.5
    assert gradgradcheck(BoundedELU.apply, (input, alpha)), "Double backward pass (gradient of gradient) failed"
