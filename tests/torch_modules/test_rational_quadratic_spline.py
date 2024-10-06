import pytest
import torch
from irtorch.torch_modules import RationalQuadraticSpline

def test_rational_quadratic_spline_forward():
    """Test the forward transformation of RationalQuadraticSpline."""
    variables = 2
    spline = RationalQuadraticSpline(variables=variables)
    inputs = torch.randn(10, variables)
    outputs, logabsdet = spline(inputs)
    assert outputs.shape == inputs.shape
    assert logabsdet.shape == inputs.shape

def test_rational_quadratic_spline_inverse():
    """Test the inverse transformation of RationalQuadraticSpline."""
    variables = 2
    spline = RationalQuadraticSpline(variables=variables)
    inputs = torch.randn(10, variables)
    outputs, logabsdet = spline(inputs, inverse=True)
    assert outputs.shape == inputs.shape
    assert logabsdet.shape == inputs.shape

def test_rational_quadratic_spline_forward_inverse_consistency():
    """Test that applying forward and inverse transformations returns the original inputs."""
    variables = 2
    spline = RationalQuadraticSpline(variables=variables)
    inputs = torch.randn(10, variables)
    outputs, _ = spline(inputs)
    recon_inputs, _ = spline(outputs, inverse=True)
    assert torch.allclose(inputs, recon_inputs, atol=1e-6)

def test_rational_quadratic_spline_invalid_bin_width():
    """Test that a ValueError is raised when min_bin_width is too large."""
    with pytest.raises(ValueError):
        variables = 2
        spline = RationalQuadraticSpline(variables=variables, num_bins=10, min_bin_width=0.2)
        inputs = torch.randn(10, variables)
        spline(inputs)

def test_rational_quadratic_spline_invalid_bin_height():
    """Test that a ValueError is raised when min_bin_height is too large."""
    with pytest.raises(ValueError):
        variables = 2
        spline = RationalQuadraticSpline(variables=variables, num_bins=10, min_bin_height=0.2)
        inputs = torch.randn(10, variables)
        spline(inputs)

def test_rational_quadratic_spline_jacobian_determinant():
    """Test that the log determinant does not contain NaNs or Infs."""
    variables = 2
    spline = RationalQuadraticSpline(variables=variables)
    inputs = torch.randn(10, variables)
    _, logabsdet = spline(inputs)
    assert not torch.isnan(logabsdet).any()
    assert not torch.isinf(logabsdet).any()
