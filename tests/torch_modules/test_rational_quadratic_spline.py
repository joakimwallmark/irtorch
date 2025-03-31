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

def test_rational_quadratic_spline_out_of_bounds_forward():
    """Test the forward transformation of RationalQuadraticSpline with out-of-bounds values."""
    variables = 2
    spline = RationalQuadraticSpline(
        variables=variables,
        lower_input_bound=0.0,
        upper_input_bound=1.0,
        lower_output_bound=0.0,
        upper_output_bound=1.0
    )
    
    # Create inputs with values below, within, and above bounds
    inputs = torch.tensor([
        [-0.5, 0.5],  # below bounds in first dim
        [1.5, 0.5],   # above bounds in first dim
        [0.5, -0.5],  # below bounds in second dim
        [0.5, 1.5],   # above bounds in second dim
        [0.5, 0.5]    # within bounds
    ], dtype=torch.float32)
    
    outputs, logabsdet = spline(inputs)
    
    # Check shapes
    assert outputs.shape == inputs.shape
    assert logabsdet.shape == inputs.shape
    
    # Check that out-of-bounds values are handled by linear extrapolation
    default_deriv = 1.0  # (upper_output_bound - lower_output_bound) / (upper_input_bound - lower_input_bound)
    
    # Check below bounds
    assert torch.allclose(outputs[0, 0], torch.tensor(-0.5 * default_deriv))  # lower_output + (x - lower_input) * deriv
    assert torch.allclose(outputs[2, 1], torch.tensor(-0.5 * default_deriv))
    
    # Check above bounds
    assert torch.allclose(outputs[1, 0], torch.tensor(1.0 + 0.5 * default_deriv))  # upper_output + (x - upper_input) * deriv
    assert torch.allclose(outputs[3, 1], torch.tensor(1.0 + 0.5 * default_deriv))

def test_rational_quadratic_spline_out_of_bounds_inverse():
    """Test the inverse transformation of RationalQuadraticSpline with out-of-bounds values."""
    variables = 2
    spline = RationalQuadraticSpline(
        variables=variables,
        lower_input_bound=0.0,
        upper_input_bound=1.0,
        lower_output_bound=0.0,
        upper_output_bound=1.0
    )
    
    # Create inputs with values below, within, and above bounds
    inputs = torch.tensor([
        [-0.5, 0.5],  # below bounds in first dim
        [1.5, 0.5],   # above bounds in first dim
        [0.5, -0.5],  # below bounds in second dim
        [0.5, 1.5],   # above bounds in second dim
        [0.5, 0.5]    # within bounds
    ], dtype=torch.float32)
    
    outputs, logabsdet = spline(inputs, inverse=True)
    
    # Check shapes
    assert outputs.shape == inputs.shape
    assert logabsdet.shape == inputs.shape
    
    # Check that out-of-bounds values are handled by linear extrapolation
    default_deriv = 1.0  # (upper_output_bound - lower_output_bound) / (upper_input_bound - lower_input_bound)
    
    # Check below bounds
    assert torch.allclose(outputs[0, 0], torch.tensor(-0.5 / default_deriv))  # lower_input + (x - lower_output) / deriv
    assert torch.allclose(outputs[2, 1], torch.tensor(-0.5 / default_deriv))
    
    # Check above bounds
    assert torch.allclose(outputs[1, 0], torch.tensor(1.0 + 0.5 / default_deriv))  # upper_input + (x - upper_output) / deriv
    assert torch.allclose(outputs[3, 1], torch.tensor(1.0 + 0.5 / default_deriv))

def test_rational_quadratic_spline_forward_inverse_consistency_with_bounds():
    """Test that applying forward and inverse transformations returns the original inputs, including out-of-bounds values."""
    variables = 2
    spline = RationalQuadraticSpline(variables=variables)
    
    # Create inputs with values below, within, and above bounds
    inputs = torch.tensor([
        [-0.5, 0.5],  # below bounds
        [1.5, 0.5],   # above bounds
        [0.5, 0.5]    # within bounds
    ], dtype=torch.float32)
    
    # Forward then inverse should return original inputs
    outputs, _ = spline(inputs)
    recon_inputs, _ = spline(outputs, inverse=True)
    assert torch.allclose(inputs, recon_inputs, atol=1e-6)
    
    # Inverse then forward should return original inputs
    outputs, _ = spline(inputs, inverse=True)
    recon_inputs, _ = spline(outputs)
    assert torch.allclose(inputs, recon_inputs, atol=1e-6)
