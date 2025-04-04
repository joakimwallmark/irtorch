import pytest
import torch
from irtorch.torch_modules import GaussianMixtureModel

def test_gaussian_mixture_model_1d():
    torch.manual_seed(42)  # Fixed seed for reproducibility
    gmm = GaussianMixtureModel(n_components=2, n_features=1)
    
    # Generate well-separated data
    n_samples = 2000
    data = torch.cat([
        torch.normal(mean=-3, std=0.5, size=(n_samples // 2, 1)),
        torch.normal(mean=3, std=0.5, size=(n_samples // 2, 1)),
    ])
    
    gmm.fit(data)  # Removed n_init parameter
    
    # Sort means to ensure consistent comparison
    sorted_means = torch.sort(gmm.means.view(-1))[0]
    expected_means = torch.tensor([-3.0, 3.0])
    
    assert torch.allclose(sorted_means, expected_means, atol=0.2)
    assert torch.allclose(gmm.weights, torch.tensor([0.5, 0.5]), atol=0.05)
    assert torch.allclose(gmm.covariances, torch.tensor([[[0.25]], [[0.25]]]), atol=0.1)

def test_gaussian_mixture_model_2d():
    torch.manual_seed(42)
    gmm = GaussianMixtureModel(n_components=2, n_features=2)
    
    # Generate well-separated 2D data with known parameters
    n_samples_1, n_samples_2 = 3000, 1000
    mean_1, mean_2 = torch.tensor([3., 2.]), torch.tensor([-3., -2.])
    std_1, std_2 = torch.tensor([1., 0.5]), torch.tensor([1., 0.5])
    
    data = torch.cat([
        torch.cat([
            torch.normal(mean=mean_1[0], std=std_1[0], size=(n_samples_1, 1)),
            torch.normal(mean=mean_1[1], std=std_1[1], size=(n_samples_1, 1)),
        ], dim=1),
        torch.cat([
            torch.normal(mean=mean_2[0], std=std_2[0], size=(n_samples_2, 1)),
            torch.normal(mean=mean_2[1], std=std_2[1], size=(n_samples_2, 1)),
        ], dim=1)
    ])
    
    gmm.fit(data)  # Removed n_init parameter
    
    # Sort components by first dimension of means for consistent comparison
    idx = torch.argsort(gmm.means[:, 0])
    sorted_means = gmm.means[idx]
    sorted_covs = gmm.covariances[idx]
    sorted_weights = gmm.weights[idx]
    
    expected_means = torch.stack([mean_2, mean_1])
    expected_weights = torch.tensor([0.25, 0.75])
    expected_covs = torch.stack([
        torch.diag(std_2 * std_2),
        torch.diag(std_1 * std_1)
    ])
    
    assert torch.allclose(sorted_means, expected_means, atol=0.2)
    assert torch.allclose(sorted_weights, expected_weights, atol=0.05)
    assert torch.allclose(sorted_covs, expected_covs, atol=0.1)

def test_pdf():
    gmm = GaussianMixtureModel(n_components=2, n_features=2)
    gmm.weights = torch.nn.Parameter(torch.tensor([0.75, 0.25]))
    gmm.means = torch.nn.Parameter(torch.tensor([[0., 0.], [0., 0.]]))
    gmm.covariances = torch.nn.Parameter(torch.tensor([
        [[1., 0.], [0., 1.]],
        [[1., 0.], [0., 1.]]
    ]))
    
    # Test multiple points
    test_points = torch.tensor([
        [0., 0.],  # At the mean
        [1., 1.],  # Away from mean
        [-1., -1.],  # Symmetric point
        [2., 2.],  # Further away
    ])
    
    pdfs = gmm.pdf(test_points)
    
    # PDF should be highest at the mean and decrease with distance
    assert torch.all(pdfs[0] > pdfs[1])
    assert torch.allclose(pdfs[1], pdfs[2])  # Symmetric points should have same density
    assert torch.all(pdfs[1] > pdfs[3])  # Density should decrease with distance
    
    # Test PDF normalization
    x = torch.linspace(-5, 5, 100)
    y = torch.linspace(-5, 5, 100)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    pdfs = gmm.pdf(points)
    total_prob = pdfs.sum() * ((x[1] - x[0]) * (y[1] - y[0]))
    assert torch.isclose(total_prob, torch.tensor(1.0), atol=0.1)

def test_initialization():
    """Test initialization with invalid parameters"""
    with pytest.raises(ValueError):
        GaussianMixtureModel(n_components=0, n_features=1)
    
    with pytest.raises(ValueError):
        GaussianMixtureModel(n_components=1, n_features=0)

def test_fit_input_validation():
    """Test fit method with invalid inputs"""
    gmm = GaussianMixtureModel(n_components=2, n_features=2)
    
    # Wrong number of features
    with pytest.raises(ValueError):
        gmm.fit(torch.randn(100, 3))
    
    # Empty data
    with pytest.raises(ValueError):
        gmm.fit(torch.randn(0, 2))

def test_convergence():
    """Test convergence with simple data"""
    torch.manual_seed(42)
    gmm = GaussianMixtureModel(n_components=2, n_features=1)
    
    # Generate simple data with clear clusters
    data = torch.cat([
        torch.normal(-5, 0.1, size=(100, 1)),
        torch.normal(5, 0.1, size=(100, 1))
    ])
    
    # First fit
    gmm.fit(data)
    params1 = (gmm.weights.clone(), gmm.means.clone(), gmm.covariances.clone())
    
    # Second fit with same data should give similar results
    gmm.fit(data)
    params2 = (gmm.weights.clone(), gmm.means.clone(), gmm.covariances.clone())
    
    # Compare parameters instead of log likelihood
    assert torch.allclose(params1[0], params2[0], rtol=1e-3)
    assert torch.allclose(params1[1], params2[1], rtol=1e-3)
    assert torch.allclose(params1[2], params2[2], rtol=1e-3)
