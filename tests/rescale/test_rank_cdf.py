import pytest
import torch
from torch.distributions import Normal, Uniform
from irtorch.rescale import RankCDF

@pytest.fixture
def sample_theta():
    # Create a tensor with shape (10 samples, 3 latent variables)
    return torch.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0],
        [6.0, 7.0, 8.0],
        [7.0, 8.0, 9.0],
        [8.0, 9.0, 10.0],
        [9.0, 10.0, 11.0],
        [10.0, 11.0, 12.0],
    ], dtype=torch.float32)

def test_initialization_default_distributions(sample_theta):
    rank_cdf = RankCDF(theta=sample_theta)
    assert len(rank_cdf.distributions) == sample_theta.shape[1], "Incorrect number of distributions initialized."
    for dist in rank_cdf.distributions:
        assert isinstance(dist, Normal), "Default distribution should be Normal."
        assert torch.allclose(dist.mean, torch.tensor(0.0)), "Default Normal distribution mean should be 0."
        assert torch.allclose(dist.stddev, torch.tensor(1.0)), "Default Normal distribution stddev should be 1."

def test_initialization_single_distribution(sample_theta):
    uniform_dist = Uniform(0, 1)
    rank_cdf = RankCDF(theta=sample_theta, distributions=[uniform_dist])
    assert len(rank_cdf.distributions) == sample_theta.shape[1], "Incorrect number of distributions initialized."
    for dist in rank_cdf.distributions:
        assert dist is uniform_dist, "All distributions should be the single provided distribution."

def test_initialization_multiple_distributions(sample_theta):
    distributions = [Normal(0, 1), Uniform(0, 1), Normal(5, 2)]
    rank_cdf = RankCDF(theta=sample_theta, distributions=distributions)
    assert len(rank_cdf.distributions) == sample_theta.shape[1], "Incorrect number of distributions initialized."
    for i, dist in enumerate(rank_cdf.distributions):
        assert dist is distributions[i], f"Distribution at index {i} does not match."

def test_initialization_invalid_distributions(sample_theta):
    distributions = [Normal(0, 1), Uniform(0, 1)]  # Less than number of latent variables (3)
    with pytest.raises(ValueError, match="The number of distributions must be one or equal to the number of latent variables."):
        RankCDF(theta=sample_theta, distributions=distributions)

def compute_expected_transform(theta, distributions):
    n_samples, n_latent = theta.shape
    transformed = torch.zeros_like(theta, dtype=torch.float32)
    for latent in range(n_latent):
        sorted_vals, _ = torch.sort(theta[:, latent])
        unique_vals, inverse_indices = torch.unique(sorted_vals, return_inverse=True, sorted=True)
        ranks = torch.arange(1, n_samples + 1, dtype=torch.float32)
        rank_sum = torch.zeros_like(unique_vals)
        rank_count = torch.zeros_like(unique_vals)
        rank_sum.scatter_add_(0, inverse_indices, ranks)
        rank_count.scatter_add_(0, inverse_indices, torch.ones_like(ranks))
        avg_ranks = rank_sum / rank_count

        # Map each theta to the closest unique value's average rank
        theta_vals = theta[:, latent]
        closest_indices = torch.argmin(torch.abs(theta_vals.unsqueeze(1) - unique_vals.unsqueeze(0)), dim=1)
        new_ranks = avg_ranks[closest_indices]
        rank_normalized = new_ranks / (n_samples + 1)
        transformed[:, latent] = distributions[latent].icdf(rank_normalized)
    return transformed

def test_transform_default_distributions(sample_theta):
    distributions = [Normal(0, 1) for _ in range(sample_theta.shape[1])]
    rank_cdf = RankCDF(theta=sample_theta)
    transformed = rank_cdf.transform(sample_theta)
    expected = compute_expected_transform(sample_theta, distributions)
    assert transformed.shape == sample_theta.shape, "Transformed tensor has incorrect shape."
    assert transformed.dtype == torch.float32, "Transformed tensor has incorrect dtype."
    assert torch.allclose(transformed, expected, atol=1e-5), "Transformed values do not match expected."

def test_transform_with_ties():
    # Create theta with ties
    theta_with_ties = torch.tensor([
        [1.0, 2.0, 3.0],
        [1.0, 3.0, 4.0],
        [2.0, 3.0, 5.0],
        [2.0, 5.0, 6.0],
        [3.0, 6.0, 7.0],
        [3.0, 7.0, 8.0],
        [4.0, 8.0, 9.0],
        [4.0, 9.0, 10.0],
        [5.0, 10.0, 11.0],
        [5.0, 11.0, 12.0],
    ], dtype=torch.float32)
    distributions = [Normal(0, 1), Normal(0, 1), Normal(0, 1)]
    rank_cdf = RankCDF(theta=theta_with_ties, distributions=distributions)
    transformed = rank_cdf.transform(theta_with_ties)
    expected = compute_expected_transform(theta_with_ties, distributions)
    assert torch.allclose(transformed, expected, atol=1e-5), "Transformed values with ties do not match expected."

def test_gradients_not_implemented(sample_theta):
    rank_cdf = RankCDF(theta=sample_theta)
    with pytest.raises(NotImplementedError, match="Gradients are not available for RankCDF scale transformations."):
        rank_cdf.gradients(sample_theta)
