import pytest
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from irtorch.quantile_mv_normal import QuantileMVNormal


def test_fit():
    data = torch.randn(1000, 2)

    qmvn = QuantileMVNormal()
    qmvn.fit(data)

    assert isinstance(qmvn.mvnormal, torch.distributions.MultivariateNormal)
    assert qmvn.mvnormal.batch_shape == torch.Size([])
    assert qmvn.mvnormal.event_shape == torch.Size([2])


def test_fit_multivariate_normal():
    # Create a symmetric matrix
    symmetric_matrix = torch.tensor([[1.0, 0.2, 0.3], [0.2, 1.2, 0.4], [0.3, 0.4, 0.9]])
    # Make a Cholesky decomposition
    cho_dec = torch.linalg.cholesky(symmetric_matrix)
    # Create a positive-definite covariance matrix
    cov = cho_dec.mm(cho_dec.t())
    data_dist = MultivariateNormal(torch.tensor([2.0, 3.0, 4.0]), cov)
    data_tensor = data_dist.sample((2000, ))
    mv_norm = QuantileMVNormal().fit_multivariate_normal(data_tensor)
    # Check that the returned object is a MultivariateNormal
    assert isinstance(mv_norm, MultivariateNormal)

    # Compute the mean and covariance of the data
    mean = data_tensor.mean(dim=0)
    data_centered = data_tensor - mean
    cov = data_centered.t().mm(data_centered) / (data_tensor.shape[0] - 1)

    # Check that the mean and covariance of the distribution match the mean and covariance of the data
    assert torch.allclose(mv_norm.mean, mean)
    assert torch.allclose(mv_norm.covariance_matrix, cov)


def test_pdf():
    # Create a symmetric matrix
    symmetric_matrix = torch.tensor([[1.0, 0.2], [0.2, 1.0]])
    # Make a Cholesky decomposition
    cho_dec = torch.linalg.cholesky(symmetric_matrix)
    # Create a positive-definite covariance matrix
    cov = cho_dec.mm(cho_dec.t())
    data_dist = MultivariateNormal(torch.tensor([0.0, 0.0]), cov)

    data = torch.tensor([[1.0, 3.0], [2.0, 2.0], [4.0, 6.0], [5.0, 5.0]])
    qmvn = QuantileMVNormal()
    qmvn.fit(data)
    qmvn.mvnormal = data_dist

    pdf = qmvn.pdf(torch.tensor([[3.0, 4.0], [3.1, 4.1], [6, 8]]))
    assert pdf.shape == torch.Size([3])
    # center should be largest
    assert qmvn.pdf(torch.tensor([[3.0, 4.0]])) > qmvn.pdf(
        torch.tensor([[3.1, 4.1]])
    )
    assert qmvn.pdf(torch.tensor([[3.0, 4.0]])) > qmvn.pdf(
        torch.tensor([[2.9, 3.9]])
    )


# def test_cdf():
#     # Create a symmetric matrix
#     symmetric_matrix = torch.tensor([[1.0, 0.2], [0.2, 1.0]])
#     # Make a Cholesky decomposition
#     cho_dec = torch.linalg.cholesky(symmetric_matrix)
#     # Create a positive-definite covariance matrix
#     cov = cho_dec.mm(cho_dec.t())
#     data_dist = MultivariateNormal(torch.tensor([0.0, 0.0]), cov)

#     data = torch.tensor([[1.0, 3.0], [2.0, 2.0], [4.0, 6.0], [5.0, 5.0]])
#     qmvn = QuantileMVNormal(data)
#     qmvn.mvnormal = data_dist

#     pdf = qmvn.cdf(torch.tensor([[3.0, 4.0], [3.1, 4.1], [6, 8]]))
#     assert pdf.shape == torch.Size([3])
#     # smaller should be smaller
#     assert qmvn.cdf(torch.tensor([[3.0, 4.0]])) < qmvn.cdf(torch.tensor([[3.1, 4.1]]))
#     assert qmvn.cdf(torch.tensor([[3.0, 4.0]])) > qmvn.cdf(torch.tensor([[2.9, 3.9]]))
