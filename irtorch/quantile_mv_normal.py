import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from irtorch.latent_variable_functions import (
    quantile_transform,
    interp,
)


class QuantileMVNormal:
    """
    Quantile multivariate normal approximation of a multivariate joint density function.

    Attributes
    ----------
    data : torch.Tensor
        original data tensor used for fitting the QuantileMVNorm
    qt_data : torch.Tensor
        quantile transformed data
    mvnorm : torch.distributions.MultivariateNormal
        fitted multivariate normal distribution

    Methods
    -------
    fit_multivariate_normal(data)
        Fits a multivariate normal distribution to the data tensor.
    pdf(data)
        Computes the probability density function at the given data points.
    cdf(data)
        Computes the cumulative distribution function at the given data points.
    """

    def __init__(self):
        """
        Initializes a quantile multivariate normal distribution.
        """
        self.data = None
        self.qt_data = None
        self.mvnormal = None

    def fit(self, data: torch.Tensor):
        """
        Fits the quantile multivariate normal distribution using the provided data tensor.

        Parameters
        ----------
        data : torch.Tensor
            2D data tensor used for fitting QuantileMVNorm. Columns are variables.
        """
        self.data = data
        self.qt_data = quantile_transform(data)
        self.mvnormal = self.fit_multivariate_normal(self.qt_data)

    def fit_multivariate_normal(self, data: torch.Tensor):
        """
        Fits a multivariate normal distribution to the given data.

        Parameters
        ----------
        data : torch.tensor
            A 2D tensor with shape [num_samples, num_variables]. Each row is a sample,
            each column is a variable.

        Returns
        -------
        torch.distributions.MultivariateNormal
            The fitted multivariate normal distribution.

        Examples
        --------
        >>> data = torch.randn(1000, 2)
        >>> dist = fit_multivariate_normal(data)
        >>> samples = dist.sample((5,))  # Draw 5 samples
        >>> log_prob = dist.log_prob(data)  # Compute log probability of data
        """
        mean = data.mean(dim=0)
        data_centered = data - mean
        cov = data_centered.t().mm(data_centered) / (data.shape[0] - 1)
        return MultivariateNormal(mean, cov)

    def pdf(self, data: torch.tensor):
        """
        Computes the probability density function at the given data points.

        Parameters
        ----------
        data : torch.Tensor
            The tensor of data points at which to evaluate the PDF.

        Returns
        -------
        torch.Tensor
            The PDF evaluated at the given data points.
        """
        mv_norm_data = torch.zeros_like(data)
        for i in range(data.shape[1]):
            mv_norm_data[:, i] = interp(data[:, i], self.data[:, i], self.qt_data[:, i])

        return torch.exp(self.mvnormal.log_prob(mv_norm_data))

    def cdf(self, data: torch.tensor):
        """
        Computes the cumulative distribution function at the given data points.

        Parameters
        ----------
        data : torch.Tensor
            The tensor of data points at which to evaluate the CDF.

        Returns
        -------
        torch.Tensor
            The CDF evaluated at the given data points.
        """
        raise NotImplementedError(
            "Not yet implemented as pytorch MultivariateNormal does not support cdf"
        )
        # mv_norm_data = torch.zeros_like(data)
        # for i in range(data.shape[1]):
        #     mv_norm_data[:, i] = interp(data[:, i], self.data[:, i], self.qt_data[:, i])

        # return torch.exp(self.mvnormal.cdf(mv_norm_data))
