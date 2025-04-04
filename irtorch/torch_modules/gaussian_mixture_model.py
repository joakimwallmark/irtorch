import logging
import torch
import torch.distributions as dist

logger = logging.getLogger("irtorch")

class GaussianMixtureModel(torch.nn.Module):
    """
    Gaussian mixture model for approximating a multivariate joint density function.

    Parameters
    ----------
    n_components : int
        The number of mixture components. Must be greater than 0.
    n_features : int
        The number of features in the input data. Must be greater than 0.

    Attributes
    ----------
    weights : torch.nn.Parameter
        The mixing weights for each component. Shape: (n_components,)
    means : torch.nn.Parameter
        The mean vectors for each component. Shape: (n_components, n_features)
    covariances : torch.nn.Parameter
        The covariance matrices for each component. Shape: (n_components, n_features, n_features)

    Examples
    --------
    >>> import torch
    >>> from irtorch.torch_modules import GaussianMixtureModel
    >>> gmm = GaussianMixtureModel(n_components=2, n_features=1)
    >>> data = torch.cat([
    ...     torch.normal(-3, 0.5, size=(1000, 1)),
    ...     torch.normal(3, 0.5, size=(1000, 1))
    ... ])
    >>> gmm.fit(data)
    """

    def __init__(self, n_components, n_features):
        """
        Initialize the Gaussian Mixture Model.

        Parameters
        ----------
        n_components : int
            The number of mixture components. Must be greater than 0.
        n_features : int
            The number of features in the input data. Must be greater than 0.

        Raises
        ------
        ValueError
            If n_components or n_features is less than 1.
        """
        if n_components < 1:
            raise ValueError("Number of components must be at least 1")
        if n_features < 1:
            raise ValueError("Number of features must be at least 1")

        super().__init__()
        self.n_components = n_components
        self.n_features = n_features

        self.weights = torch.nn.Parameter(torch.ones(n_components) / n_components)
        self.means = torch.nn.Parameter(torch.randn(n_components, n_features))
        self.covariances = torch.nn.Parameter(torch.stack([torch.eye(n_features) for _ in range(n_components)]))

    def forward(self, data):
        """
        Calculate the log-likelihood of the data under the current model parameters.

        Parameters
        ----------
        data : torch.Tensor
            Input data tensor of shape (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            The total log-likelihood of the data.
        """
        # Calculate the log probability of each component
        log_probs = torch.stack([dist.MultivariateNormal(self.means[i], covariance_matrix=self.covariances[i]).log_prob(data) for i in range(self.n_components)])
        weighted_log_probs = log_probs + torch.log(self.weights[:, None])
        # Log sum exp for numerical stability
        log_sum_exp = torch.logsumexp(weighted_log_probs, dim=0)
        
        log_likelihood = torch.sum(log_sum_exp)
        
        return log_likelihood

    def _e_step(self, data) -> torch.Tensor:
        with torch.no_grad():
            log_probs = torch.stack([dist.MultivariateNormal(self.means[i], covariance_matrix=self.covariances[i]).log_prob(data) for i in range(self.n_components)])
            weighted_log_probs = log_probs + torch.log(self.weights[:, None])
            log_responsibilities = weighted_log_probs - torch.logsumexp(weighted_log_probs, dim=0, keepdim=True)
            responsibilities = torch.exp(log_responsibilities)
        return responsibilities

    def _m_step(self, data, responsibilities) -> torch.Tensor:
        weights = torch.sum(responsibilities, dim=1) / data.size(0)
        
        weighted_sum = torch.matmul(responsibilities, data)
        means = weighted_sum / torch.sum(responsibilities, dim=1, keepdim=True)
        
        covariances = torch.zeros_like(self.covariances)
        for i in range(self.n_components):
            diff = data - means[i]
            weighted_diff = diff * responsibilities[i, :, None]
            covariances[i] = torch.matmul(weighted_diff.T, diff) / torch.sum(responsibilities[i])

        self.weights.data = weights
        self.means.data = means
        self.covariances.data = covariances

    def fit(self, data, iterations=100):
        """
        Fits the Gaussian mixture model to the given data using the EM algorithm.

        Parameters
        ----------
        data : torch.Tensor
            The tensor of data points to fit the model to. Shape: (n_samples, n_features)
        iterations : int, optional
            The number of iterations to run the EM algorithm, by default 100.

        Raises
        ------
        ValueError
            If data is empty or has wrong number of features.
        """
        if data.shape[0] == 0:
            raise ValueError("Cannot fit GMM on empty data")
        
        if data.shape[1] != self.n_features:
            raise ValueError(f"Expected data with {self.n_features} features, got {data.shape[1]}")

        for _ in range(iterations):
            responsibilities = self._e_step(data)
            self._m_step(data, responsibilities)

    def pdf(self, x):
        """
        Compute the probability density function at the given data points.

        Parameters
        ----------
        x : torch.Tensor
            The tensor of data points at which to evaluate the PDF. Shape: (n_samples, n_features)

        Returns
        -------
        torch.Tensor
            The PDF evaluated at the given data points. Shape: (n_samples,)
        """
        pdf = torch.zeros(x.shape[0], device=x.device)
        for i in range(self.n_components):
            mvn = dist.MultivariateNormal(self.means[i], covariance_matrix=self.covariances[i])
            component_pdf = torch.exp(mvn.log_prob(x))  # Convert log probabilities to probabilities
            pdf += self.weights[i] * component_pdf
        return pdf
