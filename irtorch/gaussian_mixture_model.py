import logging
import torch
import torch.distributions as dist

logger = logging.getLogger('irtorch')

class GaussianMixtureModel(torch.nn.Module):
    """
    Gaussian mixture model for approximating a multivariate joint density function.

    Parameters
    ----------
    n_components : int, optional
        The number of components in the mixture model. If None, the number of components is selected using 5 fold cross validation.
    """
    def __init__(self, n_components, n_features):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features

        self.weights = torch.nn.Parameter(torch.ones(n_components) / n_components)
        self.means = torch.nn.Parameter(torch.randn(n_components, n_features))
        self.covariances = torch.nn.Parameter(torch.stack([torch.eye(n_features) for _ in range(n_components)]))

    def forward(self, data):
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
            The tensor of data points to fit the model to.
        iterations : int, optional
            The number of iterations to run the EM algorithm. The default is 100.
        """
        for _ in range(iterations):
            responsibilities = self._e_step(data)
            self._m_step(data, responsibilities)

    def pdf(self, x) -> torch.Tensor:
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
        pdf = torch.zeros(x.shape[0], device=x.device)
        for i in range(self.n_components):
            mvn = dist.MultivariateNormal(self.means[i], covariance_matrix=self.covariances[i])
            component_pdf = torch.exp(mvn.log_prob(x)) # Convert log probabilities to probabilities
            pdf += self.weights[i] * component_pdf
        return pdf
