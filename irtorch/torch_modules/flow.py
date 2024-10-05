import torch.nn as nn
from torch.distributions.distribution import Distribution
from irtorch.torch_modules import RationalQuadraticSpline

class RationalQuadraticSplineFlow(nn.Module):
    def __init__(self, transformation: RationalQuadraticSpline, distribution: Distribution):
        """
        Base class for flow objects.

        Parameters
        ----------
        transform: RationalQuadraticSpline
            Transforms values from the observed data to the chosen distribution.
        distribution: Distribution 
            The base distribution of the flow that generates the observed data.
        """
        super().__init__()
        self._transformation = transformation
        self._distribution = distribution

    def log_prob(self, value):
        """
        The log-likelihood of the observed data in the chosen distribution after transformation. Usually the negative loss when training.

        Parameters
        ----------
        value: torch.Tensor
            A tensor of shape [batch_size, ...], the data to be transformed.

        Returns
        -------
        torch.Tensor
            A tensor of shape [batch_size, ...].
        """
        noise, logabsdet = self._transformation(value, inverse=False)
        log_prob = self._distribution.log_prob(noise)
        return log_prob + logabsdet

    def _sample_obs(self, num_samples):
        noise = self._distribution.sample(num_samples)
        samples, _ = self._transformation(noise, inverse=True)
        return samples

    def forward(self, inputs):
        """
        Transforms from observed data into the chosen distribution (noise).

        Parameters
        ----------
        inputs: torch.Tensor
            A tensor of shape [batch_size, ...], the data to be transformed.

        Returns
        -------
        torch.Tensor
            A tensor of shape [batch_size, ...].
        """
        noise, _ = self._transformation(inputs, inverse=False)
        return noise

    def inverse(self, inputs):
        """
        Transforms from the chosen distribution (noise) into observed data.

        Parameters
        ----------
        inputs: torch.Tensor
            A tensor of shape [batch_size, ...], the data to be transformed.

        Returns
        -------
        torch.Tensor
            A tensor of shape [batch_size, ...].
        """
        noise, _ = self._transformation(inputs, inverse=True)
        return noise
