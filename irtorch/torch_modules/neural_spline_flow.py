import torch
import torch.nn as nn
from torch.distributions import Distribution
from irtorch.torch_modules import RationalQuadraticSpline

class NeuralSplineFlow(nn.Module):
    def __init__(self, transformation: RationalQuadraticSpline, distribution: Distribution):
        """
        Normalizing flow using rational-quadratic splines :cite:p:`Durkan2019`.

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

    def forward(self, inputs) -> torch.Tensor:
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

    def log_prob(self, value) -> torch.Tensor:
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

    def _sample_obs(self, num_samples) -> torch.Tensor:
        noise = self._distribution.sample(num_samples)
        samples, _ = self._transformation(noise, inverse=True)
        return samples

    def inverse(self, inputs) -> torch.Tensor:
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
