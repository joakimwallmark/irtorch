import torch
import pytest
from irtorch.gaussian_mixture_torch import GaussianMixtureTorch


def test_gaussian_mixture_torch():
    # Create an instance of GaussianMixtureTorch
    gmm = GaussianMixtureTorch()

    # Generate some example data
    data = torch.cat(
        [
            torch.normal(mean=-3, std=1, size=(100, 1)),
            torch.normal(mean=3, std=1, size=(100, 1)),
        ]
    )

    # Test the fit method
    gmm.fit(data)
    assert gmm.n_components is not None
    assert gmm.gmm is not None

    # Test the pdf method
    new_data = torch.tensor([[2.5]])  # a single observation
    density = gmm.pdf(new_data)
    assert torch.is_tensor(density)
    assert density is not None
    assert density > 0

    # Test the error handling in the pdf method
    gmm_empty = GaussianMixtureTorch()
    with pytest.raises(AttributeError):
        gmm_empty.pdf(new_data)
