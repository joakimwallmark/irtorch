import pytest
import torch
from irtorch.gaussian_mixture_model import GaussianMixtureModel


def test_gaussian_mixture_model():
    gmm = GaussianMixtureModel(n_components=2, n_features=1)
    torch.manual_seed(0)
    data = torch.cat(
        [
            torch.normal(mean=-3, std=1, size=(1000, 1)),
            torch.normal(mean=3, std=1, size=(1000, 1)),
        ]
    )

    gmm.fit(data)
    print(gmm.means)
    assert torch.isclose(gmm.weights, torch.tensor([0.5, 0.5]), atol=0.1).all()
    assert all([any(torch.isclose(gmm.means, torch.tensor([[value]]), atol=0.1).view(-1)) for value in [3., -3.]])
    assert torch.isclose(gmm.covariances, torch.tensor([[[1.]], [[1.]]]), atol=0.1).all()

    gmm = GaussianMixtureModel(n_components=2, n_features=2)
    torch.manual_seed(0)
    data = torch.cat(
        [torch.cat([
            torch.normal(mean=-3, std=1, size=(1000, 1)),
            torch.normal(mean=3, std=1, size=(3000, 1)),
        ]),
        torch.cat([
            torch.normal(mean=-2, std=0.5, size=(1000, 1)),
            torch.normal(mean=2, std=0.5, size=(3000, 1)),
        ])],
        dim=1
    )

    gmm.fit(data)
    assert torch.isclose(gmm.weights, torch.tensor([0.75, 0.25]), atol=0.1).all()
    assert torch.isclose(gmm.means, torch.tensor([[3., 2.], [-3., -2.]]), atol=0.1).all()
    assert torch.isclose(
        gmm.covariances,
        torch.tensor([
            [[1., 0.], [0., 0.25]],
            [[1., 0.], [0., 0.25]]
        ]),
        atol=0.1
    ).all()

def test_pdf():
    gmm = GaussianMixtureModel(n_components=2, n_features=2)
    gmm.weights = torch.nn.Parameter(torch.tensor([0.75, 0.25]))
    gmm.means = torch.nn.Parameter(torch.tensor([[0., 0.], [0., 0.]]))
    gmm.covariances = torch.nn.Parameter(torch.tensor([
        [[1., 0.], [0., 1.]],
        [[1., 0.], [0., 1.]]
    ]))
    # Test the pdf method
    new_data = torch.tensor([[0, 0]])  # a single observation
    pdf = gmm.pdf(new_data)
    assert torch.isclose(pdf, torch.tensor([0.15915495]))

    new_data = torch.tensor([[1, 1]])  # a single observation
    pdf2 = gmm.pdf(new_data)
    assert pdf2 < pdf
