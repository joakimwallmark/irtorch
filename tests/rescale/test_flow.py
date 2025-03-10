import pytest
import torch
from irtorch.rescale import Flow

def test_fit_and_inverse():
    torch.manual_seed(42)
    original_thetas = torch.rand(1000, 5) * 20 - 5

    flow = Flow(latent_variables=5)
    flow.fit(
        theta=original_thetas,
        batch_size=128,
        learning_rate_updates_before_stopping=2,
        learning_rate=0.01,
        evaluation_interval_size=20
    )
    transformed_thetas = flow(original_thetas)
    inverse_thetas = flow.inverse(transformed_thetas)

    assert torch.all(torch.isclose(transformed_thetas.mean(dim=0), torch.tensor(0.0), atol=0.3)), f"Means are off: {transformed_thetas.mean(dim=0)}"
    assert torch.all(torch.isclose(transformed_thetas.var(dim=0), torch.tensor(1.0), atol=0.4)), f"Variances are off: {transformed_thetas.var(dim=0)}"
    assert torch.allclose(inverse_thetas, original_thetas, atol=0.001), "Original thetas are off"

def test_gradients():
    flow = Flow(latent_variables=2)
    thetas = torch.randn(5, 2)
    flow.fit(theta=thetas, max_epochs=1)
    gradients = flow.jacobian(thetas)
    assert gradients.size() == torch.Size([5, 2, 2]), f"Gradients size is off: {gradients.size()}"
    assert torch.allclose(gradients[:, 1, 0], torch.tensor(0.0)) and torch.allclose(gradients[:, 0, 1], torch.tensor(0.0)), f"Off diagonals are non-zero: {gradients}"
