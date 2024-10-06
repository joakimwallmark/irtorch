import pytest
import torch
from irtorch.rescale import Flow
from irtorch.models import BaseIRTModel

def test_fit_and_inverse(
    mml_1d_gpc_natmat_thetas: torch.Tensor,
    ae_1d_mmc_swesat_thetas: torch.Tensor,
    vae_5d_graded_big_five_thetas: torch.Tensor
):
    torch.manual_seed(42)
    flow = Flow(latent_variables=5)
    flow.fit(theta=vae_5d_graded_big_five_thetas, learning_rate_updates_before_stopping=1, evaluation_interval_size=10)
    thetas1 = flow(vae_5d_graded_big_five_thetas)
    original_thetas1 = flow.inverse(thetas1)

    flow = Flow(latent_variables=1)
    flow.fit(theta=ae_1d_mmc_swesat_thetas, learning_rate_updates_before_stopping=1, evaluation_interval_size=10)
    thetas2 = flow(ae_1d_mmc_swesat_thetas)
    original_thetas2 = flow.inverse(thetas2)

    flow = Flow(latent_variables=1)
    flow.fit(theta=mml_1d_gpc_natmat_thetas, learning_rate_updates_before_stopping=1, evaluation_interval_size=10)
    thetas3 = flow(mml_1d_gpc_natmat_thetas)
    original_thetas3 = flow.inverse(thetas3)

    means = torch.cat([thetas1.mean(dim=0), thetas2.mean().unsqueeze(0), thetas3.mean().unsqueeze(0)])
    variances = torch.cat([thetas1.var(dim=0), thetas2.var().unsqueeze(0), thetas3.var().unsqueeze(0)])
    assert torch.all(torch.isclose(means, torch.tensor(0.0), atol=0.3)), f"Means are off: {means}"
    assert torch.all(torch.isclose(variances, torch.tensor(1.0), atol=0.4)), f"Variances are off: {variances}"


    assert torch.allclose(vae_5d_graded_big_five_thetas, original_thetas1, atol=0.001), "Original thetas 1 are off"
    assert torch.allclose(ae_1d_mmc_swesat_thetas, original_thetas2, atol=0.001), "Original thetas 2 are off"
    assert torch.allclose(mml_1d_gpc_natmat_thetas, original_thetas3, atol=0.001), "Original thetas 3 are off"

def test_gradients():
    flow = Flow(latent_variables=2)
    thetas = torch.randn(5, 2)
    flow.fit(theta=thetas, max_epochs=1)
    gradients = flow.gradients(thetas)
    assert gradients.size() == torch.Size([5, 2, 2]), f"Gradients size is off: {gradients.size()}"
    assert torch.allclose(gradients[:, 1, 0], torch.tensor(0.0)) and torch.allclose(gradients[:, 0, 1], torch.tensor(0.0)), f"Off diagonals are non-zero: {gradients}"

def _plot_thetas(thetas: torch.Tensor):
    import matplotlib.pyplot as plt
    if thetas.size(1) > 1:
        fig, axes = plt.subplots(thetas.size(1), 1, figsize=(6, 10))  # 4 rows, 1 column for the subplots
        fig.tight_layout(pad=3.0)
        for i in range(thetas.size(1)):
            axes[i].hist(thetas[:, i].detach().numpy(), bins=30, density=True, alpha=0.6, color='b')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Density')
        plt.show()
    else:
        plt.hist(thetas[:, 0].detach().numpy(), bins=30, density=True, alpha=0.6, color='b')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.show()
