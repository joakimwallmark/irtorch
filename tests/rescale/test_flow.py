import pytest
import torch
from irtorch.rescale import Flow
from irtorch.models import BaseIRTModel

def test_fit(
    mml_1d_gpc_natmat_thetas: torch.Tensor,
    ae_1d_mmc_swesat_thetas: torch.Tensor,
    vae_5d_graded_big_five_thetas: torch.Tensor
):
    torch.manual_seed(42)
    flow = Flow(latent_variables=5)
    flow.fit(theta=vae_5d_graded_big_five_thetas, learning_rate_updates_before_stopping=1, evaluation_interval_size=10)
    thetas1 = flow(vae_5d_graded_big_five_thetas)

    flow = Flow(latent_variables=1)
    flow.fit(theta=ae_1d_mmc_swesat_thetas, learning_rate_updates_before_stopping=1, evaluation_interval_size=10)
    thetas2 = flow(ae_1d_mmc_swesat_thetas)
    # original = flow.inverse(thetas)

    flow = Flow(latent_variables=1)
    flow.fit(theta=mml_1d_gpc_natmat_thetas, learning_rate_updates_before_stopping=1, evaluation_interval_size=10)
    thetas3 = flow(mml_1d_gpc_natmat_thetas)

    means = torch.cat([thetas1.mean(dim=0), thetas2.mean().unsqueeze(0), thetas3.mean().unsqueeze(0)])
    variances = torch.cat([thetas1.var(dim=0), thetas2.var().unsqueeze(0), thetas3.var().unsqueeze(0)])
    assert torch.all(torch.isclose(means, torch.tensor(0.0), atol=0.3)), f"Means are off: {means}"
    assert torch.all(torch.isclose(variances, torch.tensor(1.0), atol=0.4)), f"Variances are off: {variances}"

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
