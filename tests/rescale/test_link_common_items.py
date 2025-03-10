import pytest
import torch
import torch.nn as nn
from irtorch.rescale import LinkCommonItems
from irtorch.models import BaseIRTModel

class DummyIRTModel(BaseIRTModel):
    def __init__(self, latent_variables=1):
        super().__init__(latent_variables, [2] * 10)
        self.bias_param = nn.Parameter(torch.randn(self.items))

    def forward(self, theta):
        return theta

    def item_probabilities(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            3D tensor with dimensions (respondents, items, item categories)
        """
        output = theta + self.bias_param
        probs = torch.sigmoid(output)
        probs = probs.view(-1, self.items, 1)
        return torch.cat((1-probs, probs), 2)

def test_fit_and_inverse():
    torch.manual_seed(42)
    model1 = DummyIRTModel(latent_variables=1)
    model2 = DummyIRTModel(latent_variables=1)
    original_thetas = torch.rand(1000, 1)

    # spline linking
    link = LinkCommonItems(model2, model1, range(4), range(4))
    link.fit(
        theta_from=original_thetas,
        batch_size=128,
        learning_rate_updates_before_stopping=2,
        learning_rate=0.01,
        evaluation_interval_size=20,
        max_epochs=10
    )

    assert torch.all(link._transformation.state_dict()['unnormalized_widths'] != 0.0), "Widths are not zero"
    assert torch.all(link._transformation.state_dict()['unnormalized_heights'] != 0.0), "Heights are not zero"
    assert torch.all(link._transformation.state_dict()['lower_output_bound'] != -5.5), "Lower bound did not change"
    assert torch.all(link._transformation.state_dict()['upper_output_bound'] != 5.5), "Upper bound did not change"
    transformed_thetas = link(original_thetas)
    inverse_thetas = link.inverse(transformed_thetas)

    assert torch.allclose(inverse_thetas, original_thetas, atol=0.001), "Original thetas are off"

    # neural network linking
    link = LinkCommonItems(model2, model1, range(4), range(4), method="neuralnet", neurons = 6)
    link.fit(
        theta_from=original_thetas,
        batch_size=128,
        learning_rate_updates_before_stopping=2,
        learning_rate=0.01,
        evaluation_interval_size=20,
        max_epochs=10
    )
    assert torch.all(link._transformation.state_dict()['raw_weight_param'] != 0.0), "Widths are not zero"
    assert torch.all(link._transformation.state_dict()['bias_param'] != 0.0), "Heights are not zero"
    transformed_thetas = link(original_thetas)

    with pytest.raises(NotImplementedError):
        inverse_thetas = link.inverse(transformed_thetas)


def test_gradients():
    torch.manual_seed(42)
    model1 = DummyIRTModel(latent_variables=1)
    model2 = DummyIRTModel(latent_variables=1)
    link = LinkCommonItems(model2, model1, range(4), range(4))
    thetas = torch.randn(5, 1)
    link.fit(theta_from=thetas, max_epochs=1)
    gradients = link.jacobian(thetas)
    assert gradients.size() == torch.Size([5, 1, 1]), f"Gradients size is off: {gradients.size()}"

    torch.manual_seed(42)
    model1 = DummyIRTModel(latent_variables=1)
    model2 = DummyIRTModel(latent_variables=1)
    link = LinkCommonItems(model2, model1, range(4), range(4), method="neuralnet", neurons = 6)
    link.fit(theta_from=thetas, max_epochs=1)
    gradients = link.jacobian(thetas)
    assert gradients.size() == torch.Size([5, 1, 1]), f"Gradients size is off: {gradients.size()}"
