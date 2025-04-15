import pytest
import torch
from irtorch.models import MonotoneBSpline

@pytest.fixture
def sample_data():
    # Create sample data with 100 respondents and 3 items
    # Items have 2, 3, and 4 categories respectively
    data = torch.randint(0, 4, (100, 3)).float()
    data[:, 0].clamp_(max=1)  # 2 categories for first item
    data[:, 1].clamp_(max=2)  # 3 categories for second item
    return data

def test_init_with_data(sample_data):
    model = MonotoneBSpline(data=sample_data)
    assert model.items == 3
    assert model.item_categories == [2, 3, 4]
    assert model.latent_variables == 1  # default value

def test_init_with_explicit_params():
    model = MonotoneBSpline(
        latent_variables=2,
        item_categories=[2, 3, 4],
        knots=[-0.5, 0.0, 0.5],
        degree=3,
        separate="caetegories"
    )
    assert model.items == 3
    assert model.item_categories == [2, 3, 4]
    assert model.latent_variables == 2
    assert len(model.knots) == 11  # 5 internal knots + 2*(degree+1) boundary knots

def test_init_validation():
    # Test missing required parameters
    with pytest.raises(ValueError, match="Either item_categories or data must be provided"):
        MonotoneBSpline()

def test_forward():
    model = MonotoneBSpline(latent_variables=1, item_categories=[2, 3], separate="categories")
    # Create tensor directly with the correct shape (leaf tensor)
    theta = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float32, requires_grad=True)
    output = model(theta)

    assert output.shape == (3, 2, 3)
    loss = output.sum()
    loss.backward()
    assert theta.grad is not None
    
    model = MonotoneBSpline(latent_variables=1, item_categories=[2, 3])
    # Create tensor directly with the correct shape (leaf tensor)
    theta = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float32, requires_grad=True)
    output = model(theta)

    # Check output shape: (batch_size, items, item response categories)
    assert output.shape == (3, 2, 3)

    loss = output.sum()
    loss.backward()
    assert theta.grad is not None


def test_log_likelihood():
    model = MonotoneBSpline(item_categories=[2, 3])
    theta = torch.randn((3, 1))
    output = model(theta)

    # Create sample responses
    responses = torch.tensor([[0, 1], [1, 2], [1, 0]]).float()

    # Test log likelihood computation
    ll = model.log_likelihood(responses, output)
    assert ll.shape == torch.Size([]) # One likelihood per sample
    assert torch.isfinite(ll)

    ll = model.log_likelihood(responses, output, loss_reduction="none")
    assert ll.shape == (6,)  # One likelihood per item response
    assert torch.all(torch.isfinite(ll))

def test_monotonicity():
    model = MonotoneBSpline(item_categories=[2])

    # Test increasing theta leads to increasing probabilities
    theta1 = torch.tensor([[-1.0], [0.0], [1.0]])
    theta2 = torch.tensor([[0.0], [1.0], [2.0]])

    output1 = model(theta1)
    output2 = model(theta2)

    # Higher theta should lead to higher probabilities for higher categories
    probs1 = torch.softmax(output1, dim=2)
    probs2 = torch.softmax(output2, dim=2)

    assert torch.all(probs2[:, :, -1] >= probs1[:, :, -1])

def test_with_missing_data(sample_data):
    # Create data with some missing values
    data_with_missing = sample_data.clone()
    data_with_missing[0, 0] = float('nan')

    model = MonotoneBSpline(data=data_with_missing)
    theta = torch.randn(100, 1)
    output = model(theta)

    # Create missing mask
    missing_mask = torch.isnan(data_with_missing)

    # Test log likelihood with missing data
    ll = model.log_likelihood(data_with_missing, output, missing_mask=missing_mask)
    assert torch.isfinite(ll)

    # Test with loss_reduction="none"
    ll_none = model.log_likelihood(data_with_missing, output, missing_mask=missing_mask, loss_reduction="none")
    assert ll_none.shape == (300,)  # 100 respondents * 3 items
    assert torch.isnan(ll_none[0])  # First value should be NaN (missing)
    assert torch.all(torch.isfinite(ll_none[1:]))  # Rest should be finite
