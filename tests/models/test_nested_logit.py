import pytest
import torch
from irtorch.models import NestedLogit, TwoParameterLogistic

@pytest.fixture
def sample_data():
    # Create sample data with 100 respondents and 3 items
    # Items have 4, 3, and 5 categories respectively
    data = torch.randint(0, 5, (100, 3)).float()
    data[:, 1].clamp_(max=2)  # 3 categories for second item
    data[:, 0].clamp_(max=3)  # 4 categories for first item
    return data

@pytest.fixture
def correct_response_model():
    model = TwoParameterLogistic(latent_variables=2, items=3)
    return model

def test_init_with_nominal():
    model = NestedLogit(
        mc_correct=[1, 0, 2],
        correct_response_model=TwoParameterLogistic(latent_variables=2, items=3),
        incorrect_response_model="nominal",
        item_categories=[4, 3, 5],
        latent_variables=2
    )
    assert model.items == 3
    assert model.item_categories == [4, 3, 5]
    assert model.latent_variables == 2
    assert model.mc_correct == [1, 0, 2]

def test_init_with_bspline():
    model = NestedLogit(
        mc_correct=[1, 0, 2],
        correct_response_model=TwoParameterLogistic(latent_variables=2, items=3),
        incorrect_response_model="bspline",
        item_categories=[4, 3, 5],
        latent_variables=2,
        degree=3,
        knots=[-1.0, 0.0, 1.0]
    )
    assert model.incorrect_response_model == "bspline"
    assert model.degree == 3
    assert len(model.knots) == 11  # degree+1 + len(knots) + degree+1

def test_init_invalid_response_model():
    with pytest.raises(ValueError, match="Incorrect response model must be 'nominal' or 'bspline'"):
        NestedLogit(
            mc_correct=[1, 0, 2],
            correct_response_model=TwoParameterLogistic(latent_variables=2, items=3),
            incorrect_response_model="invalid",
            item_categories=[4, 3, 5]
        )

def test_dichotomize_data():
    model = NestedLogit(
        mc_correct=[1, 0, 2],
        correct_response_model=TwoParameterLogistic(latent_variables=2, items=3),
        incorrect_response_model="nominal",
        item_categories=[4, 3, 5]
    )
    data = torch.tensor([
        [1, 0, 2],  # all correct
        [0, 1, 1],  # all incorrect
        [1, 0, 0]   # mixed
    ]).float()
    
    dichotomized = model._dichotomize_data(data)
    expected = torch.tensor([
        [1., 1., 1.],
        [0., 0., 0.],
        [1., 1., 0.]
    ])
    assert torch.equal(dichotomized, expected)

def test_forward_nominal():
    model = NestedLogit(
        mc_correct=[1, 0, 2],
        correct_response_model=TwoParameterLogistic(latent_variables=2, items=3),
        incorrect_response_model="nominal",
        item_categories=[4, 3, 5],
        latent_variables=2
    )
    theta = torch.randn(10, 2)  # 10 respondents, 2 latent variables
    output = model(theta)
    
    assert output.shape == (10, 3, 5)  # (respondents, items, max_categories)
    assert torch.allclose(output.sum(dim=2), torch.ones_like(output.sum(dim=2)))  # probabilities sum to 1

def test_forward_bspline():
    model = NestedLogit(
        mc_correct=[1, 0, 2],
        correct_response_model=TwoParameterLogistic(latent_variables=2, items=3),
        incorrect_response_model="bspline",
        item_categories=[4, 3, 5],
        latent_variables=2
    )
    theta = torch.randn(10, 2)
    output = model(theta)
    
    assert output.shape == (10, 3, 5)
    assert torch.allclose(output.sum(dim=2), torch.ones_like(output.sum(dim=2)))

def test_log_likelihood():
    model = NestedLogit(
        mc_correct=[1, 0, 2],
        correct_response_model=TwoParameterLogistic(latent_variables=2, items=3),
        incorrect_response_model="nominal",
        item_categories=[4, 3, 5]
    )
    
    # Create sample data and output
    data = torch.tensor([[1, 0, 2], [0, 1, 1]]).long()
    output = torch.ones(2, 3, 5) / 5  # Equal probabilities for simplicity
    
    # Test sum reduction
    ll_sum = model.log_likelihood(data, output, loss_reduction="sum")
    assert ll_sum.dim() == 0  # scalar
    
    # Test no reduction
    ll_none = model.log_likelihood(data, output, loss_reduction="none")
    assert ll_none.shape == (6,)  # One value per respondent and item

def test_log_likelihood_with_missing():
    model = NestedLogit(
        mc_correct=[1, 0, 2],
        correct_response_model=TwoParameterLogistic(latent_variables=2, items=3),
        incorrect_response_model="nominal",
        item_categories=[4, 3, 5]
    )
    
    data = torch.tensor([[1, 0, 2], [0, 1, 1]]).long()
    output = torch.ones(2, 3, 5) / 5
    missing_mask = torch.tensor([[False, True, False], [False, False, True]])
    
    ll = model.log_likelihood(data, output, missing_mask=missing_mask, loss_reduction="none")
    assert torch.isnan(ll).any()  # Should contain nan values for missing responses

def test_item_theta_relationship_directions():
    correct_model = TwoParameterLogistic(latent_variables=2, items=3)
    model = NestedLogit(
        mc_correct=[1, 0, 2],
        correct_response_model=correct_model,
        incorrect_response_model="nominal",
        item_categories=[4, 3, 5]
    )
    
    theta = torch.randn(10, 2)
    directions = model.item_theta_relationship_directions(theta)
    assert directions.shape == (3, 2)  # (items, latent_variables)