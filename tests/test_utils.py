import pytest
import torch
from irtorch.utils import get_item_categories, one_hot_encode_test_data, decode_one_hot_test_data, split_data

def test_get_item_categories(test_data, item_categories):
    result = get_item_categories(test_data)
    assert result == item_categories

def test_one_hot_encode_test_data(device):
    # Define a small tensor of test scores and a list of maximum scores
    #
    scores = (
        torch.tensor([[0, 1, 2], [1, 2, 3], [2, 0, float("nan")], [2, 0, -1]])
        .float()
        .to(device)
    )
    item_categories = [3, 4, 4]

    # Call the function to get the one-hot encoded tensor, with encode_missing set to True
    one_hot_scores = one_hot_encode_test_data(
        scores, item_categories, encode_missing=True
    )

    expected = (
        torch.tensor(
            [
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )
        .float()
        .to(device)
    )

    assert torch.all(one_hot_scores == expected)

    # Call the function to get the one-hot encoded tensor
    one_hot_scores = one_hot_encode_test_data(
        scores, item_categories, encode_missing=False
    )

    expected = torch.tensor(
        [
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
        ]
    ).to(device)

    assert torch.all(one_hot_scores == expected)
    assert one_hot_scores.shape == (4, sum(item_categories))
    assert one_hot_scores.dtype == torch.float32
    with pytest.raises(ValueError):
        one_hot_encode_test_data(
            torch.tensor([0, 1, 2]), item_categories, encode_missing=False
        )
    with pytest.raises(ValueError):
        one_hot_encode_test_data(scores, [2, 2], encode_missing=False)

def test_decode_one_hot_test_data(device):
    scores = (
        torch.tensor([[0, 1, 2], [1, 2, 3], [2, 0, 1], [2, 0, 1]]).float().to(device)
    ).float()
    item_categories = [3, 4, 4]

    one_hot_scores = one_hot_encode_test_data(
        scores, item_categories, encode_missing=False
    )

    decoded_scores = decode_one_hot_test_data(one_hot_scores, item_categories)

    # Check that the decoded scores match the original scores
    assert torch.all(decoded_scores == scores)

    with pytest.raises(ValueError):
        decode_one_hot_test_data(torch.tensor([0, 1, 2]), item_categories)
    with pytest.raises(ValueError):
        decode_one_hot_test_data(one_hot_scores, [2, 2])

    # test with
    scores = (
        torch.tensor([[0, -1, 2], [1, 2, 3], [2, 0, 1], [2, -1, 1]]).float().to(device)
    ).float()
    item_categories = [3, 4, 4]

    # Call the function to get the one-hot encoded tensor
    one_hot_scores = one_hot_encode_test_data(
        scores, item_categories, encode_missing=True
    )

    # Call the function to decode the one-hot encoded tensor
    decoded_scores = decode_one_hot_test_data(
        one_hot_scores, [item_cat + 1 for item_cat in item_categories]
    )

    # Check that the decoded scores match the original scores
    assert torch.all(decoded_scores == scores + 1)

def test_split_data():
    torch.manual_seed(42)
    data = torch.rand(100, 10)

    # Test with shuffle=True
    train_data, test_data = split_data(data, train_ratio=0.8, shuffle=True)
    assert train_data.shape[0] == 80  # 80% of 100 samples
    assert test_data.shape[0] == 20  # 20% of 100 samples
    assert not torch.equal(data[:80], train_data)  # Data should be shuffled

    # Test with shuffle=False
    train_data, test_data = split_data(data, train_ratio=0.8, shuffle=False)
    assert train_data.shape[0] == 80
    assert test_data.shape[0] == 20
    assert torch.equal(data[:80], train_data)  # Data should not be shuffled

    # Test with different train_ratio
    train_data, test_data = split_data(data, train_ratio=0.6, shuffle=False)
    assert train_data.shape[0] == 60
    assert test_data.shape[0] == 40

    # Test with train_ratio=1.0
    train_data, test_data = split_data(data, train_ratio=1.0, shuffle=False)
    assert train_data.shape[0] == 100
    assert test_data.shape[0] == 0

    # Test with train_ratio=0.0
    train_data, test_data = split_data(data, train_ratio=0.0, shuffle=False)
    assert train_data.shape[0] == 0
    assert test_data.shape[0] == 100
