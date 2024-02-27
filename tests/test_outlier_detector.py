import pytest
import torch
from irtorch.outlier_detector import OutlierDetector 

def test_identify_outliers():
    detector = OutlierDetector()
    data = torch.tensor([[1.0, 2.0], [2.0, 50.0], [3.0, 4.0], [1.0, 2.0], [3.0, -3.0], [4.0, 4.0]])

    # Expected outliers are computed manually for this specific set of data.
    expected_outliers_all = torch.tensor([[False, False], [False, True], [False, False], [False, False], [False, True], [False, False]])
    expected_outliers_upper = torch.tensor([[False, False], [False, True], [False, False], [False, False], [False, False], [False, False]])
    expected_outliers_lower = torch.tensor([[False, False], [False, False], [False, False], [False, False], [False, True], [False, False]])

    outliers_all = detector.identify_outliers(data)
    assert torch.equal(outliers_all, expected_outliers_all)

    outliers_upper = detector.identify_outliers(data, upper=True, lower=False)
    assert torch.equal(outliers_upper, expected_outliers_upper)

    outliers_lower = detector.identify_outliers(data, upper=False, lower=True)
    assert torch.equal(outliers_lower, expected_outliers_lower)

    with pytest.raises(ValueError):
        detector.identify_outliers(torch.rand(5))  # not a 2D tensor

    with pytest.raises(ValueError):
        detector.identify_outliers(data, upper=False, lower=False)  # both flags should not be False


def test_is_outlier():
    detector = OutlierDetector()
    data = torch.tensor([[1.0, 2.0], [2.0, 5.0], [3.0, -3.0], [4.0, 4.0]])
    new_observation = torch.tensor([[2.5, -10.0]])

    result = detector.is_outlier(new_observation, data)
    assert torch.equal(result, torch.tensor([[False, True]]))

    with pytest.raises(ValueError):
        detector.is_outlier(torch.rand(2, 1), data)  # new_observation is not 1D

    with pytest.raises(ValueError):
        detector.is_outlier(new_observation, torch.rand(5))  # data is not 2D

    with pytest.raises(ValueError):
        detector.is_outlier(torch.tensor([[2.5, 1.0, 0.5]]), data)  # new_observation's length doesn't match data's columns

    with pytest.raises(ValueError):
        detector.is_outlier(new_observation, data, upper=False, lower=False)  # both flags should not be False


def test_smallest_largest_non_outlier():
    detector = OutlierDetector()
    data = torch.tensor([[-1000.0, 1.0], [2.0, 5.0], [1.0, 5.0], [3.0, 5.0], [-1.0, 5.0], [3.0, -3.0], [4.0, 4.0]])

    result = detector.smallest_largest_non_outlier(data)
    assert torch.equal(result, torch.tensor([[-1.0, 1.0]]))

    result = detector.smallest_largest_non_outlier(data, smallest=False)
    assert torch.equal(result, torch.tensor([[4.0, 5.0]]))

    with pytest.raises(ValueError):
        detector.smallest_largest_non_outlier(torch.rand(5))  # not a 2D tensor