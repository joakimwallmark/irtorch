import pytest
import torch
from irtorch.load_dataset import swedish_national_mathematics_2019, swedish_national_mathematics_2018, swedish_sat_verbal_2022, swedish_sat_quantitative_2022, swedish_sat_2022, swedish_sat_2022_binary, big_five

def test_swedish_national_mathematics_2019():
    data = swedish_national_mathematics_2019()
    assert data.dtype == torch.float32
    assert data.shape == (1401, 28)

def test_swedish_national_mathematics_2018():
    data = swedish_national_mathematics_2018()
    assert data.dtype == torch.float32
    assert data.shape == (1008, 28)

def test_swedish_sat_verbal_2022():
    data, correct_category = swedish_sat_verbal_2022()
    assert data.dtype == torch.float32
    assert data.shape == (38068, 80)
    assert len(correct_category) == 80
    assert all([isinstance(x, int) for x in correct_category])

def test_swedish_sat_quantitative_2022():
    data, correct_category = swedish_sat_quantitative_2022()
    assert data.dtype == torch.float32
    assert data.shape == (38068, 80)
    assert len(correct_category) == 80
    assert all([isinstance(x, int) for x in correct_category])

def test_swedish_sat_2022():
    data, correct_category = swedish_sat_2022()
    assert data.dtype == torch.float32
    assert data.shape == (38068, 160)
    assert len(correct_category) == 160
    assert all([isinstance(x, int) for x in correct_category])

def test_swedish_sat_2022_binary():
    data = swedish_sat_2022_binary()
    assert data.dtype == torch.float32
    assert data.shape == (38068, 160)
    assert data[~data.isnan()].max() == 1.0
    assert data[~data.isnan()].min() == 0.0

def test_big_five():
    data = big_five()
    assert data.dtype == torch.float32
    assert data.shape == (19719, 50)
    assert data[~data.isnan()].max() == 4.0
    assert data[~data.isnan()].min() == 0.0
