import pytest
import torch
from irtorch.load_dataset import swedish_national_mathematics_2, swedish_national_mathematics_1, swedish_sat_verbal, swedish_sat_quantitative, swedish_sat, swedish_sat_binary, big_five

def test_swedish_national_mathematics_2():
    data = swedish_national_mathematics_2()
    assert data.dtype == torch.float32
    assert data.shape == (1401, 28)

def test_swedish_national_mathematics_1():
    data = swedish_national_mathematics_1()
    assert data.dtype == torch.float32
    assert data.shape == (1008, 28)

def test_swedish_sat_verbal():
    data, correct_category = swedish_sat_verbal()
    assert data.dtype == torch.float32
    assert data.shape == (5000, 80)
    assert len(correct_category) == 80
    assert all([isinstance(x, int) for x in correct_category])

def test_swedish_sat_quantitative():
    data, correct_category = swedish_sat_quantitative()
    assert data.dtype == torch.float32
    assert data.shape == (5000, 80)
    assert len(correct_category) == 80
    assert all([isinstance(x, int) for x in correct_category])

def test_swedish_sat():
    data, correct_category = swedish_sat()
    assert data.dtype == torch.float32
    assert data.shape == (5000, 160)
    assert len(correct_category) == 160
    assert all([isinstance(x, int) for x in correct_category])

def test_swedish_sat_binary():
    data = swedish_sat_binary()
    assert data.dtype == torch.float32
    assert data.shape == (5000, 160)
    assert data[~data.isnan()].max() == 1.0
    assert data[~data.isnan()].min() == 0.0

def test_big_five():
    data, df, items = big_five()
    assert data.dtype == torch.float32
    assert data.shape == (667701, 50)
    assert data[~data.isnan()].max() == 4.0
    assert data[~data.isnan()].min() == 0.0
    assert len(df) == 1015341
    assert len(df.columns) == 110
    assert len(items) == 50
