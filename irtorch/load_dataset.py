from importlib import resources
import pandas as pd
import torch
import feather

def swedish_national_mathematics_2019():
    """
    Load the Swedish National Mathematics 2019 dataset.

    Returns
    -------
    torch.Tensor
        The loaded dataset.
        
    Raises
    ------
    RuntimeError
        If the dataset cannot be loaded.
    """
    try:
        file_path = resources.files("irtorch.datasets.national_mathematics").joinpath("mathematics_2019.feather")
        data = torch.from_numpy(feather.read_dataframe(file_path).to_numpy()).float()
    except Exception as e:
        raise RuntimeError("Failed to load data") from e
    return data

def swedish_national_mathematics_2018():
    """
    Load the Swedish National Mathematics 2018 dataset.

    Returns
    -------
    torch.Tensor
        The loaded dataset.
        
    Raises
    ------
    RuntimeError
        If the dataset cannot be loaded.
    """
    try:
        file_path = resources.files("irtorch.datasets.national_mathematics").joinpath("mathematics_2018.feather")
        data = torch.from_numpy(feather.read_dataframe(file_path).to_numpy()).float()
    except Exception as e:
        raise RuntimeError("Failed to load data") from e
    return data

def swedish_sat_verbal_2022() -> tuple[torch.Tensor, list[int]]:
    """
    Load the Swedish SAT Verbal 2022 dataset and the correct item responses. The correct item responses start from one, and thus a 1 corresponds to a response of 0 in the data.

    Returns
    -------
    tuple (torch.Tensor, list[int])
        A tuple containing the loaded dataset and the correct item responses.
        
    Raises
    ------
    RuntimeError
        If the dataset or the correct item responses cannot be loaded.
    """
    try:
        file_path = resources.files("irtorch.datasets.swedish_sat").joinpath("swesat22b_nominal_verb.feather")
        data = torch.from_numpy(feather.read_dataframe(file_path).to_numpy()).float()
    except Exception as e:
        raise RuntimeError("Failed to load data") from e

    try:
        with resources.files('irtorch.datasets.swedish_sat').joinpath('swesat22b_verb_correct.txt').open('r', encoding='utf-8') as file:
            correct_category = file.read().replace("\n", "")
            correct_category = [int(char) for char in correct_category if char.isdigit()]
    except Exception as e:
        raise RuntimeError("Failed to load correct item responses") from e

    return data - 1, correct_category

def swedish_sat_quantitative_2022() -> tuple[torch.Tensor, list[int]]:
    """
    Load the Swedish SAT Quantitative 2022 dataset and the correct item responses. The correct item responses start from one, and thus a 1 corresponds to a response of 0 in the data.

    Returns
    -------
    tuple (torch.Tensor, list[int])
        A tuple containing the loaded dataset and the correct item responses.
        
    Raises
    ------
    RuntimeError
        If the dataset or the correct item responses cannot be loaded.
    """
    try:
        file_path = resources.files("irtorch.datasets.swedish_sat").joinpath("swesat22b_nominal_quant.feather")
        data = torch.from_numpy(feather.read_dataframe(file_path).to_numpy()).float()
    except Exception as e:
        raise RuntimeError("Failed to load data") from e

    try:
        with resources.files('irtorch.datasets.swedish_sat').joinpath('swesat22b_quant_correct.txt').open('r', encoding='utf-8') as file:
            correct_category = file.read().replace("\n", "")
            correct_category = [int(char) for char in correct_category if char.isdigit()]
    except Exception as e:
        raise RuntimeError("Failed to load correct item responses") from e

    return data - 1, correct_category

def swedish_sat_2022() -> tuple[torch.Tensor, list[int]]:
    """
    Load the full Swedish SAT 2022 dataset and the correct item responses. The correct item responses start from one, and thus a 1 corresponds to a response of 0 in the data. The first 80 items are from the quantitative test, and the last 80 items are from the verbal test. 

    Returns
    -------
    tuple (torch.Tensor, list[int])
        A tuple containing the loaded dataset and the correct item responses.
        
    Raises
    ------
    RuntimeError
        If the dataset or the correct item responses cannot be loaded.
    """
    data_quant, correct_quant = swedish_sat_quantitative_2022()
    data_verb, correct_verb = swedish_sat_verbal_2022()
    data = torch.cat((data_quant, data_verb), dim=1)
    # concatenate the correct item responses
    correct_category = correct_quant + correct_verb

    return data, correct_category

def swedish_sat_2022_binary() -> torch.Tensor:
    """
    Load the full Swedish SAT 2022 dataset coded as incorrect/correct. 

    Returns
    -------
    torch.Tensor
        A tuple containing the loaded dataset and the correct item responses.
        
    Raises
    ------
    RuntimeError
        If the dataset or the correct item responses cannot be loaded.
    """
    data_quant, correct_quant = swedish_sat_quantitative_2022()
    data_verb, correct_verb = swedish_sat_verbal_2022()
    data = torch.cat((data_quant, data_verb), dim=1)
    # concatenate the correct item responses
    correct_category = correct_quant + correct_verb

    correct_scores = torch.tensor(correct_category)
    binary_data = (data == correct_scores).float()

    return binary_data

def big_five() -> torch.Tensor:
    """
    Load the big five dataset.

    Returns
    -------
    torch.Tensor
        The loaded dataset.
        
    Raises
    ------
    RuntimeError
        If the dataset or the correct item responses cannot be loaded.
    """
    try:
        file_path = resources.files("irtorch.datasets.big_five").joinpath("big_five.csv")
        data = torch.from_numpy(pd.read_csv(file_path, delimiter='\t').iloc[:, 7:].to_numpy()).float()
    except Exception as e:
        raise RuntimeError("Failed to load data") from e

    data[data == 0] = float('nan')
    return data - 1
