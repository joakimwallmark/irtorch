import logging
from importlib import resources
import torch

logger = logging.getLogger("irtorch")

def swedish_national_mathematics_2() -> torch.Tensor:
    """
    Load the second Swedish National Mathematics dataset.

    Returns
    -------
    torch.Tensor
        The loaded dataset.
    """
    try:
        file_path = resources.files("irtorch") / "datasets" / "national_mathematics" / "mathematics_2.pt"
        data = torch.load(file_path)
    except Exception as e:
        raise RuntimeError("Failed to load data") from e
    return data

def swedish_national_mathematics_1() -> torch.Tensor:
    """
    Load the first Swedish National Mathematics dataset.

    Returns
    -------
    torch.Tensor
        The loaded dataset.
    """
    try:
        file_path = resources.files("irtorch") / "datasets" / "national_mathematics" / "mathematics_1.pt"
        data = torch.load(file_path)
    except Exception as e:
        raise RuntimeError("Failed to load data") from e
    return data

def swedish_sat_verbal() -> tuple[torch.Tensor, list[int]]:
    """
    Load a sample from the verbal part of the Swedish SAT dataset and the correct item responses. The correct item responses start from one, and thus a 1 corresponds to a response of 0 in the data.

    Returns
    -------
    tuple (torch.Tensor, list[int])
        A tuple containing the loaded dataset and the correct item responses.
    """
    try:
        file_path = resources.files("irtorch") / "datasets" / "swedish_sat" / "swesat22b_nominal_verb.pt"
        data = torch.load(file_path)
    except Exception as e:
        raise RuntimeError("Failed to load data") from e

    try:
        with resources.files("irtorch").joinpath("datasets/swedish_sat/swesat22b_verb_correct.txt").open("r", encoding="utf-8") as file:
            correct_category = file.read().replace("\n", "")
            correct_category = [int(char) for char in correct_category if char.isdigit()]
    except Exception as e:
        raise RuntimeError("Failed to load correct item responses") from e

    return data - 1, correct_category

def swedish_sat_quantitative() -> tuple[torch.Tensor, list[int]]:
    """
    Load a sample from the quantitative part Swedish SAT dataset and the correct item responses. The correct item responses start from one, and thus a 1 corresponds to a response of 0 in the data.

    Returns
    -------
    tuple (torch.Tensor, list[int])
        A tuple containing the loaded dataset and the correct item responses.
    """
    try:
        file_path = resources.files("irtorch") / "datasets" / "swedish_sat" / "swesat22b_nominal_quant.pt"
        data = torch.load(file_path)
    except Exception as e:
        raise RuntimeError("Failed to load data") from e

    try:
        with resources.files("irtorch").joinpath("datasets/swedish_sat/swesat22b_quant_correct.txt").open("r", encoding="utf-8") as file:
            correct_category = file.read().replace("\n", "")
            correct_category = [int(char) for char in correct_category if char.isdigit()]
    except Exception as e:
        raise RuntimeError("Failed to load correct item responses") from e

    return data - 1, correct_category

def swedish_sat() -> tuple[torch.Tensor, list[int]]:
    """
    Load a sample from Swedish SAT dataset and the correct item responses. The correct item responses start from one, and thus a 1 corresponds to a response of 0 in the data. The first 80 items are from the quantitative test, and the last 80 items are from the verbal test. 

    Returns
    -------
    tuple (torch.Tensor, list[int])
        A tuple containing the loaded dataset and the correct item responses.
    """
    data_quant, correct_quant = swedish_sat_quantitative()
    data_verb, correct_verb = swedish_sat_verbal()
    data = torch.cat((data_quant, data_verb), dim=1)
    # concatenate the correct item responses
    correct_category = correct_quant + correct_verb

    return data, correct_category

def swedish_sat_binary() -> torch.Tensor:
    """
    Load a sample from Swedish SAT dataset coded as incorrect/correct. 
    The first 40 items are from the quantitative part of the test. The last 40 items are from the verbal part.

    Returns
    -------
    torch.Tensor
        A tuple containing the loaded dataset.
    """
    data_quant, correct_quant = swedish_sat_quantitative()
    data_verb, correct_verb = swedish_sat_verbal()
    data = torch.cat((data_quant, data_verb), dim=1)
    # concatenate the correct item responses
    correct_category = correct_quant + correct_verb

    correct_scores = torch.tensor(correct_category) - 1
    binary_data = (data == correct_scores).float()

    return binary_data

def big_five() -> torch.Tensor:
    """
    Load the big five dataset.

    Returns
    -------
    torch.Tensor
        The loaded dataset.
    """
    try:
        file_path = resources.files("irtorch") / "datasets" / "big_five" / "big_five.pt"
        data = torch.load(file_path)
    except Exception as e:
        raise RuntimeError("Failed to load data") from e

    data[data == 0] = float("nan")
    return data - 1
