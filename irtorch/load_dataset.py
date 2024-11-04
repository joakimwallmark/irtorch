import os
import logging
from importlib import resources
import urllib.request
import shutil
import tempfile
import pickle
import torch
import pandas as pd

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
        data = torch.load(file_path, weights_only=False)
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
        data = torch.load(file_path, weights_only=False)
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
        data = torch.load(file_path, weights_only=False)
    except Exception as e:
        raise RuntimeError("Failed to load data") from e

    try:
        with resources.files("irtorch").joinpath("datasets/swedish_sat/swesat22b_verb_correct.txt").open("r", encoding="utf-8") as file:
            correct_category = file.read().replace("\n", "")
            correct_category = [int(char) - 1 for char in correct_category if char.isdigit()]
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
        data = torch.load(file_path, weights_only=False)
    except Exception as e:
        raise RuntimeError("Failed to load data") from e

    try:
        with resources.files("irtorch").joinpath("datasets/swedish_sat/swesat22b_quant_correct.txt").open("r", encoding="utf-8") as file:
            correct_category = file.read().replace("\n", "")
            correct_category = [int(char) - 1 for char in correct_category if char.isdigit()]
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
    The first 80 items are from the quantitative part of the test. The last 80 items are from the verbal part.

    Returns
    -------
    torch.Tensor
        The loaded dataset.
    """
    data_quant, correct_quant = swedish_sat_quantitative()
    data_verb, correct_verb = swedish_sat_verbal()
    data = torch.cat((data_quant, data_verb), dim=1)
    # concatenate the correct item responses
    correct_category = correct_quant + correct_verb

    correct_scores = torch.tensor(correct_category)
    binary_data = (data == correct_scores).float()

    return binary_data

def big_five() -> torch.Tensor:
    """
    Download and preprocess the Big Five personality dataset. Note that loading this dataset can be a bit slow due to the size of the dataset and the fact that it is downloaded from https://openpsychometrics.org/_rawdata/IPIP-FFM-data-8Nov2018.zip.
    The dataset is cached in the user's home directory to speed up subsequent calls.
    
    The Big Five personality traits, also known as the Five-Factor Model or the OCEAN model, are designed to assess personality traits. 
    An online Big Five personality test is available at [Open Psychometrics](https://openpsychometrics.org/tests/IPIP-BFFM/). 
    The test consists of 50 items on which the test taker rates how well each item describes them on a five-point scale.
    It contains data from between 2016 and 2018.

    Aside from the item responses, aditional features included in the dataset are:
    
    - **race**: Chosen from a drop-down menu.
        - 1 = Mixed Race
        - 2 = Arctic (Siberian, Eskimo)
        - 3 = Caucasian (European)
        - 4 = Caucasian (Indian)
        - 5 = Caucasian (Middle East)
        - 6 = Caucasian (North African, Other)
        - 7 = Indigenous Australian
        - 8 = Native American
        - 9 = North East Asian (Mongol, Tibetan, Korean, Japanese, etc.)
        - 10 = Pacific (Polynesian, Micronesian, etc.)
        - 11 = South East Asian (Chinese, Thai, Malay, Filipino, etc.)
        - 12 = West African, Bushmen, Ethiopian
        - 13 = Other
        - 0 = Missed

    - **age**: Entered as text (individuals reporting age < 13 were not recorded).

    - **engnat**: Response to "Is English your native language?".
        - 1 = Yes
        - 2 = No
        - 0 = Missed

    - **gender**: Chosen from a drop-down menu.
        - 1 = Male
        - 2 = Female
        - 3 = Other
        - 0 = Missed

    - **hand**: Response to "What hand do you use to write with?".
        - 1 = Right
        - 2 = Left
        - 3 = Both
        - 0 = Missed

    - **country**: The participant's technical location. ISO country code.

    - **source**: How the participant came to the test. Based on HTTP Referer. 
        - 1 = From another page on the test website
        - 2 = from google
        - 3 = from facebook
        - 4 = from any url with ".edu" in its domain name (e.g. xxx.edu, xxx.edu.au)
        - 6 = other source, or HTTP Referer not provided.

    On the page users were also asked to confirm that their answers were accurate and could be used for research. Participants who did not were not recorded.

    Returns
    -------
    tuple (torch.Tensor, pd.DataFrame, list[str])
        - A preprocessed version of the big-five dataset. Note that reverse-coded items have been reverse-coded. See notes for more details.
        - The original dataset as a pandas DataFrame.
        - The items in the dataset as a list of strings.

    Notes
    -----
    The returned dataset was preprocessed in the following way to directly support the package:
    
    >>> import os
    >>> import urllib.request
    >>> import shutil
    >>> import tempfile
    >>> import torch
    >>> import pandas as pd
    >>> 
    >>> data_url = "https://openpsychometrics.org/_rawdata/IPIP-FFM-data-8Nov2018.zip"
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    >>>     zip_filepath = os.path.join(tmpdirname, "big_five.zip")
    >>>     data_dir = os.path.join(tmpdirname, "IPIP-FFM-data-8Nov2018")
    >>>     filepath = os.path.join(data_dir, "data-final.csv")
    >>>     urllib.request.urlretrieve(data_url, zip_filepath)
    >>>     shutil.unpack_archive(zip_filepath, tmpdirname)
    >>>     original_data = pd.read_csv(filepath, sep="\t", header=0)
    >>> 
    >>> data = original_data.copy()
    >>> data.iloc[:, 50:100] = data.iloc[:, 50:100] / 1000 # time in seconds
    >>> # drop anyone with negative or really long response times (5min+).
    >>> keep = ((data.iloc[:, 50:100] < 0).sum(axis=1) == 0) & ((data.iloc[:, 50:100] > 300).sum(axis=1) == 0)
    >>> data = data[keep]
    >>> data = data[data["IPC"] == 1] # Drop multiple submissions from same IP address.
    >>> data = data.iloc[:, :50] # keep only response columns
    >>> data = data.dropna() # Drop people with NaN values.
    >>> data = data[data.sum(1) > 0] # Drop people with all missing responses (0 is missing).
    >>> data = data.mask(data == 0, float("nan")) # encode missing as nan
    >>> # Reverse-code reverse-coded items.
    >>> reverse_items = [
    >>>     "EXT2", "EXT4", "EXT6", "EXT8", "EXT10", "AGR1", "AGR3", "AGR5", "AGR7",
    >>>     "CSN2", "CSN4", "CSN6", "CSN8", "EST2", "EST4", "OPN2", "OPN4", "OPN6"
    >>> ]
    >>> data[reverse_items] = ((1 - data[reverse_items] / 5) * 5 + 1).mask(lambda col: col == 6, 0)
    >>> data = data - 1 # code first option as 0, second as 1, etc.
    >>> torch_data = torch.tensor(data.to_numpy(), dtype=torch.float32)
    """
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "irtorch", "big_five")
    os.makedirs(cache_dir, exist_ok=True)
    processed_data_path = os.path.join(cache_dir, "big_five_processed.pkl")

    if os.path.exists(processed_data_path):
        with open(processed_data_path, "rb") as f:
            torch_data, original_data = pickle.load(f)
    else:
        data_url = "https://openpsychometrics.org/_rawdata/IPIP-FFM-data-8Nov2018.zip"
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_filepath = os.path.join(tmpdirname, "big_five.zip")
            data_dir = os.path.join(tmpdirname, "IPIP-FFM-data-8Nov2018")
            filepath = os.path.join(data_dir, "data-final.csv")
            urllib.request.urlretrieve(data_url, zip_filepath)
            shutil.unpack_archive(zip_filepath, tmpdirname)
            original_data = pd.read_csv(filepath, sep="\t", header=0)

        data = original_data.copy()
        data.iloc[:, 50:100] = data.iloc[:, 50:100] / 1000 # time in seconds
        # drop anyone with negative or really long response times (5min+).
        keep = ((data.iloc[:, 50:100] < 0).sum(axis=1) == 0) & ((data.iloc[:, 50:100] > 300).sum(axis=1) == 0)
        data = data[keep]
        data = data[data["IPC"] == 1] # Drop multiple submissions from same IP address.
        data = data.iloc[:, :50] # keep only response columns
        data = data.dropna() # Drop people with NaN values.
        data = data[data.sum(1) > 0] # Drop people with all missing responses (0 is missing).
        data = data.mask(data == 0, float("nan")) # encode missing as nan
        # Reverse-code reverse-coded items.
        reverse_items = [
            "EXT2", "EXT4", "EXT6", "EXT8", "EXT10", "AGR1", "AGR3", "AGR5", "AGR7",
            "CSN2", "CSN4", "CSN6", "CSN8", "EST2", "EST4", "OPN2", "OPN4", "OPN6"
        ]
        data[reverse_items] = ((1 - data[reverse_items] / 5) * 5 + 1).mask(lambda col: col == 6, 0)
        data = data - 1 # code first option as 0, second as 1, etc.
        torch_data = torch.tensor(data.to_numpy(), dtype=torch.float32)
        # Save preprocessed data to cache
        with open(processed_data_path, "wb") as f:
            pickle.dump((torch_data, original_data), f)

    try:
        with resources.files("irtorch").joinpath("datasets/big_five/items.txt").open("r", encoding="utf-8") as file:
            items = [line.strip() for line in file.readlines() if line.strip()]
    except Exception as e:
        raise RuntimeError("Failed to load items.") from e

    return torch_data, original_data, items
