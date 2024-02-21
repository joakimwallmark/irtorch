import torch

class PytorchIRTDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor):
        super().__init__()
        self.data = data
        # set missing responses to 0 in the response mask (all non-missing are ones)
        self.mask = torch.zeros_like(data, dtype=torch.int)
        self.mask[data == -1] = 1

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.mask[index]

def is_jupyter():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            raise ImportError("Not in IPython")
        return True
    except Exception:
        return False

def dynamic_print(string_to_print):
    """
    Dynamically update terminal printout.

    Parameters
    ----------
    *args : str
        Strings to print

    """
    formatted_string = f"\r{string_to_print} " # small space after to make it look better in terminal
    print(formatted_string, end='', flush=True)
