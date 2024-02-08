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


class PytorchLinkingDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor):
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]
