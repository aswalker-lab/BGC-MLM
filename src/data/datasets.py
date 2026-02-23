# -*- coding: utf-8 -*-

from torch import tensor
from torch.utils.data import Dataset

class BGCDataset(Dataset):
    """
    Dataset for all BGC-MLM tasks. Data input should be CSV file with this format:
        classification, regression, embedding, [tokanized BGC sequence]
    Tokanized BGC sequence should already have tags and padding added as needed
    """
    def __init__(self, data, device=None):
        self.data = data
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {key: tensor(value).to(self.device) for key, value in self.data[idx].items()}
    