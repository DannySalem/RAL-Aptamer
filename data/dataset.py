import os
import warnings
from torch.utils.data import Dataset
import torch

path = "/datasets/aptamer"


class Aptamer_Dataset(Dataset):
    def __init__(self, data: list, islabelled: bool = False):
        self.islabelled = islabelled
        self.data = data
        self.device = torch.device("cuda:0")
        self.make_dataset()
        if len(self.data) == 0:
            raise RuntimeError("Found 0 sequences, please check the data set")

    # TODO Write a make_dataset method to
    def make_dataset(self):
        self.sequences = torch.tensor(self.data["samples"]).to(self.device)
        if self.islabelled:
            self.labels = torch.tensor(self.data["scores"], dtype=torch.float32).to(self.device)

    def add_datapoint(self, datapoint):
        if self.islabelled:
            self.sequences = torch.cat((self.sequences, torch.unsqueeze(datapoint[0], 0)))
            self.labels = torch.cat(
                (self.labels, torch.tensor(datapoint[1], device=self.device, dtype=torch.float32))
            )
        else:
            raise RuntimeError("Trying to Add Datapoint to Unlabelled Dataset. This is an error.")

    def __getitem__(self, index):
        if self.islabelled:
            return (self.sequences[index], self.labels[index])
        else:
            return self.sequences[index]

    def __len__(self):
        return len(self.sequences)
