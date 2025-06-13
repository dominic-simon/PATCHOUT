import torch
import numpy as np
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.iloc[idx, 0]
        output = self.df.iloc[idx, 1:]
        label = torch.from_numpy(np.array(label))
        output = torch.from_numpy(output.to_numpy())
        return output, label