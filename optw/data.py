from torch.utils.data import Dataset
from optw import settings
import numpy as np
import torch
from optw.utils.logger import set_logger

class ParticleDataset(Dataset):

    def __init__(self, split):

        self.logger = set_logger("ParticleDataset")
        path = settings.PARTICLE_DATASET_PATH[split]
        self.data = np.load(path, allow_pickle=False)
        self.data = self.data.astype(np.float32)
        self.data = torch.from_numpy(self.data)
        self.x = self.data[:, :-1]
        self.y = self.data[:, -1].long()
        self.sequence_length = self.x.shape[1]
        

    def __len__(self):
        
        self.logger.debug(f"ParticleDataset size is: {self.data.shape[0]}")
        return self.data.shape[0]

    def __getitem__(self, i):

        return (self.x[i], self.y[i])
