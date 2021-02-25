from torch.utils.data import Dataset
from optw import settings
from typing import Literal
import numpy as np
import torch

def ParticleDataset(Dataset):

    def __init__(self, split: Literal["train", "test", "val"]):

        self.logger = set_logger("ParticleDataset")
        
        self.train = np.load(settings.PARTICLE_TRAIN, allow_pickle=False)
        self.test = np.load(settings.PARTICLE_TEST, allow_pickle=False)
        self.val = np.load(settings.PARTICLE_VAL, allow_pickle=False)

        self.data = getattr(split, self)
        self.data = self.data.astype(np.float32)
        self.data = torch.from_numpy(self.data)
        

    def __len__(self):
        
        self.logger.debug(f"ParticleDataset size is: {self.data.shape[0]}")
        return self.data.shape[0]

    def __getitem__(self, i):

        return self.data[i]
        


