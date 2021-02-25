from os.path import dirname
import os
from pathlib import Path
import numpy as np

# logger = set_logger("DocumentController", verbose=settings.verbose)
# logger = set_logger(Path(__file__).name, verbose=settings.verbose)

encoding = 'utf8'
verbose = True

ROOT = dirname(dirname(__file__))
DATA_PATH = Path(ROOT, "data")
MODELS_PATH = Path(ROOT, "models")
LOG_PATH = Path(ROOT, "data", "log.txt")

PARTICLE_TRAIN = Path(ROOT, "data", "train.npy")
PARTICLE_VAL = Path(ROOT, "data", "val.npy")
PARTICLE_TEST = Path(ROOT, "data", "test.npy")
PARTICLE_DATASET_PATH = {"train": PARTICLE_TRAIN, "val": PARTICLE_VAL, "test": PARTICLE_TEST}

Path(ROOT, "models").mkdir(parents=True, exist_ok=True) # Model checkpoints or features
Path(ROOT, "reports").mkdir(parents=True, exist_ok=True) # Documents with text and charts documenting the results
Path(ROOT, "scripts").mkdir(parents=True, exist_ok=True) # Scripts to be executed manually
Path(ROOT, "data").mkdir(parents=True, exist_ok=True) # The data used in the experiments