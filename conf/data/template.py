from dataclasses import dataclass, field
from typing import Any, Type
import hydra
from omegaconf import DictConfig
import pandas as pd
import importlib
from pathlib import Path

@dataclass
class DatasetConfig:
    train_size: float
    validation_size: float
    test_size: float
    path_raw: Path
    sequence_size: int
    batch_size: int
    num_classes : int
    transform: Any = field(default=None)




