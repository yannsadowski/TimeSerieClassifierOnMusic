from dataclasses import dataclass

from .data.template import DatasetConfig
from .models.template import ModelConfig
from .trainer.template import TrainerConfig
from .predict.template import PredictConfig
from .wandb.template import WandbConfig


@dataclass
class MainConfig:
    data: DatasetConfig
    models: ModelConfig
    trainer: TrainerConfig
    predict : PredictConfig
    wandb: WandbConfig
