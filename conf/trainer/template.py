from dataclasses import dataclass


@dataclass
class TrainerConfig:
    epochs: int
    learning_rate: float
    save: bool