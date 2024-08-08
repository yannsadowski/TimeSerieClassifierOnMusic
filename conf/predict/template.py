from dataclasses import dataclass


@dataclass
class PredictConfig:
    audio_file: str
    class_mapping: str
    single_audio_file: str