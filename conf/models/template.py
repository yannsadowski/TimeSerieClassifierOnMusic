from dataclasses import dataclass

@dataclass
class ModelConfig:
    hidden_size_multiplier: int 
    num_layers_lstm: int 
    num_layers_dense: int 
    dropout: float 
    norm_type: str 
    model_path: str
