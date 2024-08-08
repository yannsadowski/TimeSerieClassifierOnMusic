# Time Series Classifier Configuration Guide

This guide provides an overview of the configuration setup for the Time Series Classifier project. The configuration is organized using YAML files and Python scripts to streamline data processing, model training, prediction, and experiment tracking.

## Directory Structure

The main configuration files and directories are structured as follows:

```
D:\code\TimeSerieClassifier\conf
│
├── data
│   ├── __pycache__
│   ├── default.yaml
│   └── template.py
│
├── models
│   ├── __pycache__
│   ├── default.yaml
│   └── template.py
│
├── predict
│   ├── __pycache__
│   ├── default.yaml
│   └── template.py
│
├── trainer
│   ├── __pycache__
│   ├── default.yaml
│   └── template.py
│
├── wandb
│   ├── __pycache__
│   ├── default.yaml
│   └── template.py
│
├── __pycache__
├── default.yaml
├── template.py
└── __init__.py
```

### Main Configuration (`template.py`)

The `template.py` file sets up the main configuration class using dataclasses. It imports configuration classes from various submodules.

```python
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
    predict: PredictConfig
    wandb: WandbConfig
```

### Default Configuration (`default.yaml`)

The `default.yaml` file specifies the default configurations to be used.

```yaml
defaults:
  - _self_
  - data: default
  - models: default
  - trainer: default
  - predict: default
  - wandb: default

hydra:
  job:
    chdir: true
```

## Data Configuration

The data configuration is defined in the `data/default.yaml` file and the `data/template.py` script.

### Data Default Configuration (`data/default.yaml`)

```yaml
dist:
  train_size: 0.7
  validation_size: 0.15
  test_size: 0.15

path:
  path_raw: /home/paperspace/Desktop/TimeSerieClassifier/data

params:
  sequence_size: 43
  transform: sklearn.preprocessing.MinMaxScaler
  batch_size: 25000
  num_classes: 10
```

### Data Template (`data/template.py`)

This script defines the dataset configuration dataclass.

```python
from dataclasses import dataclass

@dataclass
class DatasetConfig:
    dist: dict
    path: dict
    params: dict
```

## Model Configuration

The model configuration is defined in the `models/default.yaml` file.

### Model Default Configuration (`models/default.yaml`)

```yaml
hidden_size_multiplier: 7    # Example hidden size for LSTM
num_layers_lstm: 3 # Number of LSTM layers
num_layers_dense: 5 # Number of Dense layers
dropout: 0.1437946524814418
norm_type: rmsnorm
model_path: D:\code\TimeSerieClassifier\best_model.pth
```

## Prediction Configuration

The prediction configuration is defined in the `predict/default.yaml` file.

### Predict Default Configuration (`predict/default.yaml`)

```yaml
audio_file: D:\code\TimeSerieClassifier\data
class_mapping: D:\code\TimeSerieClassifier\class_mapping.yaml
single_audio_file: D:\code\TimeSerieClassifier\data_test\Dream On - Aerosmith.wav
```

## Trainer Configuration

The trainer configuration is defined in the `trainer/default.yaml` file.

### Trainer Default Configuration (`trainer/default.yaml`)

```yaml
epochs: 150
learning_rate: 0.0009642643558379736
save: True
```

## Weights & Biases (Wandb) Configuration

The Wandb configuration is defined in the `wandb/default.yaml` file.

### Wandb Default Configuration (`wandb/default.yaml`)

```yaml
api_key: 
project_name: 
```

## Summary

This setup allows you to manage and customize the configuration for different components of the Time Series Classifier project efficiently. By organizing the configuration settings in a structured way, you can easily modify parameters, paths, and other settings to fit your specific needs.