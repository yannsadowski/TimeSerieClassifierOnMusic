import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import torch

from src.data.data import prepare_dataloaders
from src.models.LSTMModel import LSTMModel
from src.trainer.train import train

@hydra.main(config_path="conf/", config_name="default", version_base="1.1")
def main(dict_config: DictConfig):
    # Retrieve specific configurations
    dataset_config = dict_config.data
    model_config = dict_config.models
    trainer_config = dict_config.trainer
    wandb_config = dict_config.wandb

    # Determine the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if GPU is available, otherwise raise an error
    if device.type != 'cuda':
        raise RuntimeError("This program requires a GPU to run.")

    # Prepare the DataLoaders
    train_loader, val_loader, test_loader, output_size, input_size, genres = prepare_dataloaders(
        config=dataset_config
    )
    
    # Initialize the model with the configurations
    model = LSTMModel(
        input_size=input_size,
        output_size=output_size,
        num_layers_lstm=model_config.num_layers_lstm,
        num_layers_dense=model_config.num_layers_dense,
        hidden_size_multiplier=model_config.hidden_size_multiplier,
        dropout=model_config.dropout,
        norm_type=model_config.norm_type
    ).to(device)
    
    # Print the model for verification
    print(model)
    
    # Connect to Weights and Biases
    wandb.login(key=wandb_config.api_key)
    
    # Convert DictConfig to a standard dictionary for wandb
    wandb_config_dict = OmegaConf.to_container(dict_config, resolve=True, throw_on_missing=True)
    
    # Initialize Weights and Biases
    wandb.init(project=wandb_config.project_name, entity=wandb_config.get('entity', None), config=wandb_config_dict, settings=wandb.Settings(start_method="thread"))
    wandb.config.update({"chosen_genres": genres})
    
    # Start the training process
    train(model, train_loader, val_loader, test_loader, trainer_config)

if __name__ == "__main__":
    main()
