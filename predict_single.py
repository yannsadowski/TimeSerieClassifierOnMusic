import os
import yaml
import torch
import hydra
from omegaconf import DictConfig
from src.models.LSTMModel import LSTMModel
from src.data.data import extract_sequences
from src.predict.predict import predict
from collections import defaultdict

@hydra.main(config_path="conf/", config_name="default", version_base="1.1")
def predict_single(dict_config: DictConfig):
    # Retrieve specific configurations
    dataset_config = dict_config.data
    model_config = dict_config.models
    predict_config = dict_config.predict
    
    # Load class mapping
    with open(predict_config.class_mapping, 'r') as file:
        class_mapping = yaml.safe_load(file)["classes"]
    
    # Invert the mapping to get {class: index}
    class_to_index = {v: k for k, v in class_mapping.items()}
    index_to_class = {v: k for k, v in class_to_index.items()}
    
    # Prepare parameters
    sequence_size = dataset_config.params.sequence_size
    transformer_class = dataset_config.params.transform
    num_classes = dataset_config.params.num_classes
    skip_size = dataset_config.params.get("skip_size", 22)  # Default to 22 if not specified

    # Determine the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if GPU is available, otherwise raise an error
    if device.type != 'cuda':
        raise RuntimeError("This program requires a GPU to run.")

    # Initialize the model with the configurations
    model = LSTMModel(
        input_size=7,
        output_size=num_classes,
        num_layers_lstm=model_config.num_layers_lstm,
        num_layers_dense=model_config.num_layers_dense,
        hidden_size_multiplier=model_config.hidden_size_multiplier,
        dropout=model_config.dropout,
        norm_type=model_config.norm_type
    ).to(device)
    
    # Load the model
    model_path = model_config.model_path
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Get the path of the audio file to predict
    audio_file_path = predict_config.single_audio_file
    
    # Extract sequences from the audio file
    sequences = extract_sequences(audio_file_path, sequence_size, transformer_class)
    if len(sequences) == 0:
        print(f"No sequences extracted from {audio_file_path}. Skipping file.")
        return
    
    # Perform predictions on each sequence, skipping skip_size
    aggregated_probabilities = torch.zeros(num_classes, device=device)
    for i in range(0, len(sequences), skip_size):
        sequence = sequences[i]
        prediction = predict(model, sequence)
        aggregated_probabilities += torch.tensor(prediction, device=device)
    
    # Display probabilities for each class
    print(f"Probabilities for {audio_file_path}:")
    for class_index, prob in enumerate(aggregated_probabilities):
        class_name = index_to_class[class_index]
        print(f"{class_name}: {prob.item():.4f}")
    
    # Determine the class with the highest probability
    final_prediction = aggregated_probabilities.argmax().item()
    final_class_name = index_to_class[final_prediction]
    print(f"\nPredicted class: {final_class_name}")

if __name__ == "__main__":
    predict_single()
