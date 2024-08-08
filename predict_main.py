import os
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from src.models.LSTMModel import LSTMModel
from src.data.data import extract_sequences
from src.predict.predict import predict
from sklearn.metrics import accuracy_score
from collections import defaultdict
from tqdm import tqdm  

@hydra.main(config_path="conf/", config_name="default", version_base="1.1")
def predict_main(dict_config: DictConfig):
    
    # Retrieve specific configurations
    dataset_config = dict_config.data
    model_config = dict_config.models
    predict_config = dict_config.predict
    
    # Load class mapping
    with open(predict_config.class_mapping, 'r') as file:
        class_mapping = yaml.safe_load(file)["classes"]
    
    # Invert the mapping to get {class: index}
    class_to_index = {v: k for k, v in class_mapping.items()}
    
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
    
    # Store predictions and true labels
    all_predictions = []
    all_true_labels = []

    # Get the list of all .wav files
    all_files = []
    for class_name in os.listdir(predict_config.audio_file):
        class_dir = os.path.join(predict_config.audio_file, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith(".wav"):
                    all_files.append((class_name, os.path.join(class_dir, file_name)))

    # Iterate over each file with a progress bar
    for class_name, file_path in tqdm(all_files, desc="Processing files", unit="file"):
        true_label = class_to_index[class_name]
                    
        # Extract sequences from the audio file
        sequences = extract_sequences(file_path, sequence_size, transformer_class)
        if len(sequences) == 0:
            print(f"No sequences extracted from {file_path}. Skipping file.")
            continue
        
        # Perform predictions on each sequence, skipping skip_size
        aggregated_probabilities = torch.zeros(num_classes, device=device)
        for i in range(0, len(sequences), skip_size):
            sequence = sequences[i]
            prediction = predict(model, sequence)
            aggregated_probabilities += torch.tensor(prediction, device=device)
        
        # Determine the class with the highest overall probability
        final_prediction = aggregated_probabilities.argmax().item()
        
        # Store the prediction and the true label
        all_predictions.append(final_prediction)
        all_true_labels.append(true_label)
    
    # Calculate class-wise accuracy
    class_accuracies = defaultdict(list)
    for true_label, pred_label in zip(all_true_labels, all_predictions):
        class_accuracies[true_label].append(pred_label)
    
    print("Class-wise Accuracy:")
    for class_index, preds in class_accuracies.items():
        accuracy = accuracy_score([class_index] * len(preds), preds)
        class_name = class_mapping[class_index]
        print(f"{class_name}: {accuracy:.2f}")
    
    # Calculate overall accuracy
    overall_accuracy = accuracy_score(all_true_labels, all_predictions)
    print(f"Overall Accuracy: {overall_accuracy:.2f}")

if __name__ == "__main__":
    predict_main()
