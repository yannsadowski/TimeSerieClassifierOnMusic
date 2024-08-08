import numpy as np
import os
import random
import librosa
import pandas as pd
from typing import Any
import importlib
from sklearn.preprocessing import LabelEncoder
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to extract audio characteristics
def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        y_harm, y_perc = librosa.effects.hpss(y)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        rms = librosa.feature.rms(y=y)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        times = librosa.times_like(spectral_centroid)

        df = pd.DataFrame({
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'zcr': zcr,
            'rms': rms,
            'harmonic': y_harm[:len(times)],
            'percussive': y_perc[:len(times)],
            'spectral_rolloff': spectral_rolloff
        })

        return df
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

def get_transformer(class_path: str) -> Any:
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    transformer_class = getattr(module, class_name)
    return transformer_class

# Function to divide files into training, validation and test sets
def split_data(files, dist):
    random.shuffle(files)
    train_end = int(dist['train_size'] * len(files))
    val_end = train_end + int(dist['validation_size'] * len(files))
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    return train_files, val_files, test_files

# Function to create sequences for LSTM
def create_sequences(data, sequence_size):
    sequences = []
    for i in range(len(data) - sequence_size + 1):
        sequences.append(data[i:i + sequence_size])
    return np.array(sequences)

# Dataset class for PyTorch
class MusicDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def process_genre(genre, pathRaw, dist, sequence_size, transformer_class, label_encoder):
    genre_start_time = time.time()
    genre_path = os.path.join(pathRaw, genre)
    files = [f for f in os.listdir(genre_path) if os.path.isfile(os.path.join(genre_path, f))]
    train_files, val_files, test_files = split_data(files, dist)

    sets = {
        'train': (train_files, [], []),
        'validation': (val_files, [], []),
        'test': (test_files, [], [])
    }

    for set_type in sets.keys():
        set_files, set_data, set_labels = sets[set_type]
        
        for file in set_files:
            file_path = os.path.join(genre_path, file)
            features = extract_audio_features(file_path)

            if features.empty:
                print(f"Skipping {file_path} due to empty features.")
                continue
            
            # Create sequence
            sequences = create_sequences(features.values, sequence_size)

            if sequences.size == 0:
                print(f"No sequences created for {file_path}.")
                continue
            
            # Create new transformer and fit transform for EACH sequences
            transformer = transformer_class()
            transformed_sequences = np.array([transformer.fit_transform(seq) for seq in sequences])
            
            # Add the transformed sequences and labels to the correct set
            set_data.append(transformed_sequences)
            encoded_labels = label_encoder.transform([genre] * len(transformed_sequences))
            set_labels.append(encoded_labels)
        
        # Reassign updated data in the dictionary
        sets[set_type] = (set_files, np.concatenate(set_data, axis=0), np.concatenate(set_labels, axis=0))

    genre_end_time = time.time()
    genre_time_taken = genre_end_time - genre_start_time
    print(f"Genre '{genre}' processed in {genre_time_taken:.2f} seconds.")
    
    return sets['train'][1], sets['train'][2], sets['validation'][1], sets['validation'][2], sets['test'][1], sets['test'][2]

# Main function to transform data and create DataLoaders
def prepare_dataloaders(config):
    # Load the configuration file
    pathRaw = config['path']['path_raw']
    dist = config['dist']
    sequence_size = config['params']['sequence_size']
    transformer_class = get_transformer(config['params']['transform'])
    batch_size = config['params']['batch_size']
    num_classes = config['params']['num_classes']
    
    genres = [d for d in os.listdir(pathRaw) if os.path.isdir(os.path.join(pathRaw, d))]
    
    # Select genres randomly 
    # Usefull when less than 10 genres are selected
    if len(genres) > num_classes:
        genres = random.sample(genres, num_classes)
    
    # Initialize the label encoder and adjust on selected genres
    label_encoder = LabelEncoder()
    label_encoder.fit(genres)
    
    train_data, val_data, test_data = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    
    with ThreadPoolExecutor() as executor:
        future_to_genre = {executor.submit(process_genre, genre, pathRaw, dist, sequence_size, transformer_class, label_encoder): genre for genre in genres}
        for future in as_completed(future_to_genre):
            genre = future_to_genre[future]
            try:
                train_df, train_lbl, val_df, val_lbl, test_df, test_lbl = future.result()

                train_data.append(train_df)
                train_labels.append(train_lbl)

                val_data.append(val_df)
                val_labels.append(val_lbl)

                test_data.append(test_df)
                test_labels.append(test_lbl)
            except Exception as e:
                print(f"Error processing genre {genre}: {e}")
    
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    val_data = np.concatenate(val_data, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    test_data = np.concatenate(test_data, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Create PyTorch datasets
    train_dataset = MusicDataset(train_data, train_labels)
    val_dataset = MusicDataset(val_data, val_labels)
    test_dataset = MusicDataset(test_data, test_labels)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create and save the class mapping file
    # Need it to be able to found the class after inference
    class_mapping = {i: str(genre) for i, genre in enumerate(label_encoder.classes_)}
    # Specify the path of the class mapping file to root
    mapping_file_path = 'class_mapping.yaml'

    # Save the mapping to a YAML file at root
    with open(mapping_file_path, 'w') as f:
        yaml.dump({'classes': class_mapping}, f, default_flow_style=False)
    
    return train_loader, val_loader, test_loader, len(genres), train_data.shape[2], genres

def extract_sequences(file_path, sequence_size, transformer_class):
    try:
        # Load the soundtrack
        y, sr = librosa.load(file_path, sr=None)
        y_harm, y_perc = librosa.effects.hpss(y)

        # Extract the audio characteristics
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        rms = librosa.feature.rms(y=y)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

        features = pd.DataFrame({
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'zcr': zcr,
            'rms': rms,
            'harmonic': y_harm[:len(spectral_centroid)],
            'percussive': y_perc[:len(spectral_centroid)],
            'spectral_rolloff': spectral_rolloff
        })
        
        
        # create sequences
        sequences = create_sequences(features.values, sequence_size)

        if sequences.size == 0:
            print(f"No sequences created for {file_path}. in fonction")
            return np.array([])  # Return an empty array if no sequence is created
        
        # Transform the sequences
        transformer_class = get_transformer(transformer_class)
        transformer = transformer_class()
        transformed_sequences = np.array([transformer.fit_transform(seq) for seq in sequences])

        return transformed_sequences

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.array([])  # Return an empty table in case of error

