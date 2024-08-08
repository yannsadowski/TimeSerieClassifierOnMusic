# Time Series Classifier for Music Genre Classification

This project aims to build and train a time series classification model using LSTM networks to classify music genres. The data used in this project comes from the [GTZAN Dataset for Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data). This dataset contains a collection of 1000 audio tracks of 22050Hz Mono 16-bit audio files in .wav format categorized into 10 different genres, each 30 seconds long. The genres are Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, and Rock. This dataset is used to train and evaluate the performance of the LSTM-based model.

## Project Structure

The main directories and files in the project are structured as follows:

```
TimeSerieClassifier
│
├── conf
│   ├── data
│   ├── models
│   ├── predict
│   ├── trainer
│   ├── wandb
│   ├── default.yaml
│   ├── template.py
│   └── __init__.py
├── data
├── data_test
├── dev_env
├── model
├── outputs
├── wandb
├── src
│   ├── data
│   │   ├── data.py
│   ├── models
│   │   ├── LSTMModel.py
│   ├── predict
│   │   ├── predict.py
│   ├── trainer
│   │   ├── train.py
├── sweep_file
│   ├── batch.yaml
│   ├── class.yaml
│   ├── sequence_size.yaml
│   ├── wandb_sweep.yaml
├── main.py
├── predict_main.py
├── predict_single.py
├── pred_all.png
├── README.md
└── requirements.txt
```

## Environment Setup

### Create Virtual Environment

1. Create a virtual environment:

    ```sh
    python -m venv dev_env
    ```

2. Activate the virtual environment:

    ```sh
    ./dev_env/Scripts/Activate
    ```

3. Upgrade `pip` and install necessary packages:

    ```sh
    python.exe -m pip install --upgrade pip
    pip install pandas
    pip install numpy
    pip install wandb
    pip install hydra-core
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install librosa
    pip install matplotlib
    pip install tqm
    ```

### Start Virtual Environment

Activate the virtual environment whenever you start working on the project:

```sh
./dev_env/Scripts/Activate
```

### Update Requirements

If you install new packages, update the `requirements.txt` file:

```sh
pip freeze > requirements.txt
```

## Usage

### Training the Model

Run the `main.py` script to train the model:

```sh
python main.py
```

### Predicting

Use `predict_main.py` for all datasets:

```sh
python predict_main.py
```

Use `predict_single.py` for single audio file prediction:

```sh
python predict_single.py
```

## Configuration

Detailed configuration files are located in the `conf` directory. They are organized as follows:

- `conf/data`
- `conf/models`
- `conf/predict`
- `conf/trainer`
- `conf/wandb`

Refer to the respective YAML files for parameter settings and modify them as needed for your experiments.

## Dataset Preparation

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data). The downloaded dataset will contain a `data` folder with two subdirectories and two CSV files. Rename the `genres_original` folder to `data` and place it in the project's root directory.

After these steps, the root directory should contain a `data` folder with 10 subdirectories, each representing a music genre. Each subdirectory should contain 100 audio clips, each 30 seconds long.

## Testing on Custom Music

To test the model on music files other than those in the GTZAN dataset, ensure that they are in the following format:

- 22050Hz
- Mono
- 16-bit audio
- .wav format

## Experiment Tracking

We use [Weights & Biases](https://wandb.ai/) for experiment tracking. Ensure you have your API key configured in `conf/wandb/default.yaml`.