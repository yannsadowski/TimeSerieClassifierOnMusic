import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_predictions = []

    for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_targets.extend(targets.tolist())
        all_predictions.extend(outputs.argmax(dim=1).tolist())

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = accuracy_score(all_targets, all_predictions)
    epoch_precision = precision_score(all_targets, all_predictions, average='macro')
    epoch_recall = recall_score(all_targets, all_predictions, average='macro')
    epoch_f1 = f1_score(all_targets, all_predictions, average='macro')

    return epoch_loss, epoch_accuracy, epoch_precision, epoch_recall, epoch_f1

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            all_targets.extend(targets.tolist())
            all_predictions.extend(outputs.argmax(dim=1).tolist())

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_accuracy = accuracy_score(all_targets, all_predictions)
    epoch_precision = precision_score(all_targets, all_predictions, average='macro')
    epoch_recall = recall_score(all_targets, all_predictions, average='macro')
    epoch_f1 = f1_score(all_targets, all_predictions, average='macro')

    return epoch_loss, epoch_accuracy, epoch_precision, epoch_recall, epoch_f1

def train(model, train_loader, val_loader, test_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # TODO: It will be better to make the choce of optimizer with configuration file to try different thing during sweep
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Initialize the learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        start_time = time.time()
        print(f"Epoch {epoch+1}/{config.epochs}")

        train_loss, train_accuracy, train_precision, train_recall, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = validate(model, val_loader, criterion, device)
        epoch_time = time.time() - start_time

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
        print(f"Train Precision: {train_precision:.4f}, Val Precision: {val_precision:.4f}")
        print(f"Train Recall: {train_recall:.4f}, Val Recall: {val_recall:.4f}")
        print(f"Train F1 Score: {train_f1:.4f}, Val F1 Score: {val_f1:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "train_precision": train_precision,
            "val_precision": val_precision,
            "train_recall": train_recall,
            "val_recall": val_recall,
            "train_f1": train_f1,
            "val_f1": val_f1,
            "epoch_time": epoch_time
        })

        # Step the scheduler based on the validation loss
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved!")

    print("Training complete. Testing model...")

    test_loss, test_accuracy, test_precision, test_recall, test_f1 = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")

    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1
    })
