import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Import project modules
import config
from src.data_loader import get_dataloaders
from src.model import ComplexityClassifier


def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with autocast():  # Mixed precision
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    """Main function to run the training pipeline."""
    set_seed(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # --- 1. Load Data ---
    train_dl, val_dl, _, tokenizer, label2id, _, class_weights = get_dataloaders(config)

    # --- 2. Initialize Model, Loss, and Optimizer ---
    model = ComplexityClassifier(
        vocab_size=len(tokenizer),
        num_labels=len(label2id),
        d_model=config.D_MODEL,
        nhead=config.N_HEAD,
        num_layers=config.NUM_LAYERS,
        max_len=config.MAX_TOKEN_LENGTH,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.LABEL_SMOOTHING)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler(config.DEVICE)

    # --- 3. Training Loop ---
    best_accuracy = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_accuracy': []}

    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.NUM_EPOCHS} ---")

        train_loss = train_one_epoch(model, train_dl, criterion, optimizer, scaler, config.DEVICE)
        val_accuracy = evaluate(model, val_dl, config.DEVICE)

        history['train_loss'].append(train_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f"Avg Train Loss: {train_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

        scheduler.step(val_accuracy)

        # Early Stopping and Model Saving
        if val_accuracy > best_accuracy + config.ES_DELTA:
            best_accuracy = val_accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"✓ Model saved to {config.MODEL_SAVE_PATH}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= config.ES_PATIENCE:
                print("⏹️ Early stopping triggered.")
                break

    # --- 4. Plotting Results ---
    plt.figure()
    plt.plot(history['train_loss'], label="Train Loss")
    plt.plot(history['val_accuracy'], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.title("Training History")
    plt.legend()
    plt.savefig(config.PLOT_SAVE_PATH)
    print(f"Training plot saved to {config.PLOT_SAVE_PATH}")


if __name__ == '__main__':
    main()