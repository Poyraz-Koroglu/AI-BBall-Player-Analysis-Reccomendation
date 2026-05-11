import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import copy
import os

# Import your custom dataset class
from GameLogsDataset import BasketballPlayerDataset


# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================
class BasketballImprovementModel(nn.Module):
    def __init__(self, num_numerical_cols=13, cat_dims=[10, 4], embedding_dims=[5, 2]):
        super(BasketballImprovementModel, self).__init__()

        # Categorical Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dims[i], embedding_dims[i]) for i in range(len(cat_dims))
        ])

        total_input_dim = num_numerical_cols + sum(embedding_dims)

        # Deep Network
        self.network = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)  # Output logits
        )

    def forward(self, x_num, x_cat):
        embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat_emb = torch.cat(embeddings, dim=1)
        x = torch.cat([x_num, x_cat_emb], dim=1)
        return self.network(x)


# ==========================================
# 2. TRAINING & EVALUATION FUNCTIONS
# ==========================================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, patience=7, scheduler=None):
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"🔥 Training started on {device}...")
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss, correct, total = 0, 0, 0

        # tqdm progress bar for the training batches
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for x_num, x_cat, y in train_pbar:
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(x_num, x_cat)
            loss = criterion(logits, y)
            loss.backward()

            # Gradient Clipping: Prevents math from exploding (nan protection)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            # Threshold at 0.5 for training metrics
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

            # Update progress bar with live loss
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # --- Validation Phase ---
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for x_num, x_cat, y in val_loader:
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device).unsqueeze(1)
                logits = model(x_num, x_cat)
                loss = criterion(logits, y)

                v_loss += loss.item()
                # Tracking validation accuracy
                preds = (torch.sigmoid(logits) > 0.5).float()
                v_correct += (preds == y).sum().item()
                v_total += y.size(0)

        # --- Record Metrics ---
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = v_loss / len(val_loader)
        epoch_train_acc = 100 * correct / total
        epoch_val_acc = 100 * v_correct / v_total

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        print(
            f"Epoch {epoch + 1}: T-Loss: {epoch_train_loss:.4f} | V-Loss: {epoch_val_loss:.4f} | V-Acc: {epoch_val_acc:.2f}%")

        # --- Scheduler Step ---
        # This is where we kickstart the learning rate if the model plateaus
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # ReduceLROnPlateau needs the validation loss to decide to drop LR
                scheduler.step(epoch_val_loss)
            else:
                # Other schedulers (like StepLR) just step once per epoch
                scheduler.step()

        # --- Early Stopping Logic ---
        # We only save the "Best" weights if the validation loss improves
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Optional: Print when a new best model is saved
            # print(f"✨ New best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered after {epoch + 1} epochs.")
                break

    # Load the absolute best weights before returning
    model.load_state_dict(best_model_wts)
    return history


def evaluate_model(model, loader, device, threshold=0.40):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_num, x_cat, y in loader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            logits = model(x_num, x_cat)
            preds = (torch.sigmoid(logits) > threshold).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    return np.array(all_labels), np.array(all_preds)


# ==========================================
# 3. PLOTTING SUITE
# ==========================================
def plot_results(history, test_labels, test_preds):
    # Plot 1: Loss & Accuracy Curves
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].plot(history['train_loss'], label='Train Loss')
    ax[0].plot(history['val_loss'], label='Val Loss')
    ax[0].set_title('Loss Curves')
    ax[0].legend()

    ax[1].plot(history['train_acc'], label='Train Acc')
    ax[1].plot(history['val_acc'], label='Val Acc')
    ax[1].set_title('Accuracy Curves')
    ax[1].legend()
    plt.savefig('Data/training_plots.png')

    # Plot 2: Confusion Matrix (Test Set)
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(test_labels, test_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Test Set Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('Data/test_confusion_matrix.png')
    print("📈 Plots saved to Data/ folder.")


# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Data
    df = pd.read_csv("Data/ml_ready_data.csv")
    full_dataset = BasketballPlayerDataset(df)

    # Triple Split (70/15/15)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    # Initialize Model
    n_leagues = len(full_dataset.label_encoders['competition'].classes_)
    n_archetypes = len(full_dataset.label_encoders['archetype_cluster'].classes_)
    model = BasketballImprovementModel(cat_dims=[n_leagues, n_archetypes]).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 'min' because we want to minimize validation loss
        factor=0.5,  # Multiplies LR by 0.5 when it plateaus
        patience=3,  # Number of epochs to wait before dropping LR
        verbose=True
    )
    # Train
    history = train_model(model, train_loader, val_loader, criterion, optimizer, 50, device,patience=10,scheduler=scheduler)

    # Final Test
    y_true, y_pred = evaluate_model(model, test_loader, device)
    print("\n--- Final Test Report ---")
    print(classification_report(y_true, y_pred))

    # Plot & Save
    plot_results(history, y_true, y_pred)
    torch.save(model.state_dict(), "Data/basketball_model.pth")