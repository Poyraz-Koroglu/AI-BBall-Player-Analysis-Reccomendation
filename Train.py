import copy

import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def train_model(self,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer: Adam,
                criterion: nn.BCEWithLogitsLoss,
                epochs: int,
                device: str = "cuda" if torch.cuda.is_available() else "cpu",
                save_path: str = None,
                learning_rate: float = 1e-3,
                weight_decay: float = 1e-5,
                patience: int = 5,
                label_smoothing: float = 0.1,
                ):



   self.to(device)

   for param_group in optimizer.param_groups:
       param_group['lr'] = learning_rate
       param_group['weight_decay'] = weight_decay

   best_val_loss = float('inf')
   best_model_wts = copy.deepcopy(self.state_dict())
   patience_counter = 0


   history = {
       'train_loss': [],
       'train_acc': [],
       'val_loss': [],
       'val_acc': [],
       'learning_rates': [],
       'train_loss_smooth': [],  # Added this
       'val_loss_smooth': [],
       'all_batch_losses': []# Added this
   }
   print(f"🔥 Starting training on {device}...")

   for epoch in range(epochs):
       self.train()
       train_loss = 0.0
       train_acc = 0.0
       val_loss = 0.0
       val_acc = 0.0
       train_predictions = []
       train_labels = []

       batch_losses = []

       train_pbar = tqdm(train_loader,  desc=f"Epoch {epoch + 1}/{epochs} - Training")

       for x_num, x_cat, y in (train_pbar):
           x_num = x_num.to(device)
           x_cat = x_cat.to(device)
           y = y.to(device).unsqueeze(1)

           y_smoothed = y * (1 - label_smoothing) + 0.5 * label_smoothing
           optimizer.zero_grad()
           logits = self(x_num, x_cat)
           loss = criterion(logits, y_smoothed)
           loss.backward()
           optimizer.step()

           loss_val = loss.item()
           batch_losses.append(loss_val)
           train_loss += loss_val

           # 2. Predictions & Labels
           probs = torch.sigmoid(logits)
           preds = (probs > 0.5).float()

           # Important: detach() and cpu() prevent GPU memory leaks
           train_predictions.extend(preds.detach().cpu().numpy().flatten())
           train_labels.extend(y.detach().cpu().numpy().flatten())

           train_pbar.set_postfix({'loss': f'{loss_val:.4f}'})

           # Calculate Epoch Metrics using the full lists
       epoch_train_loss = np.mean(batch_losses)
       # Simple accuracy calculation using numpy
       epoch_train_acc = 100 * np.mean(np.array(train_predictions) == np.array(train_labels))

       # Save batch losses to global history (for plotting later)
       history['all_batch_losses'].extend(batch_losses)

       # --- Validation Phase ---
       self.eval()
       val_loss = 0.0
       val_predictions = []
       val_labels = []

       with torch.no_grad():
           for x_num, x_cat, y in val_loader:
               x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device).unsqueeze(1)

               logits = self(x_num, x_cat)
               loss = criterion(logits, y)  # Validate on REAL targets

               val_loss += loss.item()

               # Store Val Preds
               probs = torch.sigmoid(logits)
               preds = (probs > 0.5).float()
               val_predictions.extend(preds.cpu().numpy().flatten())
               val_labels.extend(y.cpu().numpy().flatten())

       epoch_val_loss = val_loss / len(val_loader)
       epoch_val_acc = 100 * np.mean(np.array(val_predictions) == np.array(val_labels))

       # Record history
       history['train_loss'].append(epoch_train_loss)
       history['val_loss'].append(epoch_val_loss)
       history['train_acc'].append(epoch_train_acc)
       history['val_acc'].append(epoch_val_acc)

       print(f"Epoch {epoch + 1}: Train Loss {epoch_train_loss:.4f} ({epoch_train_acc:.1f}%) | "
             f"Val Loss {epoch_val_loss:.4f} ({epoch_val_acc:.1f}%)")


       if epoch_val_loss < best_val_loss:
           best_val_loss = epoch_val_loss
           best_model_wts = copy.deepcopy(self.state_dict())
           patience_counter = 0
           if save_path:
               torch.save(self.state_dict(), save_path)
       else:
           patience_counter += 1

       if patience_counter >= patience:
           print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
           break

       self.load_state_dict(best_model_wts)
       return history