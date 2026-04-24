import torch
import pandas as pd
from torch.utils.data import DataLoader
import os

# --- IMPORT YOUR CUSTOM CLASSES ---
from Dataset import BasketballPlayerDataset
from Model import ImprovementPredictor
from Train import evaluate

# ==========================================
# 1. CONFIGURATION
# ==========================================
# 1. Your original massive training file (We just need it to remember the scaling/teams)
TRAIN_CSV_FILE = "final_training_data_cumulative.csv"

# 2. Your newly generated test file
TEST_CSV_FILE = "NBA-Test-Result.csv"

MODEL_WEIGHTS = "basketball_model_best.pth"

BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Starting Final Evaluation on {DEVICE}")

# ==========================================
# 2. REBUILD THE EXACT TRAINING ENVIRONMENT
# ==========================================
print(f"Loading original training data from {TRAIN_CSV_FILE} to reconstruct the model shape...")
df_train = pd.read_csv(TRAIN_CSV_FILE)

# We initialize this ONLY so the code remembers all 759 teams and the original averages
train_dataset = BasketballPlayerDataset(
    df=df_train,
    target_col='Improved',
    scale_features=True
)

# ==========================================
# 3. PREPARE THE TEST DATA
# ==========================================
print(f"Loading test data from {TEST_CSV_FILE}...")
df_test = pd.read_csv(TEST_CSV_FILE)

# Apply the strict historical scalers and team IDs to the future data!
test_dataset = BasketballPlayerDataset(
    df=df_test,
    target_col='Improved',
    scaler=train_dataset.scaler,                # Inherit original scaling!
    label_encoders=train_dataset.label_encoders # Inherit original team IDs!
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 4. BUILD MODEL & LOAD WEIGHTS
# ==========================================
print(f"\nBuilding model architecture and loading weights from {MODEL_WEIGHTS}...")

# Calculate cardinalities based on the MASSIVE dataset, not the test dataset
cat_cardinalities = []
for col in train_dataset.categorical_cols:
    le = train_dataset.label_encoders[col]
    cat_cardinalities.append(len(le.classes_) + 1)

# The model is now the correct size!
model = ImprovementPredictor(
    num_numerical_features=len(train_dataset.numerical_cols),
    categorical_cardinalities=cat_cardinalities,
    hidden_units=[256, 128, 64],
    dropout=0.4
)

# This will now load flawlessly without a size mismatch
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True))
model.to(DEVICE)

# ==========================================
# 5. RUN EVALUATION
# ==========================================
print(f"\n🧪 Running Evaluation on {len(df_test)} Unseen Test Rows...")

# Call your evaluate function from Train.py
test_results = evaluate(model, dataloader=test_loader, device=DEVICE)

print("\n======================================")
print("🏆 FINAL OUT-OF-TIME TEST RESULTS 🏆")
print("======================================")
print(f"Test Loss:     {test_results['loss']:.4f}")
print(f"Test Accuracy: {test_results['accuracy']:.2f}%\n")

print("--- Classification Report ---")
print(test_results['classification_report'])

print("--- Confusion Matrix ---")
print(test_results['confusion_matrix'])