import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Optional, Dict, Tuple, List
# preprocess.py

import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Update this path to your file
file_path = r"C:\Users\spoyr\OneDrive\Masaüstü\Clean Data (1).xlsx"
output_filename = "final_training_data.csv"

# ==========================================
# 2. LOAD DATA (CORRECTED)
# ==========================================
print(f"Reading Excel file from: {file_path}...")

try:
    all_sheets = pd.read_excel(file_path, sheet_name=None)
except FileNotFoundError:
    print("❌ Error: File not found. Please check the 'file_path'.")
    exit()

df_list = []
for sheet_name, sheet_df in all_sheets.items():
    # Clean whitespace from column names
    sheet_df.columns = sheet_df.columns.str.strip()

    # CHECK: Only add 'League' if it's missing.
    # Since you confirmed it exists, we just use the data as-is.
    if 'League' not in sheet_df.columns:
        sheet_df['League'] = sheet_name

    df_list.append(sheet_df)
    print(f"   -> Loaded sheet: {sheet_name} ({len(sheet_df)} rows)")

full_df = pd.concat(df_list, ignore_index=True)

# ==========================================
# 3. PREPROCESSING & CLUSTERING
# ==========================================
print("Running Clustering...")

# Filter for clustering (ignore players with < 50 mins)
clustering_df = full_df[full_df['MIN'] > 50].copy()

# Create per-minute stats
stat_cols = ['FGA', '3PA', 'FTA', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']
features_for_clustering = []

for col in stat_cols:
    new_col = f'{col}_per_min'
    # safe division
    clustering_df[new_col] = clustering_df[col] / (clustering_df['MIN'] + 1e-6)
    features_for_clustering.append(new_col)

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(clustering_df[features_for_clustering].fillna(0))

# K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
clustering_df['Archetype'] = kmeans.fit_predict(X)

# Merge back: We prefer to keep all rows, even if they weren't clustered (assign -1 or similar if needed)
# For simplicity in ML, we usually just keep the clustered ones or infer the rest.
# Let's keep only valid players for the training set.
full_df = clustering_df.copy()

# ==========================================
# 4. CREATE TARGET (Improved?)
# ==========================================
print("Calculating improvement targets...")


def get_start_year(s):
    try:
        return int(str(s).split(' - ')[0])
    except:
        return 0


full_df['Year'] = full_df['Season'].apply(get_start_year)

# Calculate EFF
full_df['EFF'] = (
        full_df['PTS'] + full_df['REB'] + full_df['AST'] + full_df['STL'] + full_df['BLK']
        - (full_df['FGA'] - full_df['FGM'])
        - (full_df['FTA'] - full_df['FTM'])
        - full_df['TOV']
)
full_df['EFF_per_min'] = full_df['EFF'] / (full_df['MIN'] + 1e-6)

# Sort and Shift
full_df = full_df.sort_values(['Player', 'Year'])
full_df['Next_Season_EFF'] = full_df.groupby('Player')['EFF_per_min'].shift(-1)
full_df['Improved'] = (full_df['Next_Season_EFF'] > full_df['EFF_per_min']).astype(int)

# Drop rows without a next season
final_data = full_df.dropna(subset=['Next_Season_EFF'])

# ==========================================
# 5. SAVE
# ==========================================
final_path = os.path.join(os.getcwd(), output_filename)
final_data.to_csv(final_path, index=False)

print(f"\n✅ SUCCESS! Saved to: {final_path}")
print(f"Rows: {len(final_data)}")

class BasketballPlayerDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            target_col: Optional[str] = None,
            numerical_cols: Optional[List[str]] = None,
            categorical_cols: Optional[List[str]] = None,
            # Pass existing scalers/encoders for Test sets
            scaler: Optional[StandardScaler] = None,
            label_encoders: Optional[Dict[str, LabelEncoder]] = None
    ):
        self.df = df.copy()
        self.target_col = target_col

        # 1. Setup Column Definitions
        self.categorical_cols = categorical_cols if categorical_cols else ['League', 'Stage', 'Team']
        self.numerical_cols = numerical_cols if numerical_cols else [
            'GP', 'MIN', 'FGM', 'FGA', '3PM', '3PA',
            'FTM', 'FTA', 'TOV', 'PF', 'DRB', 'ORB',
            'REB', 'AST', 'STL', 'BLK', 'PTS'
        ]

        # 2. Handle Scalers & Encoders (Avoid Data Leakage)
        self.scaler = scaler
        self.label_encoders = label_encoders if label_encoders else {}

        # 3. Process Data
        self._process_features()

        # 4. Process Target
        if self.target_col and self.target_col in self.df.columns:
            self.targets = self.df[self.target_col].values
            self.has_targets = True
        else:
            self.targets = None
            self.has_targets = False

    def _process_features(self):
        # --- A. Categorical Encoding ---
        # We keep these as Integers (Long) for Embeddings
        self.cat_features = []
        for col in self.categorical_cols:
            # Handle unknown categories safely by converting to string
            self.df[col] = self.df[col].astype(str)

            if col not in self.label_encoders:
                # Fit new encoder (Training mode)
                le = LabelEncoder()
                le.fit(self.df[col])
                self.label_encoders[col] = le

            # Transform (handle unseen labels in test by mapping to a default if needed,
            # here we blindly transform for simplicity but real ML needs 'unknown' handling)
            # A simple trick: use .map and fillna for unknown classes
            le = self.label_encoders[col]

            # Safe transform: unknown classes get set to 0
            known_classes = set(le.classes_)
            self.df[col] = self.df[col].apply(lambda x: x if x in known_classes else le.classes_[0])

            encoded_col = le.transform(self.df[col])
            self.cat_features.append(encoded_col)

        # Stack categorical features: Shape (N, num_cat_features)
        self.cat_features = np.stack(self.cat_features, axis=1).astype(np.int64)

        # --- B. Numerical Scaling ---
        # We keep these as Floats for the Network
        num_data = self.df[self.numerical_cols].values.astype(np.float32)

        if self.scaler is None:
            self.scaler = StandardScaler()
            self.num_features = self.scaler.fit_transform(num_data)
        else:
            self.num_features = self.scaler.transform(num_data)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Return Numeric and Categorical separately
        x_num = torch.tensor(self.num_features[idx], dtype=torch.float32)
        x_cat = torch.tensor(self.cat_features[idx], dtype=torch.long)

        if self.has_targets:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)  # float for BCELoss, long for CrossEntropy
            return x_num, x_cat, y
        else:
            return x_num, x_cat
