import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Optional, Dict, Tuple, List
'''
# ==========================================
# PART 1: PREPROCESSING SCRIPT
# (Run this once to generate the CSV, then comment it out)
# ==========================================

# 1. CONFIGURATION
file_path = r"D:\PycharmProjects\final_training_data_cumulative.csv"
output_filename = "final_training_data_cumulative.csv"

# 2. LOAD DATA
print(f"Reading Excel file...")
try:
    all_sheets = pd.read_excel(file_path, sheet_name=None)
except FileNotFoundError:
    print("❌ Error: File not found.")
    exit()

df_list = []
for sheet_name, sheet_df in all_sheets.items():
    sheet_df.columns = sheet_df.columns.str.strip()
    if 'League' not in sheet_df.columns:
        sheet_df['League'] = sheet_name
    df_list.append(sheet_df)

full_df = pd.concat(df_list, ignore_index=True)

# --- ADD THIS FIX ---
# Rename the columns so they match what your formulas and PyTorch model expect
global_column_mapping = {
    'FG': 'FGM',
    '3P': '3PM',
    'FT': 'FTM',
    'TRB': 'REB',
    'G': 'GP',
    'MP': 'MIN'
}
# Only rename columns that actually exist to prevent errors
full_df.rename(columns={k: v for k, v in global_column_mapping.items() if k in full_df.columns}, inplace=True)
# --------------------

# 3. ADVANCED FEATURE ENGINEERING
# ==========================================
print("Generating Advanced Basketball Stats...")
eps = 1e-6

# A. Efficiency Metrics
# True Shooting % (Adjusts for the value of FT and 3PT)
full_df['TS_pct'] = full_df['PTS'] / (2 * (full_df['FGA'] + 0.44 * full_df['FTA']) + eps)
# Effective FG %
full_df['eFG_pct'] = (full_df['FGM'] + 0.5 * full_df['3PM']) / (full_df['FGA'] + eps)
# Free Throw Rate
full_df['FT_Rate'] = full_df['FTA'] / (full_df['FGA'] + eps)

# B. Playmaking & Workload
full_df['AST_TOV_ratio'] = full_df['AST'] / (full_df['TOV'] + eps)
full_df['Usage_Proxy'] = (full_df['FGA'] + 0.44 * full_df['FTA'] + full_df['TOV'])


# C. Biological Context (Age) --- [FIXED SECTION] ---
def get_start_year(s):
    try:
        # .strip() and splitting by '-' handles both '2021-22' and '2021 - 22' safely
        return int(str(s).split('-')[0].strip())
    except:
        return 0

full_df['Year'] = full_df['Season'].apply(get_start_year)

# 1. Standardize 'Age' if it came from the new CSV/Excel
if 'Age' in full_df.columns:
    full_df.rename(columns={'Age': 'age'}, inplace=True)

# 2. Find and standardize the birth year column (e.g., 'birth year', 'Birth_Year')
for col in full_df.columns:
    if col.lower().replace(' ', '_') == 'birth_year':
        full_df.rename(columns={col: 'birth_year'}, inplace=True)
        break

# 3. Calculate Age Safely
if 'age' not in full_df.columns:
    full_df['age'] = pd.NA

if 'birth_year' in full_df.columns:
    full_df['birth_year'] = pd.to_numeric(full_df['birth_year'], errors='coerce')
    # Fill missing ages using the math. Leaves existing ages alone!
    full_df['age'] = full_df['age'].fillna(full_df['Year'] - full_df['birth_year'])

# 4. Safety net: Default missing ages to 25 to prevent model crashes
full_df['age'] = full_df['age'].fillna(25)
# ---------------------------------------------------


# D. Per Minute Normalization
#  identify players improving in quality even if minutes stay the same
full_df['PTS_per_min'] = full_df['PTS'] / (full_df['MIN'] + eps)
full_df['AST_per_min'] = full_df['AST'] / (full_df['MIN'] + eps)
full_df['REB_per_min'] = (full_df['ORB'] + full_df['DRB']) / (full_df['MIN'] + eps)

full_df['FGA_per_min'] = full_df['FGA'] / (full_df['MIN'] + eps)
full_df['3PA_per_min'] = full_df['3PA'] / (full_df['MIN'] + eps)
# 4. CLEANING & CUMULATIVE FEATURES
# ==========================================
full_df['EFF'] = (full_df['PTS'] + full_df['REB'] + full_df['AST'] + full_df['STL'] + full_df['BLK']
                  - (full_df['FGA'] - full_df['FGM']) - (full_df['FTA'] - full_df['FTM']) - full_df['TOV'])
full_df['EFF_per_min'] = full_df['EFF'] / (full_df['MIN'] + eps)

full_df = full_df.sort_values(['Player', 'Year'])
grouped = full_df.groupby('Player')

full_df['Career_GP'] = grouped['GP'].cumsum()
full_df['Career_MIN'] = grouped['MIN'].cumsum()
full_df['Career_EFF_Avg'] = grouped['EFF_per_min'].expanding().mean().reset_index(0, drop=True)
full_df['Prev_Season_EFF'] = grouped['EFF_per_min'].shift(1)
full_df['Trend_EFF'] = (full_df['EFF_per_min'] - full_df['Prev_Season_EFF']).fillna(0)

# 5. CLUSTERING & TARGET CREATION
# ==========================================
print("Finalizing Clustering and Targets...")
clustering_df = full_df[full_df['MIN'] > 50].copy()
# Use your original stat_cols but add our new normalized ones
stat_cols = ['FGA_per_min', '3PA_per_min', 'AST_per_min', 'REB_per_min', 'TS_pct', 'age']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(clustering_df[stat_cols].fillna(0))
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clustering_df['Archetype'] = kmeans.fit_predict(X_scaled)

full_df = clustering_df.copy()
full_df['Next_Season_EFF'] = full_df.groupby('Player')['EFF_per_min'].shift(-1)
full_df['Improved'] = (full_df['Next_Season_EFF'] > full_df['EFF_per_min']).astype(int)

final_data = full_df.dropna(subset=['Next_Season_EFF'])

# 6. SAVE
final_path = os.path.join(os.getcwd(), output_filename)
final_data.to_csv(final_path, index=False)
print(f"\n✅ SUCCESS! Data ready for GPU training with advanced features.")
'''

# ==========================================
# PART 2: DATASET CLASS
# (This is what you import in your training script)
# ==========================================
class BasketballPlayerDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            target_col: Optional[str] = None,
            numerical_cols: Optional[List[str]] = None,
            categorical_cols: Optional[List[str]] = None,
            scaler: Optional[StandardScaler] = None,
            label_encoders: Optional[Dict[str, LabelEncoder]] = None,
            scale_features: bool = True,
    ):
        self.df = df.copy()
        self.target_col = target_col

        # 1. Setup Column Definitions
        # Default categorical columns (Added Archetype!)
        self.categorical_cols = categorical_cols if categorical_cols else ['League', 'Stage', 'Team', 'Archetype']

        # Default numeric columns (Added Cumulative Features!)
        self.numerical_cols = numerical_cols if numerical_cols else [
            'GP', 'MIN', 'FGM', 'FGA', '3PM', '3PA',
            'FTM', 'FTA', 'TOV', 'PF', 'DRB', 'ORB',
            'REB', 'AST', 'STL', 'BLK', 'PTS',
            # New History Features
            'Career_GP', 'Career_MIN', 'Career_EFF_Avg', 'Trend_EFF'
        ]

        # 2. Handle Scalers & Encoders
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
        self.cat_features = []
        for col in self.categorical_cols:
            self.df[col] = self.df[col].astype(str)

            if col not in self.label_encoders:
                le = LabelEncoder()
                le.fit(self.df[col])
                self.label_encoders[col] = le

            le = self.label_encoders[col]

            # Handle unknown classes safely
            known_classes = set(le.classes_)
            self.df[col] = self.df[col].apply(lambda x: x if x in known_classes else le.classes_[0])

            encoded_col = le.transform(self.df[col])
            self.cat_features.append(encoded_col)

        # Stack: Shape (N, num_cat_features)
        self.cat_features = np.stack(self.cat_features, axis=1).astype(np.int64)

        # --- B. Numerical Scaling ---
        # Ensure we only pick columns that actually exist in the DF
        valid_num_cols = [c for c in self.numerical_cols if c in self.df.columns]
        num_data = self.df[valid_num_cols].values.astype(np.float32)

        if self.scaler is None:
            self.scaler = StandardScaler()
            self.num_features = self.scaler.fit_transform(num_data)
        else:
            self.num_features = self.scaler.transform(num_data)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x_num = torch.tensor(self.num_features[idx], dtype=torch.float32)
        x_cat = torch.tensor(self.cat_features[idx], dtype=torch.long)

        if self.has_targets:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x_num, x_cat, y
        else:
            return x_num, x_cat