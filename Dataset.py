import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Optional, Dict, Tuple, List

# ==========================================
# PART 1: PREPROCESSING SCRIPT
# (Run this once to generate the CSV, then comment it out)
# ==========================================

if __name__ == "__main__":
    # 1. CONFIGURATION
    # ==========================================
    # Your specific path
    file_path = r"C:\Users\spoyr\OneDrive\Masaüstü\Clean Data (1).xlsx"
    output_filename = "final_training_data_cumulative.csv"

    # 2. LOAD DATA
    # ==========================================
    print(f"Reading Excel file from: {file_path}...")

    try:
        all_sheets = pd.read_excel(file_path, sheet_name=None)
    except FileNotFoundError:
        print("❌ Error: File not found. Please check the 'file_path'.")
        exit()

    df_list = []
    for sheet_name, sheet_df in all_sheets.items():
        sheet_df.columns = sheet_df.columns.str.strip()

        # Add League if missing
        if 'League' not in sheet_df.columns:
            sheet_df['League'] = sheet_name

        df_list.append(sheet_df)
        print(f"   -> Loaded sheet: {sheet_name} ({len(sheet_df)} rows)")

    full_df = pd.concat(df_list, ignore_index=True)


    # 3. CLEANING & FORMATTING
    # ==========================================
    def get_start_year(s):
        try:
            return int(str(s).split(' - ')[0])
        except:
            return 0


    full_df['Year'] = full_df['Season'].apply(get_start_year)

    # Calculate Efficiency (EFF) per Minute
    full_df['EFF'] = (
            full_df['PTS'] + full_df['REB'] + full_df['AST'] + full_df['STL'] + full_df['BLK']
            - (full_df['FGA'] - full_df['FGM'])
            - (full_df['FTA'] - full_df['FTM'])
            - full_df['TOV']
    )
    full_df['EFF_per_min'] = full_df['EFF'] / (full_df['MIN'] + 1e-6)

    # Sort by Player and Year for history calculations
    full_df = full_df.sort_values(['Player', 'Year'])

    # 4. ADD CUMULATIVE / HISTORY FEATURES (NEW!)
    # ==========================================
    print("Generating Cumulative Career Stats...")

    grouped = full_df.groupby('Player')

    # A. Experience
    full_df['Career_GP'] = grouped['GP'].cumsum()
    full_df['Career_MIN'] = grouped['MIN'].cumsum()

    # B. Career Averages (Expanding Mean)
    full_df['Career_EFF_Avg'] = grouped['EFF_per_min'].expanding().mean().reset_index(0, drop=True)

    # C. Trend (vs Previous Season)
    full_df['Prev_Season_EFF'] = grouped['EFF_per_min'].shift(1)
    full_df['Trend_EFF'] = (full_df['EFF_per_min'] - full_df['Prev_Season_EFF']).fillna(0)

    # 5. CLUSTERING (ARCHETYPES)
    # ==========================================
    print("Running Clustering...")

    # Filter for clustering (valid minutes only)
    clustering_df = full_df[full_df['MIN'] > 50].copy()

    stat_cols = ['FGA', '3PA', 'FTA', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']
    features_for_clustering = []

    for col in stat_cols:
        new_col = f'{col}_per_min'
        clustering_df[new_col] = clustering_df[col] / (clustering_df['MIN'] + 1e-6)
        features_for_clustering.append(new_col)

    scaler = StandardScaler()
    X = scaler.fit_transform(clustering_df[features_for_clustering].fillna(0))

    kmeans = KMeans(n_clusters=5, random_state=42)
    clustering_df['Archetype'] = kmeans.fit_predict(X)

    # Merge Archetypes back
    full_df = clustering_df.copy()

    # 6. CREATE TARGET (Improved?)
    # ==========================================
    print("Calculating improvement targets...")

    # Re-sort to be safe
    full_df = full_df.sort_values(['Player', 'Year'])

    full_df['Next_Season_EFF'] = full_df.groupby('Player')['EFF_per_min'].shift(-1)
    full_df['Improved'] = (full_df['Next_Season_EFF'] > full_df['EFF_per_min']).astype(int)

    # Drop rows without a next season (Training Data)
    final_data = full_df.dropna(subset=['Next_Season_EFF'])

    # 7. SAVE
    # ==========================================
    final_path = os.path.join(os.getcwd(), output_filename)
    final_data.to_csv(final_path, index=False)

    print(f"\n✅ SUCCESS! Saved to: {final_path}")
    print(f"Rows: {len(final_data)}")
    print("New columns added: 'Career_GP', 'Career_MIN', 'Career_EFF_Avg', 'Trend_EFF'")


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
            label_encoders: Optional[Dict[str, LabelEncoder]] = None
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
