import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Optional, List, Dict


# ==========================================
# 1. DATA PREPROCESSING PIPELINE
# ==========================================
def preprocess_game_logs(input_csv, output_csv):
    print("1. Loading raw game logs...")
    df = pd.read_csv(input_csv)

    # Drop rows where minutes_played is 0 or NaN
    df = df.dropna(subset=['minutes_played'])
    df = df[df['minutes_played'] > 0]

    print("2. Sorting and grouping by player chronologically...")
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(by=['player_id', 'game_date'])

    print("3. Calculating Per-Minute Stats & True Shooting...")
    stat_cols = ['points', 'assists', 'tot_reb', 'steals', 'blocks', 'turnovers']

    # First: Calculate per-minute stats
    for col in stat_cols:
        df[f'{col}_per_min'] = df[col] / df['minutes_played']
        # CLIP OUTLIERS: Keep math stable for the Neural Network
        df[f'{col}_per_min'] = df[f'{col}_per_min'].clip(lower=0, upper=3.0)

    # Second: Calculate fga (needed for ts_pct)
    df['fga'] = df['fg2a'] + df['fg3a']

    # Third: Calculate ts_pct (CREATE the column first)
    df['ts_pct'] = np.where(
        (df['fga'] + 0.44 * df['fta']) > 0,
        df['points'] / (2 * (df['fga'] + 0.44 * df['fta'])) * 100,
        0
    )

    # Fourth: Now you can safely CLIP it
    df['ts_pct'] = df['ts_pct'].clip(lower=0, upper=120.0)

    print("4. Running K-Means Clustering for Player Archetypes...")
    cluster_features = [f'{col}_per_min' for col in stat_cols]
    df_cluster = df[cluster_features].fillna(0)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['archetype_cluster'] = kmeans.fit_predict(df_cluster)

    print("5. Calculating Role-Adjusted Efficiency (Cluster_EFF)...")
    df['missed_fg'] = df['fga'] - (df['fg2m'] + df['fg3m'])
    df['missed_ft'] = df['fta'] - df['ftm']
    df['base_eff'] = (df['points'] + df['tot_reb'] + df['assists'] + df['steals'] + df['blocks']) \
                     - (df['missed_fg'] + df['missed_ft'] + df['turnovers'])

    # Calculate first
    df['cluster_eff'] = df['base_eff'] * (df['ts_pct'] / 100 + 0.5)

    # CLIP to keep math stable (Prevents NaN)
    df['cluster_eff'] = df['cluster_eff'].clip(lower=-20, upper=100)

    print("6. Generating 15-Game Rolling Snapshots & Momentum...")
    df['recent_15g_eff'] = df.groupby('player_id')['cluster_eff'].transform(
        lambda x: x.rolling(window=15, min_periods=5).mean()
    )
    df['recent_15g_pts'] = df.groupby('player_id')['points_per_min'].transform(
        lambda x: x.rolling(window=15, min_periods=5).mean()
    )
    df['career_eff'] = df.groupby('player_id')['cluster_eff'].transform(
        lambda x: x.expanding().mean()
    )
    df['trend_eff'] = df['recent_15g_eff'] - df['career_eff']

    print("7. Calculating Upcoming Season Target Variable...")
    df['season_year'] = df['game_date'].dt.year + (df['game_date'].dt.month >= 8).astype(int)
    season_averages = df.groupby(['player_id', 'season_year'])['cluster_eff'].mean().reset_index()
    season_averages.rename(columns={'cluster_eff': 'season_avg_eff'}, inplace=True)
    season_averages['next_season_year'] = season_averages['season_year'] - 1
    season_averages.rename(columns={'season_avg_eff': 'next_season_eff'}, inplace=True)

    df = pd.merge(
        df,
        season_averages[['player_id', 'next_season_year', 'next_season_eff']],
        left_on=['player_id', 'season_year'],
        right_on=['player_id', 'next_season_year'],
        how='left'
    )

    df = df.dropna(subset=['next_season_eff', 'recent_15g_eff'])
    df['improved'] = (df['next_season_eff'] > df['recent_15g_eff']).astype(int)
    df = df.drop(columns=['season_year', 'next_season_year', 'next_season_eff'])

    print("8. Purging IDs and Exporting Tensor-Ready CSV...")
    # Dynamically check which columns exist before dropping to avoid KeyErrors
    potential_drops = [
        'log_id', 'player_id', 'team_id', 'game_date',
        'fg2m', 'fg2a', 'fg3m', 'fg3a', 'ftm', 'fta', 'points', 'tot_reb',
        'assists', 'steals', 'blocks', 'turnovers', 'fga', 'missed_fg', 'missed_ft', 'base_eff', 'efficiency'
    ]
    cols_to_drop = [c for c in potential_drops if c in df.columns]

    ml_ready_df = df.drop(columns=cols_to_drop)
    ml_ready_df.to_csv(output_csv, index=False)
    print(f"✅ Success! Tensor-ready dataset saved to: {output_csv}")
    print(f"Final shape: {ml_ready_df.shape}\n")


# ==========================================
# 2. PYTORCH DATASET CLASS
# ==========================================
class BasketballPlayerDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            target_col: Optional[str] = 'improved',
            numerical_cols: Optional[List[str]] = None,
            categorical_cols: Optional[List[str]] = None,
            scaler: Optional[StandardScaler] = None,
            label_encoders: Optional[Dict[str, LabelEncoder]] = None,
    ):
        self.df = df.copy()
        self.target_col = target_col

        self.categorical_cols = categorical_cols if categorical_cols else [
            'competition', 'archetype_cluster'
        ]

        self.numerical_cols = numerical_cols if numerical_cols else [
            'minutes_played', 'points_per_min', 'assists_per_min', 'tot_reb_per_min',
            'steals_per_min', 'blocks_per_min', 'turnovers_per_min', 'ts_pct',
            'cluster_eff', 'recent_15g_eff', 'recent_15g_pts', 'career_eff', 'trend_eff'
        ]

        self.scaler = scaler
        self.label_encoders = label_encoders if label_encoders else {}
        self._process_features()

        if self.target_col and self.target_col in self.df.columns:
            self.targets = self.df[self.target_col].values
            self.has_targets = True
        else:
            self.targets = None
            self.has_targets = False

    def _process_features(self):
        self.cat_features = []
        for col in self.categorical_cols:
            self.df[col] = self.df[col].astype(str)
            if col not in self.label_encoders:
                le = LabelEncoder()
                le.fit(self.df[col])
                self.label_encoders[col] = le
            le = self.label_encoders[col]
            known_classes = set(le.classes_)
            self.df[col] = self.df[col].apply(lambda x: x if x in known_classes else le.classes_[0])
            self.cat_features.append(le.transform(self.df[col]))

        self.cat_features = np.stack(self.cat_features, axis=1).astype(np.int64)

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
        return x_num, x_cat


if __name__ == "__main__":
    INPUT_FILE = "Data/game_logs.csv"
    OUTPUT_FILE = "Data/ml_ready_data.csv"
    preprocess_game_logs(INPUT_FILE, OUTPUT_FILE)

    print("Testing PyTorch Dataset Loader...")
    final_df = pd.read_csv(OUTPUT_FILE)
    dataset = BasketballPlayerDataset(df=final_df)
    x_num, x_cat, y = dataset[0]
    print(f"Numerical Tensor Shape: {x_num.shape}")
    print(f"Categorical Tensor Shape: {x_cat.shape}")
    print(f"Target Label: {y.item()}")