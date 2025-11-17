import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Optional, Dict, Tuple


class BasketballPlayerDataset(Dataset):
    """
    PyTorch Dataset for basketball player statistics and improvement prediction.

    Args:
        df: DataFrame containing player statistics
        target_col: Name of the target column (improvement label)
        scale_features: Whether to standardize numerical features
        categorical_cols: List of categorical columns to encode
        feature_cols: List of feature columns to use (if None, uses all stat columns)
    """

    def __init__(
            self,
            df: pd.DataFrame,
            target_col: Optional[str] = None,
            scale_features: bool = True,
            categorical_cols: Optional[list] = None,
            feature_cols: Optional[list] = None
    ):
        self.df = df.copy()
        self.target_col = target_col
        self.scale_features = scale_features

        # Default categorical columns
        if categorical_cols is None:
            categorical_cols = ['League', 'Stage', 'Team']
        self.categorical_cols = categorical_cols

        # Default feature columns (all numeric stats)
        if feature_cols is None:
            feature_cols = [
                'GP', 'MIN', 'FGM', 'FGA', '3PM', '3PA',
                'FTM', 'FTA', 'TOV', 'PF', 'DRB', 'ORB',
                'REB', 'AST', 'STL', 'BLK', 'PTS'
            ]
        self.feature_cols = feature_cols

        # Initialize encoders and scalers
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None

        # Process the data
        self._process_data()

    def _process_data(self):
        """Process and encode categorical variables, scale features."""

        # Encode categorical columns
        for col in self.categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le

        # Create list of encoded categorical columns
        self.encoded_cat_cols = [f'{col}_encoded' for col in self.categorical_cols
                                 if col in self.df.columns]

        # Combine feature columns with encoded categoricals
        self.all_feature_cols = self.feature_cols + self.encoded_cat_cols

        # Extract features
        self.features = self.df[self.all_feature_cols].values.astype(np.float32)

        # Scale features if requested
        if self.scale_features:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)

        # Extract targets if provided
        if self.target_col and self.target_col in self.df.columns:
            self.targets = self.df[self.target_col].values
            self.has_targets = True
        else:
            self.targets = None
            self.has_targets = False

        # Store metadata (player names, season, etc.)
        self.metadata = self.df[['Player', 'Season']].copy() if 'Player' in self.df.columns else None

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single sample from the dataset.

        Returns:
            If targets exist: (features, target)
            If no targets: (features,)
        """
        features = torch.tensor(self.features[idx], dtype=torch.float32)

        if self.has_targets:
            target = torch.tensor(self.targets[idx], dtype=torch.long)
            return features, target
        else:
            return (features,)

    def get_feature_names(self) -> list:
        """Return list of feature names in order."""
        return self.all_feature_cols

    def get_metadata(self, idx: int) -> Dict:
        """Get metadata for a specific sample."""
        if self.metadata is not None:
            return self.metadata.iloc[idx].to_dict()
        return {}

    def inverse_transform_features(self, scaled_features: np.ndarray) -> np.ndarray:
        """Inverse transform scaled features back to original scale."""
        if self.scaler is not None:
            return self.scaler.inverse_transform(scaled_features)
        return scaled_features

    def get_categorical_mapping(self, col: str) -> Dict:
        """Get the mapping for a categorical column."""
        if col in self.label_encoders:
            le = self.label_encoders[col]
            return dict(zip(le.classes_, le.transform(le.classes_)))
        return {}


if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'League': ['NBA', 'NBA', 'EuroLeague'],
        'Season': ['2023-24', '2023-24', '2023-24'],
        'Stage': ['Regular', 'Regular', 'Regular'],
        'Player': ['Player A', 'Player B', 'Player C'],
        'Team': ['Team1', 'Team2', 'Team3'],
        'GP': [72, 65, 58],
        'MIN': [32.5, 28.3, 25.1],
        'FGM': [8, 7, 6],
        'FGA': [16, 14, 13],
        '3PM': [2, 3, 2],
        '3PA': [6, 7, 5],
        'FTM': [4, 3, 2],
        'FTA': [5, 4, 3],
        'TOV': [2, 3, 2],
        'PF': [2, 3, 2],
        'DRB': [4, 3, 3],
        'ORB': [1, 1, 1],
        'REB': [5, 4, 4],
        'AST': [6, 4, 3],
        'STL': [1, 1, 1],
        'BLK': [0, 1, 1],
        'PTS': [22, 20, 16],
        'improved': [1, 0, 1]  # Target variable
    }

    df = pd.DataFrame(sample_data)

    # Create dataset
    dataset = BasketballPlayerDataset(
        df=df,
        target_col='improved',
        scale_features=True
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Feature names: {dataset.get_feature_names()}")
    print(f"Sample: {dataset[0]}")