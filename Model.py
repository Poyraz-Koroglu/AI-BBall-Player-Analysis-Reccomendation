import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovementPredictor(nn.Module):
    def __init__(
            self,
            num_numerical_features,
            categorical_cardinalities,
            embedding_dims=None,
            hidden_units=[64, 32]
    ):
        """
        Args:
            num_numerical_features (int): Number of stat columns (PTS, AST, etc.)
            categorical_cardinalities (list of int): Number of unique classes for each category.
            embedding_dims (list of int): Embedding size for each category.
            hidden_units (list of int): Size of hidden layers.
        """
        super().__init__()

        # --- 1. Embedding Layers for Categorical Data ---
        # We create one embedding layer for each categorical column (League, Archetype, etc.)
        self.embeddings = nn.ModuleList()

        # If no custom dimensions provided, we use a heuristic: min(50, (x + 1) // 2)
        if embedding_dims is None:
            embedding_dims = [(x + 1) // 2 for x in categorical_cardinalities]

        for num_classes, emb_dim in zip(categorical_cardinalities, embedding_dims):
            self.embeddings.append(nn.Embedding(num_classes, emb_dim))

        # Calculate total input size for the first linear layer
        # Input = (All Embeddings combined) + (Numerical Features)
        total_input_dim = sum(embedding_dims) + num_numerical_features

        # --- 2. Feed Forward Network ---
        layers = []
        in_dim = total_input_dim

        for hidden_dim in hidden_units:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # Batch Norm helps convergence
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))  # Dropout prevents overfitting
            in_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # --- 3. Output Layer ---
        # Binary Classification: Output 1 value (logit)
        self.output_layer = nn.Linear(in_dim, 1)

    def forward(self, x_numerical, x_categorical):
        """
        x_numerical: Tensor of shape (Batch_Size, Num_Stats)
        x_categorical: Tensor of shape (Batch_Size, Num_Cat_Cols) (Ints)
        """

        # 1. Process Embeddings
        embedded_features = []
        for i, emb_layer in enumerate(self.embeddings):
            # Get the i-th column of categorical data
            col_data = x_categorical[:, i]
            # Lookup embedding
            embedded_features.append(emb_layer(col_data))

        # Concatenate all embeddings: (Batch, Total_Emb_Dim)
        x_emb = torch.cat(embedded_features, dim=1)

        # 2. Combine with Numerical Stats
        # Concatenate Embeddings + Stats: (Batch, Total_Emb_Dim + Num_Stats)
        x = torch.cat([x_emb, x_numerical], dim=1)

        # 3. Pass through network
        x = self.feature_extractor(x)

        # 4. Final Prediction
        logits = self.output_layer(x)

        # Return logits (we use BCEWithLogitsLoss later)
        return logits