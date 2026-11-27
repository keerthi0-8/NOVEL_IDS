
import torch
import torch.nn as nn


class SharedFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.numeric_projection = nn.Linear(1, 64)
        self.protocol_embedding = nn.Embedding(134, 64)
        self.service_embedding = nn.Embedding(78, 64)
        self.feature_identity = nn.Embedding(5, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True, norm_first=True, activation='gelu'
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=2
        )

        self.layer_norm = nn.LayerNorm(64)

    def forward(self, shared_numeric, shared_categorical):

        numeric_tokens = self.numeric_projection(
            shared_numeric.unsqueeze(-1)
        )

        protocol_tokens = self.protocol_embedding(
            shared_categorical[:, 0]
        ).unsqueeze(1)

        service_tokens = self.service_embedding(
            shared_categorical[:, 1]
        ).unsqueeze(1)

        categorical_tokens = torch.cat(
            [protocol_tokens, service_tokens], dim=1
        )

        tokens = torch.cat(
            [numeric_tokens, categorical_tokens], dim=1
        )

        feature_ids = torch.arange(
            5, device=shared_numeric.device
        )

        identity_tokens = self.feature_identity(feature_ids)

        tokens = tokens + identity_tokens.unsqueeze(0)

        encoded_tokens = self.transformer(tokens)

        return self.layer_norm(encoded_tokens)


class DatasetSpecificAdapter(nn.Module):
    def __init__(self, num_numeric_features, num_categories):
        super().__init__()

        self.numeric_projection = nn.Linear(1, 64)
        self.category_embedding = nn.Embedding(num_categories, 64)

        self.feature_identity = nn.Embedding(
            num_numeric_features + 1, 64
        )

        self.adapter = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64)
        )

        self.layer_norm = nn.LayerNorm(64)

    def forward(self, specific_numeric, specific_categorical):

        numeric_tokens = self.numeric_projection(
            specific_numeric.unsqueeze(-1)
        )

        category_token = self.category_embedding(
            specific_categorical[:, 0]
        ).unsqueeze(1)

        tokens = torch.cat(
            [numeric_tokens, category_token], dim=1
        )

        feature_ids = torch.arange(
            tokens.size(1), device=tokens.device
        )

        tokens = tokens + self.feature_identity(
            feature_ids
        ).unsqueeze(0)

        adapted_tokens = self.adapter(tokens)

        residual_tokens = self.layer_norm(
            tokens + adapted_tokens
        )

        return residual_tokens


class AdaptiveResidualFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(64)

    def forward(self, shared_tokens, residual_tokens):

        shared_summary = shared_tokens.mean(
            dim=1, keepdim=True
        )

        shared_expanded = shared_summary.expand(
            -1, residual_tokens.size(1), -1
        )

        combined = torch.cat(
            [shared_expanded, residual_tokens], dim=-1
        )

        gate_values = self.gate(combined)

        fused_tokens = (
            gate_values * shared_expanded
            + (1 - gate_values) * residual_tokens
        )

        return self.layer_norm(fused_tokens), gate_values


class FinalTransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True, norm_first=True, activation='gelu'
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=2
        )

        self.layer_norm = nn.LayerNorm(64)

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)
        )

    def forward(self, shared_tokens, fused_tokens):

        combined_tokens = torch.cat(
            [shared_tokens, fused_tokens], dim=1
        )

        encoded_tokens = self.transformer(combined_tokens)
        encoded_tokens = self.layer_norm(encoded_tokens)

        pooled_representation = encoded_tokens.mean(dim=1)

        logits = self.classifier(pooled_representation)

        return logits


class AttackCategoryClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)


class NOVELIDSModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.shared_encoder = SharedFeatureEncoder()

        self.unsw_adapter = DatasetSpecificAdapter(36, 8)
        self.nsl_adapter = DatasetSpecificAdapter(34, 12)

        self.unsw_fusion = AdaptiveResidualFusion()
        self.nsl_fusion = AdaptiveResidualFusion()

        self.final_classifier = FinalTransformerClassifier()

    def forward(
        self,
        shared_numeric,
        shared_categorical,
        specific_numeric,
        specific_categorical,
        dataset
    ):

        shared_tokens = self.shared_encoder(
            shared_numeric, shared_categorical
        )

        if dataset == "UNSW":
            residual_tokens = self.unsw_adapter(
                specific_numeric, specific_categorical
            )
            fused_tokens, gates = self.unsw_fusion(
                shared_tokens, residual_tokens
            )

        elif dataset == "NSL":
            residual_tokens = self.nsl_adapter(
                specific_numeric, specific_categorical
            )
            fused_tokens, gates = self.nsl_fusion(
                shared_tokens, residual_tokens
            )

        else:
            raise ValueError(
                "Dataset must be 'UNSW' or 'NSL'."
            )

        logits = self.final_classifier(
            shared_tokens, fused_tokens
        )

        return logits, shared_tokens, gates
