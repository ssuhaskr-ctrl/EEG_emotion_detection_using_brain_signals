
"""
transformer_model.py

Transformer Model for GAMEEMO EEG Emotion Recognition

Input Shape:
    (batch_size, 14, 256)

Output:
    4 emotion classes
"""

import torch
import torch.nn as nn


class TransformerModel(nn.Module):

    def __init__(
        self,
        input_dim=256,
        seq_length=14,
        num_classes=4,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.2
    ):

        super(TransformerModel, self).__init__()

        self.input_projection = nn.Linear(
            input_dim,
            d_model
        )

        self.position_embedding = nn.Parameter(
            torch.randn(1, seq_length, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=nhead,

            dim_feedforward=dim_feedforward,

            dropout=dropout,

            batch_first=True,

            activation="gelu"

        )

        self.transformer_encoder = nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers

        )

        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(

            nn.Linear(d_model, 256),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(128, num_classes)

        )

    def forward(self, x):

        # x shape
        # (batch,14,256)

        x = self.input_projection(x)

        x = x + self.position_embedding

        x = self.transformer_encoder(x)

        x = self.norm(x)

        # Global Average Pooling

        x = torch.mean(x, dim=1)

        x = self.dropout(x)

        x = self.classifier(x)

        return x


if __name__ == "__main__":

    print("=" * 60)

    print("TRANSFORMER MODEL")

    print("=" * 60)

    model = TransformerModel()

    print(model)

    print()

    dummy = torch.randn(

        32,

        14,

        256

    )

    output = model(dummy)

    print("Input Shape  :", dummy.shape)

    print("Output Shape :", output.shape)

    print()

    total_params = sum(

        p.numel()

        for p in model.parameters()

    )

    trainable_params = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )

    print("Total Parameters     :", total_params)

    print("Trainable Parameters :", trainable_params)

    print()

    print("=" * 60)

    print("Forward Pass Successful")

    print("=" * 60)

