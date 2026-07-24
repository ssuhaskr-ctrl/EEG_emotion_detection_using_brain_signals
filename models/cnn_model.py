
"""
cnn_model.py

1D CNN model for GAMEEMO EEG Emotion Recognition
Input  : (batch_size, 14, 256)
Output : 4 emotion classes
"""

import torch
import torch.nn as nn


class CNNModel(nn.Module):

    def __init__(self, num_channels=14, num_classes=4):

        super(CNNModel, self).__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=num_channels,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)

        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(64, num_classes)

        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


if __name__ == "__main__":

    model = CNNModel()

    print(model)

    x = torch.randn(32, 14, 256)

    y = model(x)

    print()

    print("Input Shape :", x.shape)

    print("Output Shape:", y.shape)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()

    print("Total Parameters     :", total_params)

    print("Trainable Parameters :", trainable_params)
