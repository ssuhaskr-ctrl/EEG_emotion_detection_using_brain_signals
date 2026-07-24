
"""
cnn_lstm_model.py

Hybrid CNN + LSTM Model for GAMEEMO EEG Emotion Recognition

Input:
    (batch_size, 14, 256)

Output:
    4 emotion classes
"""

import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):

    def __init__(
        self,
        num_channels=14,
        hidden_size=128,
        num_layers=2,
        num_classes=4,
        dropout=0.5
    ):

        super(CNNLSTMModel, self).__init__()

        # -------------------------------------------------
        # CNN Feature Extractor
        # -------------------------------------------------

        self.conv1 = nn.Conv1d(
            in_channels=num_channels,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm1d(32)

        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm1d(64)

        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm1d(128)

        self.pool3 = nn.MaxPool1d(2)

        self.relu = nn.ReLU()

        # -------------------------------------------------
        # LSTM
        # -------------------------------------------------

        self.lstm = nn.LSTM(

            input_size=128,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True,

            dropout=dropout,

            bidirectional=True

        )

        # -------------------------------------------------
        # Classifier
        # -------------------------------------------------

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_size * 2, 128)

        self.fc2 = nn.Linear(128, 64)

        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):

        # x
        # (batch,14,256)

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.pool1(x)

        x = self.conv2(x)

        x = self.bn2(x)

        x = self.relu(x)

        x = self.pool2(x)

        x = self.conv3(x)

        x = self.bn3(x)

        x = self.relu(x)

        x = self.pool3(x)

        # Shape:
        # (batch,128,32)

        x = x.permute(0, 2, 1)

        # Shape:
        # (batch,32,128)

        output, (hidden, cell) = self.lstm(x)

        x = output[:, -1, :]

        x = self.dropout(x)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc3(x)

        return x


if __name__ == "__main__":

    print("=" * 60)

    print("CNN + LSTM MODEL")

    print("=" * 60)

    model = CNNLSTMModel()

    print(model)

    print()

    x = torch.randn(32, 14, 256)

    y = model(x)

    print("Input Shape  :", x.shape)

    print("Output Shape :", y.shape)

    print()

    total = sum(

        p.numel()

        for p in model.parameters()

    )

    trainable = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )

    print("Total Parameters     :", total)

    print("Trainable Parameters :", trainable)

    print("=" * 60)

