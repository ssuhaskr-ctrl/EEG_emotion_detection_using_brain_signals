

"""
lstm_model.py

LSTM model for GAMEEMO EEG Emotion Recognition

Input Shape:
    (batch_size, 14, 256)

Output:
    4 emotion classes
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size=256,
        hidden_size=128,
        num_layers=2,
        num_classes=4,
        dropout=0.3
    ):

        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(

            input_size=input_size,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True,

            dropout=dropout,

            bidirectional=True

        )

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_size * 2, 128)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(128, 64)

        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):

        # x shape
        # (batch, 14, 256)

        output, (hidden, cell) = self.lstm(x)

        # Take last time step

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

    model = LSTMModel()

    print("=" * 60)

    print("LSTM MODEL")

    print("=" * 60)

    print(model)

    print()

    x = torch.randn(32, 14, 256)

    y = model(x)

    print("Input Shape  :", x.shape)

    print("Output Shape :", y.shape)

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

    print("=" * 60)

