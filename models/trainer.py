
"""
trainer.py

Generic PyTorch trainer for GAMEEMO models.
"""

import os
import copy
import torch
import torch.nn as nn


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion=None,
        optimizer=None,
        scheduler=None,
        device=None,
        save_path="results/best_model.pth"
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = model.to(self.device)

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.criterion = criterion or nn.CrossEntropyLoss()

        self.optimizer = optimizer or torch.optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

        self.scheduler = scheduler

        self.save_path = save_path

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # --------------------------------------------------

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        for X, y in self.train_loader:

            X = X.to(self.device)

            y = y.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(X)

            loss = self.criterion(outputs, y)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item() * X.size(0)

            _, predicted = torch.max(outputs, 1)

            total += y.size(0)

            correct += (predicted == y).sum().item()

        epoch_loss = running_loss / total

        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    # --------------------------------------------------

    def validate(self):

        self.model.eval()

        running_loss = 0.0

        correct = 0

        total = 0

        with torch.no_grad():

            for X, y in self.val_loader:

                X = X.to(self.device)

                y = y.to(self.device)

                outputs = self.model(X)

                loss = self.criterion(outputs, y)

                running_loss += loss.item() * X.size(0)

                _, predicted = torch.max(outputs, 1)

                total += y.size(0)

                correct += (predicted == y).sum().item()

        loss = running_loss / total

        acc = correct / total

        return loss, acc

    # --------------------------------------------------

    def fit(self, epochs=50):

        best_acc = 0.0

        best_weights = copy.deepcopy(
            self.model.state_dict()
        )

        print("=" * 60)
        print("Training Started")
        print("Device :", self.device)
        print("=" * 60)

        for epoch in range(epochs):

            train_loss, train_acc = self.train_one_epoch()

            val_loss, val_acc = self.validate()

            if self.scheduler is not None:

                try:
                    self.scheduler.step(val_loss)
                except:
                    self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch+1}/{epochs}"
            )

            print(
                f"Train Loss : {train_loss:.4f}"
            )

            print(
                f"Train Acc  : {train_acc*100:.2f}%"
            )

            print(
                f"Val Loss   : {val_loss:.4f}"
            )

            print(
                f"Val Acc    : {val_acc*100:.2f}%"
            )

            print("-" * 50)

            if val_acc > best_acc:

                best_acc = val_acc

                best_weights = copy.deepcopy(
                    self.model.state_dict()
                )

                torch.save(
                    best_weights,
                    self.save_path
                )

                print(
                    "Best model saved."
                )

        self.model.load_state_dict(
            best_weights
        )

        print("=" * 60)
        print("Training Finished")
        print(
            f"Best Validation Accuracy : {best_acc*100:.2f}%"
        )
        print("=" * 60)

        return self.history

    # --------------------------------------------------

    def load_best_model(self):

        state = torch.load(
            self.save_path,
            map_location=self.device
        )

        self.model.load_state_dict(state)

        self.model.to(self.device)

        self.model.eval()

        return self.model

    # --------------------------------------------------

    def predict(self, loader):

        self.model.eval()

        predictions = []

        with torch.no_grad():

            for X, _ in loader:

                X = X.to(self.device)

                outputs = self.model(X)

                _, pred = torch.max(outputs, 1)

                predictions.extend(
                    pred.cpu().numpy()
                )

        return predictions


# ------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)

    print("GAMEEMO TRAINER")

    print("=" * 60)

    print()

    print("This module is intended to be")

    print("used by train_deep.py")

    print()

    print("Example:")

    print()

    print("trainer = Trainer(model,")

    print("                  train_loader,")

    print("                  val_loader)")

    print()

    print("trainer.fit(epochs=50)")

    print()

    print("=" * 60)

