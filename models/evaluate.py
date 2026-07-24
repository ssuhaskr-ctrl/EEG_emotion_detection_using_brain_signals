"""
evaluate.py

Evaluation utilities for GAMEEMO Deep Learning Models
"""

import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class Evaluator:

    def __init__(self, device):

        self.device = device

    def evaluate(self, model, dataloader):

        model.eval()

        predictions = []

        labels = []

        with torch.no_grad():

            for x, y in dataloader:

                x = x.to(self.device)

                y = y.to(self.device)

                outputs = model(x)

                _, pred = torch.max(outputs, 1)

                predictions.extend(pred.cpu().numpy())

                labels.extend(y.cpu().numpy())

        predictions = np.array(predictions)

        labels = np.array(labels)

        accuracy = accuracy_score(labels, predictions)

        precision = precision_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        )

        cm = confusion_matrix(labels, predictions)

        report = classification_report(
            labels,
            predictions,
            zero_division=0,
        )

        results = {

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "confusion_matrix": cm,

            "classification_report": report,

        }

        return results

    def print_results(self, results):

        print("=" * 60)

        print("MODEL EVALUATION")

        print("=" * 60)

        print(f"Accuracy  : {results['accuracy']*100:.2f}%")

        print(f"Precision : {results['precision']*100:.2f}%")

        print(f"Recall    : {results['recall']*100:.2f}%")

        print(f"F1 Score  : {results['f1']*100:.2f}%")

        print()

        print("Classification Report")

        print(results["classification_report"])

        print("Confusion Matrix")

        print(results["confusion_matrix"])

        print("=" * 60)


def evaluate_model(model, test_loader, device):

    evaluator = Evaluator(device)

    results = evaluator.evaluate(model, test_loader)

    evaluator.print_results(results)

    return results


if __name__ == "__main__":

    print("=" * 60)

    print("Evaluation module loaded successfully.")

    print("Use evaluate_model(model, test_loader, device)")

    print("=" * 60)