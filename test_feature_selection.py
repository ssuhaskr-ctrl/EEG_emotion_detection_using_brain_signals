"""
feature_selection.py
==========================================================
Feature Selection Module for GAMEEMO

Methods:
1. Variance Threshold
2. SelectKBest (ANOVA F-test)
3. Standard Scaling
4. Save/Load selector

Input:
    X -> (n_samples, n_features)

Output:
    X_selected -> (n_samples, k)

Author: GAMEEMO Project
==========================================================
"""

import os
import joblib
import numpy as np

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler


# ==========================================================
# Feature Selector Class
# ==========================================================

class FeatureSelector:

    def __init__(
        self,
        variance_threshold=0.0,
        k_features=150,
        save_path="results/feature_selector.pkl"
    ):

        self.variance_threshold = variance_threshold
        self.k_features = k_features
        self.save_path = save_path

        self.scaler = StandardScaler()

        self.var_selector = VarianceThreshold(
            threshold=self.variance_threshold
        )

        self.kbest_selector = SelectKBest(
            score_func=f_classif,
            k=self.k_features
        )

        self.fitted = False

    # ------------------------------------------------------

    def fit(self, X, y):

        print("\n===================================")
        print("Fitting Feature Selector")
        print("===================================")

        print("Original Shape :", X.shape)

        X_scaled = self.scaler.fit_transform(X)

        X_var = self.var_selector.fit_transform(X_scaled)

        print("After Variance :", X_var.shape)

        self.kbest_selector.fit(X_var, y)

        self.fitted = True

        print("Feature selector fitted.")

        return self

    # ------------------------------------------------------

    def transform(self, X):

        if not self.fitted:
            raise RuntimeError("Call fit() first.")

        X_scaled = self.scaler.transform(X)

        X_var = self.var_selector.transform(X_scaled)

        X_selected = self.kbest_selector.transform(X_var)

        return X_selected.astype(np.float32)

    # ------------------------------------------------------

    def fit_transform(self, X, y):

        self.fit(X, y)

        return self.transform(X)

    # ------------------------------------------------------

    def save(self):

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        joblib.dump(
            {
                "scaler": self.scaler,
                "var_selector": self.var_selector,
                "kbest_selector": self.kbest_selector,
            },
            self.save_path,
        )

        print(f"\nFeature selector saved to:\n{self.save_path}")

    # ------------------------------------------------------

    def load(self):

        data = joblib.load(self.save_path)

        self.scaler = data["scaler"]
        self.var_selector = data["var_selector"]
        self.kbest_selector = data["kbest_selector"]

        self.fitted = True

        print(f"\nFeature selector loaded from:\n{self.save_path}")


# ==========================================================
# Helper Function
# ==========================================================

def select_features(
    X,
    y,
    k_features=150,
):

    selector = FeatureSelector(
        variance_threshold=0.0,
        k_features=k_features,
    )

    X_selected = selector.fit_transform(X, y)

    return X_selected, selector


# ==========================================================
# Feature Dimension Utility
# ==========================================================

def print_feature_info(original, selected):

    print("\n===================================")
    print("FEATURE SELECTION SUMMARY")
    print("===================================")

    print("Original Features :", original.shape[1])
    print("Selected Features :", selected.shape[1])

    reduction = 100 * (
        original.shape[1] - selected.shape[1]
    ) / original.shape[1]

    print("Reduction         : {:.2f}%".format(reduction))

    print("Final Shape       :", selected.shape)

    print("===================================\n")


# ==========================================================
# Standalone Test
# ==========================================================

if __name__ == "__main__":

    np.random.seed(42)

    X = np.random.randn(1000, 522)

    y = np.random.randint(0, 4, 1000)

    X_selected, selector = select_features(
        X,
        y,
        k_features=150,
    )

    print_feature_info(
        X,
        X_selected,
    )

    selector.save()

    selector.load()

    X2 = selector.transform(X)

    print("Transformation successful.")
    print("Output Shape :", X2.shape)