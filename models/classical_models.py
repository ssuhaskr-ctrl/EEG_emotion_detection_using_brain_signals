"""
classical_models.py

Train and evaluate classical ML models for GAMEEMO
"""

import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression


class ClassicalModels:

    def __init__(self):

        self.models = {

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                ),

            "SVM":
                SVC(
                    kernel="rbf",
                    C=1,
                    gamma="scale"
                ),

            "KNN":
                KNeighborsClassifier(
                    n_neighbors=5
                ),

            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42
                ),

            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000
                )

        }

        self.best_model = None

    def train(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=0.2,

            random_state=42,

            stratify=y

        )

        print("=" * 60)

        print("Training Samples :", len(X_train))

        print("Testing Samples  :", len(X_test))

        print("=" * 60)

        scores = {}

        best_acc = 0

        for name, model in self.models.items():

            print("\nTraining", name)

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            acc = accuracy_score(y_test, pred)

            scores[name] = acc

            print("Accuracy :", round(acc * 100, 2), "%")

            print()

            print(classification_report(y_test, pred))

            print("Confusion Matrix")

            print(confusion_matrix(y_test, pred))

            print()

            if acc > best_acc:

                best_acc = acc

                self.best_model = model

        print("=" * 60)

        print("FINAL RESULTS")

        print("=" * 60)

        for k, v in scores.items():

            print(k, ":", round(v * 100, 2), "%")

        print()

        print("Best Accuracy :", round(best_acc * 100, 2), "%")

        return scores

    def save_best_model(self):

        joblib.dump(

            self.best_model,

            "results/best_model.pkl"

        )

        print()

        print("Best model saved")

        print("results/best_model.pkl")

    def load_model(self):

        model = joblib.load(

            "results/best_model.pkl"

        )

        return model